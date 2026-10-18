#!/usr/bin/env python3
"""Ingest validated ArxAgent per-chunk knowledge JSON files into FalkorDB.

This script makes no LLM/API calls. It only reads local JSON and writes to FalkorDB.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

from falkordb import FalkorDB


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:20]


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_minimum(data, path):
    errors = []
    if not isinstance(data, dict):
        return ["top-level JSON value must be an object"]
    for field in ("paper_id", "paper_title", "claims"):
        if field not in data:
            errors.append(f"missing top-level field: {field}")
    if not isinstance(data.get("claims"), list):
        errors.append("'claims' must be a list")
        return errors

    for i, claim in enumerate(data["claims"]):
        prefix = f"claim[{i}]"
        if not isinstance(claim, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for field in ("claim_type", "statement", "epistemic_status", "evidence"):
            if field not in claim:
                errors.append(f"{prefix} missing {field}")
        if not isinstance(claim.get("statement"), str) or not claim.get("statement", "").strip():
            errors.append(f"{prefix} statement must be a non-empty string")
        evidence = claim.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{prefix} must contain at least one evidence item")
        else:
            for j, ev in enumerate(evidence):
                if not isinstance(ev, dict) or not isinstance(ev.get("quote"), str) or not ev.get("quote", "").strip():
                    errors.append(f"{prefix} evidence[{j}] needs a non-empty quote")
                if not isinstance(ev, dict) or not isinstance(ev.get("page"), int) or ev.get("page", 0) < 1:
                    errors.append(f"{prefix} evidence[{j}] needs a positive integer page")
    return errors


def query(graph, cypher, params):
    graph.query(cypher, params)


def main():
    parser = argparse.ArgumentParser(description="Ingest local ArxAgent knowledge JSON into FalkorDB.")
    parser.add_argument("--graph", default="LLMStability")
    parser.add_argument("--input", default="data/knowledge/by_chunk",
                        help="Root directory containing per-chunk JSON files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and count files without connecting to FalkorDB")
    args = parser.parse_args()

    pattern = str(Path(args.input) / "**" / "*.json")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        print(f"No JSON files found under: {args.input}", file=sys.stderr)
        return 2

    valid_records = []
    invalid_count = 0
    for path in paths:
        try:
            data = read_json(path)
            errors = validate_minimum(data, path)
            if errors:
                invalid_count += 1
                print(f"INVALID {path}")
                for error in errors:
                    print(f"  - {error}")
                continue
            valid_records.append((path, data))
        except Exception as exc:
            invalid_count += 1
            print(f"INVALID {path}: {exc}")

    print(f"Files found: {len(paths)}")
    print(f"Valid files: {len(valid_records)}")
    print(f"Invalid files: {invalid_count}")
    print(f"Claims to ingest: {sum(len(data['claims']) for _, data in valid_records)}")
    if args.dry_run:
        print("Dry run only; FalkorDB was not contacted.")
        return 0

    db = FalkorDB(
        host=os.getenv("FALKORDB_HOST", "localhost"),
        port=int(os.getenv("FALKORDB_PORT", "6379")),
        password=os.getenv("FALKORDB_PASSWORD") or None,
    )
    graph = db.select_graph(args.graph)

    imported_claims = 0
    imported_evidence = 0
    for path, paper in valid_records:
        pid = str(paper["paper_id"])
        query(
            graph,
            "MERGE (p:Paper {paper_id: $id}) SET p.title = $title",
            {"id": pid, "title": str(paper["paper_title"])},
        )

        for index, claim in enumerate(paper["claims"]):
            cid = str(claim.get("claim_id") or stable_id(pid + claim["statement"]))
            query(
                graph,
                """MATCH (p:Paper {paper_id: $pid})
                   MERGE (c:Claim {claim_id: $cid})
                   SET c.text = $text,
                       c.claim_type = $claim_type,
                       c.epistemic_status = $status,
                       c.confidence = $confidence,
                       c.experiment_json = $experiment_json,
                       c.limitations_json = $limitations_json,
                       c.source_file = $source_file
                   MERGE (p)-[:REPORTS]->(c)""",
                {
                    "pid": pid,
                    "cid": cid,
                    "text": claim["statement"],
                    "claim_type": claim["claim_type"],
                    "status": claim["epistemic_status"],
                    "confidence": float(claim.get("confidence", 0.5)),
                    "experiment_json": json.dumps(claim.get("experiment", {}), ensure_ascii=False),
                    "limitations_json": json.dumps(claim.get("limitations", []), ensure_ascii=False),
                    "source_file": str(Path(path)),
                },
            )
            imported_claims += 1

            # Model each evidence item independently so no quotes/pages are lost.
            for ev_index, ev in enumerate(claim["evidence"]):
                quote = ev["quote"]
                page = int(ev["page"])
                evidence_id = stable_id(f"{cid}|{page}|{quote}")
                query(
                    graph,
                    """MATCH (c:Claim {claim_id: $cid})
                       MERGE (e:Evidence {evidence_id: $evidence_id})
                       SET e.quote = $quote,
                           e.page = $page,
                           e.section = $section,
                           e.chunk_id = $chunk_id
                       MERGE (c)-[:SUPPORTED_BY]->(e)""",
                    {
                        "cid": cid,
                        "evidence_id": evidence_id,
                        "quote": quote,
                        "page": page,
                        "section": ev.get("section"),
                        "chunk_id": ev.get("chunk_id"),
                    },
                )
                imported_evidence += 1

            # Keep the semantic tags as graph nodes/relationships.
            for field, label, relation in (
                ("failure_modes", "FailureMode", "DESCRIBES_FAILURE"),
                ("interventions", "Intervention", "TESTS_OR_PROPOSES"),
                ("metrics", "Metric", "MEASURES"),
            ):
                for raw_name in claim.get(field, []) or []:
                    name = str(raw_name).strip()
                    if not name:
                        continue
                    # label/relation come only from the fixed tuples above.
                    query(
                        graph,
                        f"""MATCH (c:Claim {{claim_id: $cid}})
                            MERGE (n:{label} {{name: $name}})
                            MERGE (c)-[:{relation}]->(n)""",
                        {"cid": cid, "name": name},
                    )

        print(f"Imported {pid}: {len(paper['claims'])} claims ({path})")

    print(f"Done. Graph: {args.graph}")
    print(f"Claims processed: {imported_claims}")
    print(f"Evidence items processed: {imported_evidence}")
    if invalid_count:
        print(f"Skipped invalid files: {invalid_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
