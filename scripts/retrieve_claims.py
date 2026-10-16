#!/usr/bin/env python3
"""Local, zero-LLM retrieval over ArxAgent's FalkorDB graph."""
import argparse
import json
import os
import re
import sys
from collections import OrderedDict

from falkordb import FalkorDB


def get_rows(result):
    """Handle the result_set exposed by current FalkorDB Python clients."""
    rows = getattr(result, "result_set", None)
    if rows is None:
        raise RuntimeError("FalkorDB query result has no result_set attribute")
    return rows


def terms_for(text):
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
        if token.lower() not in {
            "the", "and", "for", "that", "with", "from", "this", "what",
            "which", "does", "how", "why", "are", "was", "were", "into",
            "about", "paper", "papers", "research", "claim", "claims",
        }
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve claims and their evidence from FalkorDB without an LLM."
    )
    parser.add_argument("--graph", default="LLMStability")
    parser.add_argument("--query", required=True, help="Research question or keywords")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    db = FalkorDB(
        host=os.getenv("FALKORDB_HOST", "localhost"),
        port=int(os.getenv("FALKORDB_PORT", "6379")),
        password=os.getenv("FALKORDB_PASSWORD") or None,
    )
    graph = db.select_graph(args.graph)

    cypher = """
    MATCH (p:Paper)-[:REPORTS]->(c:Claim)
    OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(e:Evidence)
    RETURN p.paper_id, p.title, c.claim_id, c.text, c.claim_type,
           c.epistemic_status, c.confidence,
           e.quote, e.page, e.section, e.chunk_id
    """
    try:
        rows = get_rows(graph.query(cypher))
    except Exception as exc:
        print(f"FalkorDB query failed: {exc}", file=sys.stderr)
        return 2

    claims = OrderedDict()
    for row in rows:
        # RETURN columns are in the same order as the query above.
        paper_id, paper_title, claim_id, text, claim_type, status, confidence = row[:7]
        key = str(claim_id)
        if key not in claims:
            claims[key] = {
                "paper_id": paper_id,
                "paper_title": paper_title,
                "claim_id": claim_id,
                "statement": text,
                "claim_type": claim_type,
                "epistemic_status": status,
                "confidence": confidence,
                "evidence": [],
            }
        quote, page, section, chunk_id = row[7:11]
        if quote is not None:
            evidence = {
                "quote": quote,
                "page": page,
                "section": section,
                "chunk_id": chunk_id,
            }
            if evidence not in claims[key]["evidence"]:
                claims[key]["evidence"].append(evidence)

    query_terms = terms_for(args.query)
    for claim in claims.values():
        haystack = " ".join(
            [
                str(claim.get("paper_title") or ""),
                str(claim.get("statement") or ""),
                str(claim.get("claim_type") or ""),
                " ".join(str(e.get("quote") or "") for e in claim["evidence"]),
            ]
        ).lower()
        claim["retrieval_score"] = sum(haystack.count(term) for term in query_terms)

    ranked = sorted(
        claims.values(),
        key=lambda claim: (claim["retrieval_score"], len(claim["evidence"])),
        reverse=True,
    )
    matched = [claim for claim in ranked if claim["retrieval_score"] > 0]
    selected = (matched if matched else ranked)[:args.limit]

    print(f"Graph: {args.graph}")
    print(f"Question: {args.query}")
    print(f"Claims in graph: {len(claims)}")
    print(f"Keyword matches: {len(matched)}")
    if not selected:
        print("No claims found. Check that the graph has been ingested.")
        return 0

    for index, claim in enumerate(selected, 1):
        print(f"\n{'=' * 78}")
        print(f"[{index}] {claim['paper_title']} ({claim['paper_id']})")
        print(f"Claim ID: {claim['claim_id']}")
        print(f"Type/status: {claim['claim_type']} / {claim['epistemic_status']}")
        print(f"Local keyword score: {claim['retrieval_score']}")
        print(f"Claim: {claim['statement']}")
        if claim["evidence"]:
            print("Evidence:")
            for evidence in claim["evidence"]:
                section = f", section: {evidence['section']}" if evidence.get("section") else ""
                chunk = f", chunk: {evidence['chunk_id']}" if evidence.get("chunk_id") else ""
                print(f"  - p. {evidence['page']}{section}{chunk}: {evidence['quote']}")
        else:
            print("Evidence: none linked in graph")
    print("\nNote: this is lexical keyword retrieval, not semantic search or an LLM-generated answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
