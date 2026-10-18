"""Write one extraction JSON into the LLMStability graph (same schema as GraphTools reads)."""
import re

CONCEPT_ALIASES = {  # normalized-key -> canonical name; extend from your graph
    "deepnorm": "DeepNorm", "deep norm": "DeepNorm",
    "deepnorm normalization function": "DeepNorm",
    "qk norm": "QK-norm", "query key normalization": "QK-norm",
    "z loss": "z-loss", "layernorm": "LayerNorm",
    "layer normalization": "LayerNorm", "rmsnorm": "RMSNorm",
    "root mean square layer normalization": "RMSNorm",
    "pre ln": "Pre-LN", "post ln": "Post-LN",
}

import graph_tools
def canon_concept(raw) -> str:
    name = raw.get("name", "") if isinstance(raw, dict) else str(raw)
    name = " ".join(name.split())
    key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return CONCEPT_ALIASES.get(key, name)


def _first(d: dict, *keys, default=None):
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return default


# (extraction key, node label, relationship). Fixed constants, never user input.
CONCEPT_LINKS = (
    ("interventions", "Intervention", "TESTS_OR_PROPOSES"),
    ("failure_modes", "FailureMode", "DESCRIBES_FAILURE"),
    ("metrics", "Metric", "MEASURES"),
)


def load_extraction(graph, paper_id: str, extraction: dict, *, title: str,
                    arxiv_id: str | None = None, scope: str = "abstract") -> dict:
    graph.query(
        "MERGE (p:Paper {paper_id: $pid}) "
        "SET p.title = $title, p.arxiv_id = $aid, p.source_scope = $scope",
        {"pid": paper_id, "title": title, "aid": arxiv_id, "scope": scope},
    )
    added = 0
    for i, c in enumerate(extraction.get("claims", []), start=1):
        evidence = c.get("evidence") or []
        chunk_id = (evidence[0].get("chunk_id") if evidence
                    else f"{paper_id}_C0000")
        # GUESS: field names. Match them to your extractor's JSON.
        claim_id = _first(c, "claim_id", "id", default=f"{chunk_id}_CLAIM_{i:03d}")
        graph.query(
            """
            MATCH (p:Paper {paper_id: $pid})
            MERGE (c:Claim {claim_id: $cid})
            SET c.text = $text, c.claim_type = $ctype,
                c.epistemic_status = $status, c.confidence = $conf,
                c.source_scope = $scope
            MERGE (p)-[:REPORTS]->(c)
            """,
            {"pid": paper_id, "cid": claim_id,
             "text": _first(c, "text", "claim", "statement", default=""),
             "ctype": c.get("claim_type"),
             "status": c.get("epistemic_status"),
             "conf": c.get("confidence"), "scope": scope},
        )
        for key, label, rel in CONCEPT_LINKS:
            for raw in c.get(key) or []:
                name = canon_concept(raw)
                if name:
                    graph.query(
                        f"MATCH (c:Claim {{claim_id: $cid}}) "
                        f"MERGE (x:{label} {{name: $name}}) "
                        f"MERGE (c)-[:{rel}]->(x)",
                        {"cid": claim_id, "name": name},
                    )
        for j, ev in enumerate(evidence, start=1):
            graph.query(
                """
                MATCH (c:Claim {claim_id: $cid})
                MERGE (e:Evidence {evidence_id: $eid})
                SET e.quote = $quote, e.page = $page,
                    e.section = $section, e.chunk_id = $chunk
                MERGE (c)-[:SUPPORTED_BY]->(e)
                """,
                {"cid": claim_id, "eid": f"{claim_id}_EV_{j}",
                 "quote": ev.get("quote"), "page": ev.get("page"),
                 "section": ev.get("section"), "chunk": ev.get("chunk_id")},
            )
        added += 1
    return {"claims_added": added, "scope": scope}