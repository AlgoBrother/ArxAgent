"""Safe, read-only graph tools for ArxAgent. No LLM/API calls."""
import os
import re
from typing import Any

from falkordb import FalkorDB


DEFAULT_GRAPH = "LLMStability"

STOPWORDS = {
    "the", "and", "for", "that", "with", "how", "does", "what",
    "from", "this", "into", "are", "was", "were", "can", "could",
    "would", "should", "about", "cause", "causes", "causing",
    "research", "paper", "papers", "method", "methods", "show",
    "explain", "evidence", "find", "tell", "please",
}

# Symptom vocabulary: words a user might type -> substrings to match in
# FailureMode names. Extend these as your graph grows.
SYMPTOM_GROUPS = [
    {"triggers": ("spike", "spiked", "spiking", "diverg", "blow up", "blew up",
                  "instabil", "unstable", "loss jump"),
     "name_terms": ("spike", "diverg", "instabil", "unstable")},
    {"triggers": ("hardware", "node fail", "gpu fail", "crash", "preempt"),
     "name_terms": ("hardware", "node", "crash")},
    {"triggers": ("fp16", "bf16", "fp8", "precision", "overflow", "underflow"),
     "name_terms": ("precision", "overflow", "underflow", "fp16", "bf16", "fp8")},
    {"triggers": ("attention", "logit", "entropy collapse"),
     "name_terms": ("attention", "logit", "entropy")},
    {"triggers": ("gradient", "exploding", "vanishing", "grad norm"),
     "name_terms": ("gradient", "exploding", "vanishing")},
    {"triggers": ("deep", "depth", "layernorm", "layer norm", "residual"),
     "name_terms": ("depth", "deep", "normaliz")},
]

RANKING_NOTE = (
    "candidate_fixes are ordered by how many distinct papers discuss them, NOT "
    "by effectiveness. Read each report's statement: some report a fix that "
    "did not work or only partly worked."
)


def expand_symptom(text: str) -> list[str]:
    """Map a free-text symptom to FailureMode name substrings (no LLM)."""
    low = text.lower()
    match_terms: set[str] = set()
    for group in SYMPTOM_GROUPS:
        if any(t in low for t in group["triggers"]):
            match_terms.update(group["name_terms"])
    if not match_terms:  # unknown symptom: fall back to the user's own words
        match_terms = {w for w in terms(text) if len(w) >= 4}
    return sorted(match_terms)


def terms(text: str) -> set[str]:
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
        if word.lower() not in STOPWORDS
    }


class GraphTools:
    """Fixed-query, parameterized, read-only operations on FalkorDB."""

    _DIAGNOSE_CYPHER = """
        MATCH (p:Paper)-[:REPORTS]->(c:Claim)-[:DESCRIBES_FAILURE]->(f:FailureMode)
        WHERE any(t IN $terms WHERE toLower(f.name) CONTAINS t)
        OPTIONAL MATCH (c)-[:TESTS_OR_PROPOSES]->(i:Intervention)
        OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(e:Evidence)
        RETURN p.paper_id, p.title, c.claim_id, c.text, c.claim_type,
               c.epistemic_status, c.source_scope, c.source_file, f.name,
               collect(DISTINCT i.name) AS interventions,
               collect(DISTINCT {quote: e.quote, page: e.page,
                                 section: e.section, chunk_id: e.chunk_id}) AS evidence
    """



    def __init__(self, graph_name: str = DEFAULT_GRAPH):
        db = FalkorDB(
            host=os.getenv("FALKORDB_HOST", "localhost"),
            port=int(os.getenv("FALKORDB_PORT", "6379")),
            password=os.getenv("FALKORDB_PASSWORD") or None,
        )
        self.graph = db.select_graph(graph_name)

    def _query(self, cypher: str, params: dict | None = None) -> list:
        # All callers use fixed Cypher templates; user text is passed as parameters.
        result = self.graph.query(cypher, params or {})
        return result.result_set

    def diagnose(self, symptom: str, limit: int = 5) -> dict:
        """Symptom -> matching failure claims -> fixes grouped across papers."""
        if not isinstance(symptom, str) or not symptom.strip():
            raise ValueError("symptom must be a non-empty string")
        if not 1 <= limit <= 8:
            raise ValueError("limit must be between 1 and 8")

        match_terms = expand_symptom(symptom)
        rows = self._query(self._DIAGNOSE_CYPHER, {"terms": match_terms}) if match_terms else []

        claims: dict[str, dict] = {}
        for (paper_id, title, claim_id, claim_text, ctype, status, scope,
             source_file, failure, interventions, evidence) in rows:
            rec = claims.get(claim_id)
            if rec is None:
                evidence = [e for e in (evidence or [])
                            if isinstance(e, dict) and e.get("quote")]
                first = evidence[0] if evidence else {}
                provenance = ("abstract_only" if scope == "abstract"
                              else "hand_curated" if "_curated" in str(source_file or "")
                              else "llm_extracted")
                rec = claims[claim_id] = {
                    "claim_id": claim_id, "paper_id": paper_id,
                    "paper_title": title,
                    "statement": (claim_text or "")[:320],
                    "claim_type": ctype, "epistemic_status": status,
                    "provenance": provenance,
                    "interventions": [x for x in (interventions or []) if x],
                    "failure_modes": [],
                    "quote": (first.get("quote") or "")[:220],
                    "page": first.get("page"), "section": first.get("section"),
                }
            if failure and failure not in rec["failure_modes"]:
                rec["failure_modes"].append(failure)

        # Claims that name the symptom but no fix: the symptom as observed.
        observations = [c for c in claims.values() if not c["interventions"]]

        fixes: dict[str, dict] = {}
        for c in claims.values():
            for name in c["interventions"]:
                fix = fixes.setdefault(name, {"fix": name, "papers": set(), "reports": []})
                fix["papers"].add(c["paper_id"])
                fix["reports"].append({k: c[k] for k in (
                    "paper_id", "claim_id", "statement", "epistemic_status",
                    "provenance", "quote", "page")})
        ranked = sorted(fixes.values(),
                        key=lambda f: (-len(f["papers"]), -len(f["reports"]), f["fix"]))[:limit]
        for f in ranked:
            f["papers"] = sorted(f["papers"])
            f["paper_count"] = len(f["papers"])

        return {
            "symptom": symptom.strip(),
            "match_terms": match_terms,
            "matched_failure_modes": sorted(
                {fm for c in claims.values() for fm in c["failure_modes"]}),
            "papers_matched": sorted({c["paper_id"] for c in claims.values()}),
            "claims_matched": len(claims),
            "symptom_observations": observations[:limit],
            "candidate_fixes": ranked,
            "ranking_note": RANKING_NOTE,
        }

    def stats(self) -> dict[str, int]:
        rows = self._query("""
            MATCH (n)
            WITH count(n) AS nodes
            MATCH ()-[r]->()
            RETURN nodes, count(r) AS edges
        """)
        if not rows:
            return {"nodes": 0, "edges": 0}
        return {"nodes": int(rows[0][0]), "edges": int(rows[0][1])}

    def search_claims(self, question: str, limit: int = 8) -> list[dict]:
        """Rank graph claims lexically; this is a retrieval tool, not an answer."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")

        rows = self._query("""
            MATCH (p:Paper)-[:REPORTS]->(c:Claim)
            OPTIONAL MATCH (c)-[:TESTS_OR_PROPOSES]->(i:Intervention)
            OPTIONAL MATCH (c)-[:DESCRIBES_FAILURE]->(f:FailureMode)
            OPTIONAL MATCH (c)-[:MEASURES]->(m:Metric)
            OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(e:Evidence)
            RETURN p.paper_id, p.title, c.claim_id, c.text,
                   c.claim_type, c.epistemic_status, c.confidence,
                   collect(DISTINCT i.name) AS interventions,
                   collect(DISTINCT f.name) AS failure_modes,
                   collect(DISTINCT m.name) AS metrics,
                   collect(DISTINCT {
                       quote: e.quote, page: e.page,
                       section: e.section, chunk_id: e.chunk_id
                   }) AS evidence
        """)

        query_terms = terms(question)
        ranked = []

        for row in rows:
            (paper_id, title, claim_id, claim_text, claim_type, status,
             confidence, interventions, failure_modes, metrics, evidence) = row

            evidence = [
                item for item in (evidence or [])
                if isinstance(item, dict) and item.get("quote")
            ]
            interventions = [x for x in (interventions or []) if x]
            failure_modes = [x for x in (failure_modes or []) if x]
            metrics = [x for x in (metrics or []) if x]

            searchable = " ".join([
                str(title or ""), str(claim_text or ""),
                " ".join(map(str, interventions)),
                " ".join(map(str, failure_modes)),
                " ".join(map(str, metrics)),
                " ".join(item["quote"] for item in evidence),
            ])
            matched = query_terms & terms(searchable)
            score = len(matched)

            if score:
                ranked.append({
                    "paper_id": paper_id,
                    "paper_title": title,
                    "claim_id": claim_id,
                    "claim": claim_text,
                    "claim_type": claim_type,
                    "epistemic_status": status,
                    "confidence": confidence,
                    "interventions": interventions,
                    "failure_modes": failure_modes,
                    "metrics": metrics,
                    "evidence": evidence,
                    "matched_terms": sorted(matched),
                    "score": score,
                })

        ranked.sort(key=lambda item: (
            item["score"], len(item["evidence"])
        ), reverse=True)
        return ranked[:limit]

    def get_claim(self, claim_id: str) -> dict | None:
        """Fetch one claim, its paper, semantic links, and supporting evidence."""
        if not isinstance(claim_id, str) or not claim_id.strip():
            raise ValueError("claim_id must be a non-empty string")

        rows = self._query("""
            MATCH (p:Paper)-[:REPORTS]->(c:Claim {claim_id: $claim_id})
            OPTIONAL MATCH (c)-[:TESTS_OR_PROPOSES]->(i:Intervention)
            OPTIONAL MATCH (c)-[:DESCRIBES_FAILURE]->(f:FailureMode)
            OPTIONAL MATCH (c)-[:MEASURES]->(m:Metric)
            OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(e:Evidence)
            RETURN p.paper_id, p.title, c.claim_id, c.text,
                   c.claim_type, c.epistemic_status, c.confidence,
                   collect(DISTINCT i.name) AS interventions,
                   collect(DISTINCT f.name) AS failure_modes,
                   collect(DISTINCT m.name) AS metrics,
                   collect(DISTINCT {
                       quote: e.quote, page: e.page,
                       section: e.section, chunk_id: e.chunk_id
                   }) AS evidence
        """, {"claim_id": claim_id})

        if not rows:
            return None

        row = rows[0]
        return {
            "paper_id": row[0],
            "paper_title": row[1],
            "claim_id": row[2],
            "claim": row[3],
            "claim_type": row[4],
            "epistemic_status": row[5],
            "confidence": row[6],
            "interventions": [x for x in (row[7] or []) if x],
            "failure_modes": [x for x in (row[8] or []) if x],
            "metrics": [x for x in (row[9] or []) if x],
            "evidence": [
                item for item in (row[10] or [])
                if isinstance(item, dict) and item.get("quote")
            ],
        }

    def explore_concept(self, concept: str, limit: int = 10) -> list[dict]:
        """Find claims connected to a named intervention, failure mode, or metric."""
        if not isinstance(concept, str) or not concept.strip():
            raise ValueError("concept must be a non-empty string")
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")

        rows = self._query("""
            MATCH (p:Paper)-[:REPORTS]->(c:Claim)
            OPTIONAL MATCH (c)-[:TESTS_OR_PROPOSES]->(i:Intervention)
            OPTIONAL MATCH (c)-[:DESCRIBES_FAILURE]->(f:FailureMode)
            OPTIONAL MATCH (c)-[:MEASURES]->(m:Metric)
            WITH p, c, collect(DISTINCT i.name) AS interventions,
                 collect(DISTINCT f.name) AS failure_modes,
                 collect(DISTINCT m.name) AS metrics
            WHERE any(x IN interventions WHERE toLower(x) CONTAINS toLower($concept))
               OR any(x IN failure_modes WHERE toLower(x) CONTAINS toLower($concept))
               OR any(x IN metrics WHERE toLower(x) CONTAINS toLower($concept))
               OR toLower(c.text) CONTAINS toLower($concept)
            RETURN p.paper_id, p.title, c.claim_id, c.text,
                   c.claim_type, c.epistemic_status,
                   interventions, failure_modes, metrics
            LIMIT $limit
        """, {"concept": concept.strip(), "limit": limit})

        return [{
            "paper_id": row[0],
            "paper_title": row[1],
            "claim_id": row[2],
            "claim": row[3],
            "claim_type": row[4],
            "epistemic_status": row[5],
            "interventions": [x for x in (row[6] or []) if x],
            "failure_modes": [x for x in (row[7] or []) if x],
            "metrics": [x for x in (row[8] or []) if x],
        } for row in rows]

    def related_claims(self, claim_id: str, limit: int = 10) -> list[dict]:
        """Traverse shared interventions, failure modes, or metrics."""
        if not isinstance(claim_id, str) or not claim_id.strip():
            raise ValueError("claim_id must be a non-empty string")
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")

        rows = self._query("""
            MATCH (p1:Paper)-[:REPORTS]->(source:Claim {claim_id: $claim_id})
            OPTIONAL MATCH (source)-[:TESTS_OR_PROPOSES]->(si:Intervention)
            OPTIONAL MATCH (source)-[:DESCRIBES_FAILURE]->(sf:FailureMode)
            OPTIONAL MATCH (source)-[:MEASURES]->(sm:Metric)
            WITH source, collect(DISTINCT si.name) AS source_i,
                 collect(DISTINCT sf.name) AS source_f,
                 collect(DISTINCT sm.name) AS source_m
            MATCH (p:Paper)-[:REPORTS]->(other:Claim)
            WHERE other.claim_id <> source.claim_id
            OPTIONAL MATCH (other)-[:TESTS_OR_PROPOSES]->(oi:Intervention)
            OPTIONAL MATCH (other)-[:DESCRIBES_FAILURE]->(ofm:FailureMode)
            OPTIONAL MATCH (other)-[:MEASURES]->(om:Metric)
            WITH p, other, source,
                 source_i, source_f, source_m,
                 collect(DISTINCT oi.name) AS other_i,
                 collect(DISTINCT ofm.name) AS other_f,
                 collect(DISTINCT om.name) AS other_m
            WITH p, other,
                 [x IN source_i WHERE x IN other_i] +
                 [x IN source_f WHERE x IN other_f] +
                 [x IN source_m WHERE x IN other_m] AS shared
            WHERE size(shared) > 0
            RETURN p.paper_id, p.title, other.claim_id, other.text,
                   other.claim_type, other.epistemic_status, shared
            ORDER BY size(shared) DESC
            LIMIT $limit
        """, {"claim_id": claim_id, "limit": limit})

        return [{
            "paper_id": row[0],
            "paper_title": row[1],
            "claim_id": row[2],
            "claim": row[3],
            "claim_type": row[4],
            "epistemic_status": row[5],
            "shared_concepts": row[6] or [],
        } for row in rows]


def main():
    tools = GraphTools()
    print("Graph stats:", tools.stats())

    question = "How does DEEPNORM stabilize deep Transformers?"
    print(f"\nSearch test: {question}")
    for item in tools.search_claims(question):
        print(f"- {item['claim_id']} | score={item['score']} | {item['claim']}")

    claim = tools.get_claim("P20_C0000_CLAIM_001")
    print("\nClaim lookup:")
    print(claim if claim else "Claim not found")

    print("\nRelated claims:")
    print(tools.related_claims("P20_C0000_CLAIM_001"))


if __name__ == "__main__":
    main()
