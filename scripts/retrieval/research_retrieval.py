
from falkordb import FalkorDB
import os
import re

STOPWORDS = {
    "the", "and", "for", "that", "with", "how", "does", "what",
    "from", "this", "into", "are", "was", "were", "can", "deep",
    "research", "paper", "method", "methods",
}


def terms(text):
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
        if word.lower() not in STOPWORDS
    }


def retrieve_research(question: str, limit: int = 5) -> list[dict]:
    db = FalkorDB(
        host=os.getenv("FALKORDB_HOST", "localhost"),
        port=int(os.getenv("FALKORDB_PORT", "6379")),
        password=os.getenv("FALKORDB_PASSWORD") or None,
    )
    graph = db.select_graph("LLMStability")

    cypher = """
    MATCH (p:Paper)-[:REPORTS]->(c:Claim)
    OPTIONAL MATCH (c)-[:TESTS_OR_PROPOSES]->(i:Intervention)
    OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(e:Evidence)
    RETURN p.paper_id, p.title, c.claim_id, c.text,
           c.claim_type, c.epistemic_status,
           collect(DISTINCT i.name) AS interventions,
           collect(DISTINCT {
               quote: e.quote,
               page: e.page,
               section: e.section
           }) AS evidence
    """

    rows = graph.query(cypher).result_set
    query_terms = terms(question)
    results = []

    for row in rows:
        paper_id, title, claim_id, text, claim_type, status, interventions, evidence = row

        evidence = [
            item for item in evidence
            if item.get("quote") is not None
        ]

        searchable = " ".join([
            str(title or ""),
            str(text or ""),
            " ".join(str(item) for item in interventions),
            " ".join(item["quote"] for item in evidence),
        ])

        matched = query_terms & terms(searchable)
        score = len(matched)

        if score:
            results.append({
                "paper_id": paper_id,
                "paper_title": title,
                "claim_id": claim_id,
                "claim": text,
                "claim_type": claim_type,
                "epistemic_status": status,
                "interventions": interventions,
                "evidence": evidence,
                "matched_terms": sorted(matched),
                "score": score,
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:limit]


if __name__ == "__main__":
    question = input("Research question: ")
    results = retrieve_research(question)

    if not results:
        print("No matching claims found in the current graph.")
    else:
        for result in results:
            print(f"\n[{result['paper_id']}] {result['paper_title']}")
            print(f"Claim ID: {result['claim_id']}")
            print(f"Relevance score: {result['score']}")
            print(f"Matched terms: {', '.join(result['matched_terms'])}")
            print(f"Claim: {result['claim']}")

            for item in result["evidence"]:
                print(
                    f"  Evidence (p. {item['page']}): "
                    f"{item['quote']}"
                )
