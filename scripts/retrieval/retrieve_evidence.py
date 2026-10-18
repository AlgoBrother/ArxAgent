
from collections import OrderedDict
from falkordb import FalkorDB

STOPWORDS = {
    "what", "which", "where", "when", "does", "with", "from",
    "that", "this", "have", "been", "were", "about", "their",
    "them", "they", "into", "and", "for", "the", "are", "how",
    "can", "could", "associated", "occur", "discuss",
}

def terms_from(question):
    return list(dict.fromkeys(
        word.lower().strip(".,?!:;()[]{}")
        for word in question.split()
        if len(word.strip(".,?!:;()[]{}")) >= 4
        and word.lower().strip(".,?!:;()[]{}") not in STOPWORDS
    ))[:12]


def retrieve_evidence(question, limit=8):
    graph = FalkorDB(
        host="localhost", port=6379
    ).select_graph("LLMStability")

    terms = terms_from(question)
    if not terms:
        return []

    query = """
    MATCH (p:Paper)-[:REPORTS]->(c:Claim)
    WHERE any(term IN $terms WHERE
        toLower(c.text) CONTAINS term
        OR toLower(c.claim_type) CONTAINS term
    )
    OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(e:Evidence)
    WITH p, c, collect(DISTINCT e) AS evidence,
         size([term IN $terms WHERE
             toLower(c.text) CONTAINS term
             OR toLower(c.claim_type) CONTAINS term
         ]) AS matched_terms
    RETURN p.paper_id, p.title, c.claim_id, c.text,
           c.claim_type, matched_terms, evidence
    ORDER BY matched_terms DESC
    LIMIT $limit
    """

    rows = graph.query(
        query, {"terms": terms, "limit": limit}
    ).result_set

    results = []
    for row in rows:
        evidence = []
        for item in row[6]:
            if item is None:
                continue
            props = dict(item.properties)
            evidence.append({
                "quote": props.get("quote"),
                "page": props.get("page"),
                "chunk_id": props.get("chunk_id"),
            })

        results.append({
            "paper_id": row[0],
            "paper_title": row[1],
            "claim_id": row[2],
            "claim": row[3],
            "claim_type": row[4],
            "matched_terms": row[5],
            "evidence": evidence,
        })

    return results


if __name__ == "__main__":
    question = (
        "What training instabilities occur, and what interventions "
        "are associated with them?"
    )
    for result in retrieve_evidence(question):
        print(f"\n[{result['paper_id']}] {result['paper_title']}")
        print(f"Claim: {result['claim']}")
        print(f"Matched terms: {result['matched_terms']}")
        if not result["evidence"]:
            print("Evidence: MISSING")
        for item in result["evidence"]:
            print(
                f"Evidence (p. {item['page']}, "
                f"{item['chunk_id']}): {item['quote']}"
            )
