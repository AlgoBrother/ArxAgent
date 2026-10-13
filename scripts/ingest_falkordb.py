import argparse,glob,json,os,hashlib
from falkordb import FalkorDB
ap=argparse.ArgumentParser(); ap.add_argument("--graph",default="LLMStability"); a=ap.parse_args()
db=FalkorDB(host=os.getenv("FALKORDB_HOST","localhost"),port=int(os.getenv("FALKORDB_PORT","6379")),password=os.getenv("FALKORDB_PASSWORD") or None)
g=db.select_graph(a.graph)
def q(query,params): g.query(query,params)
def stable(s): return hashlib.sha1(s.encode()).hexdigest()[:16]
for path in sorted(glob.glob("data/knowledge/*.json")):
    paper=json.load(open(path,encoding="utf-8")); pid=paper["paper_id"]
    q("MERGE (p:Paper {paper_id:$id}) SET p.title=$title",{"id":pid,"title":paper["paper_title"]})
    for c in paper["claims"]:
        cid=c.get("claim_id") or stable(pid+c["statement"]); ev=c["evidence"][0]
        q("""MATCH (p:Paper {paper_id:$pid}) MERGE (c:Claim {claim_id:$cid})
        SET c.text=$text,c.claim_type=$type,c.epistemic_status=$status,c.confidence=$confidence,
        c.evidence_quote=$quote,c.page=$page,c.section=$section
        MERGE (p)-[:REPORTS]->(c)""",{"pid":pid,"cid":cid,"text":c["statement"],"type":c["claim_type"],"status":c["epistemic_status"],"confidence":c.get("confidence",.5),"quote":ev["quote"],"page":ev["page"],"section":ev.get("section")})
        for name in c.get("failure_modes",[]):
            q("MATCH (c:Claim {claim_id:$cid}) MERGE (f:FailureMode {name:$name}) MERGE (c)-[:DESCRIBES_FAILURE]->(f)",{"cid":cid,"name":name})
        for name in c.get("interventions",[]):
            q("MATCH (c:Claim {claim_id:$cid}) MERGE (i:Intervention {name:$name}) MERGE (c)-[:TESTS_OR_PROPOSES]->(i)",{"cid":cid,"name":name})
        for name in c.get("metrics",[]):
            q("MATCH (c:Claim {claim_id:$cid}) MERGE (m:Metric {name:$name}) MERGE (c)-[:MEASURES]->(m)",{"cid":cid,"name":name})
    print("Imported",pid,len(paper["claims"]),"claims")
print("Done. Graph:",a.graph)
