"""Verify curated claims against chunk text and fill pages from [PAGE n] markers.
Usage: python scripts/curate_check.py data/knowledge/by_chunk/P02/P02_C0002_curated.json
No LLM calls."""
import json, re, sys
from pathlib import Path

PAGE = re.compile(r"\[PAGE (\d+)\]")
norm = lambda s: " ".join(s.split())

def page_for(rows, idx, quote):
    text = norm(rows[idx]["text"])
    pos = text.find(norm(quote))
    if pos < 0:
        return None
    found = [int(m.group(1)) for m in PAGE.finditer(text[:pos])]
    if found:
        return found[-1]
    for j in range(idx - 1, -1, -1):
        found = [int(m.group(1)) for m in PAGE.finditer(rows[j]["text"])]
        if found:
            return found[-1]
    return 1

def main(path):
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    pid = data["paper_id"]
    rows = [json.loads(l) for l in open(f"data/chunks/{pid}.jsonl", encoding="utf-8") if l.strip()]
    bad = 0
    for c in data["claims"]:
        for ev in c["evidence"]:
            idx = int(ev["chunk_id"].rsplit("_C", 1)[1])
            pg = page_for(rows, idx, ev["quote"])
            if pg is None:
                bad += 1
                print(f"MISSING  {c['claim_id']}: {ev['quote'][:70]!r}")
            else:
                ev["page"] = pg
                print(f"ok p.{pg:<3} {c['claim_id']}: {ev['quote'][:55]!r}")
    if bad:
        print(f"\n{bad} quote(s) not found: file NOT modified.")
        return 1
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nAll quotes verified; pages written.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
