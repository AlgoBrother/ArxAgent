"""Paper discovery and gated ingestion tools for ArxAgent."""

import csv
import json
import re
import subprocess
import sys
import scripts.graph.graph_tools as graph_tools
from pathlib import Path

from scripts.sources.arxiv_source import fetch_by_ids, fetch_papers

ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "data" / "chunks"
BIB_PATH = ROOT / "bibliography.csv"
KNOWLEDGE_DIR = ROOT / "data" / "knowledge" / "by_chunk"
ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def normalize_arxiv_id(raw: str) -> str:
    raw = str(raw).strip()
    if not ARXIV_ID_RE.match(raw):
        raise ValueError(f"Invalid arXiv id: {raw!r}")
    return re.sub(r"v\d+$", "", raw)


def find_papers(terms: list[str], limit: int, in_graph) -> list[dict]:
    """Search arXiv. Pass 2-4 short keywords, NOT a full sentence."""
    terms = [t.strip() for t in terms if isinstance(t, str) and t.strip()][:4]
    if not terms:
        raise ValueError("find_papers needs at least one search term")
    papers = fetch_papers(terms, freshness="low", max_results=limit, adaptive=False)
    results = []
    for p in papers[:limit]:
        aid = re.sub(r"v\d+$", "", p.arxiv_id)
        results.append(
            {
                "arxiv_id": aid,
                "title": p.title,
                "year": p.published.year,
                "abstract_snippet": " ".join(p.summary.split())[:300],
                "in_graph": bool(in_graph(aid)),
            }
        )
    return results


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def write_chunk_file(paper_id: str, paper) -> str:
    """ASSUMPTION: check this row against one line of data/chunks/P20.jsonl."""
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    path = CHUNK_DIR / f"{paper_id}.jsonl"
    chunk_id = f"{paper_id}_C0000"
    text = f"{paper.title}\n\n{' '.join(paper.summary.split())}"
    if not path.exists():
        row = {
            "chunk_id": chunk_id,
            "paper_id": paper_id,
            "page": 1,
            "section": "Abstract",
            "text": text,
        }
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return text


def register_in_bibliography(local_id, aid, paper) -> None:
    with BIB_PATH.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    if any(r.get("paper_id") == local_id for r in rows):
        return
    row = {k: "" for k in fields}
    for key, val in {
        "paper_id": local_id,
        "title": paper.title,
        "arxiv_id": aid,
        "year": str(paper.published.year),
        "url": paper.link,
    }.items():
        if key in row:
            row[key] = val
    needs_newline = not BIB_PATH.read_bytes().endswith(b"\n")
    with BIB_PATH.open("a", encoding="utf-8", newline="") as f:
        if needs_newline:
            f.write("\n")
        csv.DictWriter(f, fieldnames=fields).writerow(row)


def evidence_ok(data: dict, source_text: str) -> tuple[bool, str]:
    """Every evidence quote must appear verbatim in the source text."""
    hay = _norm(source_text)
    for claim in data.get("claims", []):
        for ev in claim.get("evidence", []):
            quote = _norm(str(ev.get("quote", "")))
            if not quote or quote not in hay:
                return False, f"quote not in source: {quote[:80]!r}"
    return True, "ok"


def load_into_graph(paper_id: str, extraction: dict) -> dict:
    """
    REPLACE THIS BODY with a call to your existing loader (the code that
    turned by_chunk JSON into the 62 nodes). Return a small summary dict,
    e.g. {"claims_added": n}. Keep paper_id as the Paper key.
    """
    raise NotImplementedError("Hook up your existing by_chunk -> FalkorDB loader")


def load_bibliography() -> list[dict]:
    with BIB_PATH.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_local_id(arxiv_id: str) -> str | None:
    """Map an arXiv id to an existing local id (P20, ...) via the bibliography.
    ASSUMPTION: the arXiv id appears in a column named arxiv_id; adjust to
    your real header. Falls back to matching inside any column (e.g. url)."""
    aid = normalize_arxiv_id(arxiv_id)
    for row in load_bibliography():
        if row.get("arxiv_id", "").strip().startswith(aid):
            return row["paper_id"]
        if any(aid in (v or "") for v in row.values()):
            return row["paper_id"]
    return None


def next_local_id() -> str:
    nums = [
        int(r["paper_id"][1:])
        for r in load_bibliography()
        if r.get("paper_id", "").startswith("P") and r["paper_id"][1:].isdigit()
    ]
    return f"P{max(nums, default=0) + 1}"


# def make_paper_exists(g):
#     def exists(aid: str) -> bool:
#         res = g.query("MATCH (p:Paper {id: $id}) RETURN count(p)", {"id": aid})
#         return res.result_set[0][0] > 0

#     return exists

from scripts.graph.graph_loader import load_extraction
def ingest_paper(arxiv_id: str, model: str, graph) -> dict:
    aid = normalize_arxiv_id(arxiv_id)

    existing = resolve_local_id(aid)
    if existing:
        return {
            "status": "already_known",
            "local_id": existing,
            "note": "No extraction was run.",
        }

    papers = fetch_by_ids([aid])
    if not papers:
        raise ValueError(f"arXiv has no paper {aid}")
    paper = papers[0]

    local_id = next_local_id()
    source_text = write_chunk_file(local_id, paper)
    chunk_id = f"{local_id}_C0000"
    out_path = KNOWLEDGE_DIR / local_id / f"{chunk_id}.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "extract_one.py"),
            "--paper",
            local_id,
            "--chunk-index",
            "0",
            "--model",
            model,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"extraction failed: {(proc.stdout + proc.stderr)[-400:]}")

    data = json.loads(out_path.read_text(encoding="utf-8"))
    ok, reason = evidence_ok(data, source_text)
    if not ok:
        raise RuntimeError(f"evidence check failed, nothing loaded: {reason}")


    summary = load_extraction(graph, local_id, data,
                              title=paper.title, arxiv_id=aid)

    # Register LAST: a failed ingest must not leave a bibliography row,
    # or the next attempt would report "already_known" and never retry.
    register_in_bibliography(local_id, aid, paper)

    return {
        "status": "ingested",
        "local_id": local_id,
        "arxiv_id": aid,
        "title": paper.title,
        "scope": "abstract_only",
        "claims_extracted": len(data.get("claims", [])),
        **summary,
    }


def known_paper(arxiv_id: str) -> bool:
    """Interim stand-in for a real graph lookup: the paper is in the
    bibliography AND has extraction output. Replace with a FalkorDB
    query once I've seen graph_tools.py."""
    local = resolve_local_id(arxiv_id)
    if not local:
        return False
    d = KNOWLEDGE_DIR / local
    return d.is_dir() and any(d.glob("*.json"))
