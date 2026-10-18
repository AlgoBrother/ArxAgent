"""Turn raw tool traces into user-facing answers: handles in, citations out."""
import re

ID_RE = re.compile(r"\b[Pp]\d+_C\d+(?:_CLAIM_\d+)?\b")
CITE_RE = re.compile(r"\s*\[(E\d+(?:\s*,\s*E\d+)*)\]")
PROV_TAG = {"hand_curated": "hand-curated", "abstract_only": "abstract only"}


def short_title(title: str) -> str:
    return (title or "Unknown paper").split(":")[0].strip()


def collect_sources(trace: list[dict]) -> dict[str, dict]:
    """claim_id -> evidence record, in first-seen order. Walks any tool result."""
    found: dict[str, dict] = {}

    def visit(obj):
        if isinstance(obj, dict):
            cid = obj.get("claim_id")
            if isinstance(cid, str):
                ev = obj.get("evidence")
                first = ev[0] if isinstance(ev, list) and ev and isinstance(ev[0], dict) else {}
                rec = found.setdefault(cid, dict.fromkeys(
                    ("title", "page", "quote", "text", "status", "provenance")))
                rec["title"] = rec["title"] or obj.get("paper_title")
                rec["page"] = rec["page"] or obj.get("page") or first.get("page")
                rec["quote"] = rec["quote"] or obj.get("quote") or first.get("quote")
                rec["text"] = rec["text"] or obj.get("statement") or obj.get("claim")
                rec["status"] = rec["status"] or obj.get("epistemic_status")
                rec["provenance"] = rec["provenance"] or obj.get("provenance")
            for v in obj.values():
                visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)

    for rec in trace:
        if "result" in rec:
            visit(rec["result"])
    return found


def prepare_evidence(trace, max_sources: int = 14):
    found = collect_sources(trace)
    handles, handle_of = {}, {}
    for i, (cid, s) in enumerate(list(found.items())[:max_sources], start=1):
        handles[f"E{i}"] = {**s, "claim_id": cid}
        handle_of[cid] = f"E{i}"
    return handles, handle_of


def evidence_text(trace, handles, handle_of) -> str:
    """What the model sees at synthesis time. No claim/chunk IDs, only [E#]."""
    lines = ["EVIDENCE (cite with [E#] after each statement that relies on it):"]
    for h, s in handles.items():
        meta = " | ".join(x for x in (
            short_title(s["title"]),
            f"p.{s['page']}" if s.get("page") else None,
            s.get("status"), PROV_TAG.get(s.get("provenance") or "", None)) if x)
        lines.append(f"[{h}] {meta}\n    {(s.get('text') or '')[:300]}")
    for rec in trace:
        r = rec.get("result") if rec.get("tool") == "diagnose" else None
        if isinstance(r, dict):
            lines.append("\nFIX GROUPS (ordered by number of papers, NOT effectiveness):")
            for f in r.get("candidate_fixes", []):
                hs = ", ".join(handle_of[x["claim_id"]] for x in f.get("reports", [])
                               if x.get("claim_id") in handle_of)
                lines.append(f"- {f['fix']} ({f.get('paper_count', '?')} paper(s)): {hs}")
            lines.append(f"Coverage: {r.get('claims_matched', 0)} claim(s) from "
                         f"{len(r.get('papers_matched', []))} paper(s) in the graph.")
    return "\n".join(lines)


def render_answer(answer: str, handles: dict[str, dict]):
    """[E#] -> numbered citations; unknown handles and leaked IDs removed.
    Returns (markdown, sources) where sources keep the internal claim_id."""
    answer = ID_RE.sub("", answer)
    numbering: dict[str, int] = {}

    def repl(m):
        nums = set()
        for h in re.findall(r"E\d+", m.group(1)):
            if h in handles:
                numbering.setdefault(h, len(numbering) + 1)
                nums.add(numbering[h])
        return f" [{', '.join(map(str, sorted(nums)))}]" if nums else ""

    body = CITE_RE.sub(repl, answer).strip()
    sources, lines = [], []
    for h, n in sorted(numbering.items(), key=lambda kv: kv[1]):
        s = handles[h]
        page = f", p. {s['page']}" if s.get("page") else ""
        tag = f" · {PROV_TAG[s['provenance']]}" if s.get("provenance") in PROV_TAG else ""
        quote = f"\n   > “{s['quote']}”" if s.get("quote") else ""
        lines.append(f"{n}. *{s.get('title') or 'Unknown paper'}*{page}{tag}{quote}")
        sources.append({"n": n, "claim_id": s["claim_id"], "paper_title": s.get("title"),
                        "page": s.get("page"), "quote": s.get("quote"),
                        "provenance": s.get("provenance")})
    if lines:
        papers = sorted({short_title(handles[h].get("title")) for h in numbering})
        body += ("\n\n**Sources**\n\n" + "\n".join(lines)
                 + f"\n\n*Evidence base: {len(papers)} paper(s) in the graph "
                   f"({', '.join(papers)}). This is a small curated graph, not the "
                   "whole literature.*")
    return body, sources


def trace_summary(trace: list[dict]) -> list[str]:
    out = []
    for r in trace:
        t, a = r.get("tool"), r.get("arguments", {})
        if "error" in r:
            out.append(f"Tried {t}, which failed: {r['error'][:80]}")
        elif t == "diagnose":
            out.append(f"Looked up failure reports for “{a.get('symptom')}” "
                       f"({r.get('result_count', '?')} result group(s))")
        elif t == "search_claims":
            out.append(f"Searched the graph for “{a.get('query') or a.get('question')}”")
        elif t == "get_claim":
            out.append("Read one claim and its stored evidence")
        elif t == "explore_concept":
            out.append(f"Explored claims linked to “{a.get('concept')}”")
        elif t == "related_claims":
            out.append("Looked for related claims from other papers")
        elif t == "find_papers":
            out.append(f"Searched arXiv for “{' '.join(a.get('terms', []))}”")
        elif t == "ingest_paper":
            out.append("Added a new paper to the graph")
        else:
            out.append(f"Ran {t}")
    return out