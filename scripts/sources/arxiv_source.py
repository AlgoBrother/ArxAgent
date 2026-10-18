"""
arxiv.py v7 — adds adaptive freshness retry.

When a query implies "new/recent" but the narrow date window returns
nothing, automatically widen the search instead of giving up.

7 days → 30 days → 180 days → all-time (no cutoff)
"""

import arxiv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from pydantic import BaseModel
class Paper(BaseModel):
    title: str
    authors: List[str]
    summary: str
    published: datetime
    pdf_url: str
    arxiv_id: str
    link: str
    relevance_score: float = 0.0   # filled by evidence ranker


# Escalation ladder for adaptive retry — used only when freshness="high" or "medium"
FRESHNESS_LADDER = [7, 30, 180, None]  # None = no cutoff, search all time

FRESHNESS_DAYS: dict[str, Optional[int]] = {
    "high":   7,
    "medium": 30,
    "low":    None,
}


def _build_query(search_terms: List[str]) -> str:
    if not search_terms:
        raise ValueError("search_terms is empty")
    terms = list(dict.fromkeys(search_terms))[:4]
    wrapped = [f'"{t}"' if " " in t else t for t in terms]
    return " OR ".join(wrapped)


def _result_to_paper(result) -> Paper:
    pub = result.published
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    return Paper(
        title=result.title,
        authors=[a.name for a in result.authors],
        summary=result.summary,
        published=pub,
        pdf_url=str(result.pdf_url),
        arxiv_id=result.entry_id.split("/")[-1],
        link=result.entry_id,
    )


def fetch_by_ids(arxiv_ids: List[str]) -> List[Paper]:
    if not arxiv_ids:
        return []
    client = arxiv.Client(delay_seconds=1.0, num_retries=3)
    search = arxiv.Search(id_list=arxiv_ids)
    papers = []
    try:
        for result in client.results(search):
            papers.append(_result_to_paper(result))
    except arxiv.HTTPError as e:
        print(f"⚠️  arXiv ID fetch error ({e.status}): {e}")
    return papers


def _fetch_with_cutoff(
    query: str,
    cutoff: Optional[datetime],
    max_results: int,
    seen_ids: set,
    client: arxiv.Client,
) -> List[Paper]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    for result in client.results(search):
        pub = result.published
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        if cutoff is not None and pub < cutoff:
            continue
        if result.entry_id in seen_ids:
            continue
        papers.append(_result_to_paper(result))
    return papers


def fetch_papers(
    search_terms: List[str],
    freshness: str = "low",
    max_results: int = 20,
    seen_ids: set | None = None,
    adaptive: bool = True,
) -> List[Paper]:
    """
    Fetch papers, with adaptive freshness widening.

    If freshness="high" or "medium" returns 0 results, automatically
    retry with progressively wider date windows (30d → 180d → all-time)
    instead of giving up. This matches "find me something new, but if
    nothing is genuinely new, give me the best available answer."

    freshness="low" never had a cutoff to begin with, so adaptive
    retry doesn't apply (already searching all time).
    """
    seen_ids = seen_ids or set()
    query = _build_query(search_terms)
    print(f"   arXiv query: {query!r}")

    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=1.0,
        num_retries=3
    )

    if freshness == "low" or not adaptive:
        lookback = FRESHNESS_DAYS.get(freshness)
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=lookback) if lookback else None
        try:
            return _fetch_with_cutoff(query, cutoff, max_results, seen_ids, client)
        except Exception as e:
            print(f"⚠️  arXiv fetch error: {e}")
            return []

    # Adaptive: start at the requested freshness, widen if empty
    start_idx = 0 if freshness == "high" else 1  # medium starts at 30d rung
    for days in FRESHNESS_LADDER[start_idx:]:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days) if days else None
        label = f"{days}d" if days else "all-time"
        try:
            papers = _fetch_with_cutoff(query, cutoff, max_results, seen_ids, client)
        except Exception as e:
            print(f"⚠️  arXiv fetch error at {label}: {e}")
            continue

        if papers:
            print(f"   found {len(papers)} paper(s) at window={label}")
            return papers
        print(f"   0 papers at window={label} — widening...")

    return []