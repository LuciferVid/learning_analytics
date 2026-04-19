from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import feedparser
import requests


@dataclass(frozen=True)
class Resource:
    title: str
    url: str
    summary: str
    source: str = "arXiv"


def _arxiv_query_url(query: str, max_results: int) -> str:
    q = requests.utils.quote(query)
    return (
        "https://export.arxiv.org/api/query"
        f"?search_query=all:{q}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
    )


def search_arxiv(query: str, *, max_results: int = 5, timeout_s: int = 15) -> list[Resource]:
    """
    Free resource search via arXiv Atom API.
    Returns papers (title/url/summary) that can be used as learning resources.
    """
    url = _arxiv_query_url(query=query, max_results=max_results)
    resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "learning-analytics-study-coach/1.0"})
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)

    out: list[Resource] = []
    for entry in feed.entries[:max_results]:
        link = ""
        for l in entry.get("links", []):
            if l.get("rel") == "alternate" and l.get("href"):
                link = l["href"]
                break
        if not link:
            link = entry.get("link", "")

        out.append(
            Resource(
                title=(entry.get("title", "") or "").strip().replace("\n", " "),
                url=link,
                summary=(entry.get("summary", "") or "").strip().replace("\n", " "),
            )
        )
    return out


def build_queries_from_gaps(gaps: Iterable[str]) -> list[str]:
    """
    Convert coarse learning gaps into arXiv-friendly search queries.
    """
    mapping = {
        "math": "math education practice problem solving",
        "reading": "reading comprehension strategies education",
        "writing": "academic writing feedback rubric education",
    }
    queries: list[str] = []
    for g in gaps:
        key = g.lower().strip()
        if key in mapping:
            queries.append(mapping[key])
        else:
            queries.append(f"education {key} study strategies")
    # de-dup while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped

