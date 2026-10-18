# scripts/sources/models.py
from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class Paper:
    title: str
    authors: List[str]
    summary: str
    published: datetime
    pdf_url: str
    arxiv_id: str
    link: str