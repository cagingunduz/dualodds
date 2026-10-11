"""Video analizlerinin dosya tabanlı deposu (her maç bir JSON).

Aynı maç iki kez analiz edilmez (API maliyeti). İleride veri gölüne
(Parquet/Postgres) taşınacak; arayüz aynı kalır.
"""

from __future__ import annotations

import re
from pathlib import Path

from dualodds.video.schema import StoredVideoAnalysis

DEFAULT_DIR = Path("data/lake/video")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class VideoStore:
    def __init__(self, root: str | Path = DEFAULT_DIR):
        self.root = Path(root)

    def path_for(self, match_date: str, home: str, away: str) -> Path:
        return self.root / f"{match_date[:10]}_{_slug(home)}_{_slug(away)}.json"

    def has(self, match_date: str, home: str, away: str) -> bool:
        return self.path_for(match_date, home, away).exists()

    def save(self, record: StoredVideoAnalysis) -> Path:
        path = self.path_for(record.match_date, record.home, record.away)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return path

    def all(self) -> list[StoredVideoAnalysis]:
        if not self.root.exists():
            return []
        records = [StoredVideoAnalysis.model_validate_json(p.read_text(encoding="utf-8")) for p in self.root.glob("*.json")]
        return sorted(records, key=lambda r: r.match_date)

    def for_team(self, team: str, last_n: int = 5, before: str | None = None) -> list[StoredVideoAnalysis]:
        """Takımın son `last_n` doğrulanmış video analizi (eskiden yeniye)."""
        records = [
            r
            for r in self.all()
            if team in (r.home, r.away)
            and r.analysis.video_matches_fixture
            and (before is None or r.match_date < before[:10])
        ]
        return records[-last_n:]
