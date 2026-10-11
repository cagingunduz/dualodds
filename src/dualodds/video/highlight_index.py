"""Özet kanalının başlıklarından bu sezonun maç listesini çıkarır.

Sky Sports Premier League başlıkları maçı ve skoru içerir:
    "Jesus bags LATE winner! | Aston Villa 1-2 Nottingham Forest | Premier League Highlights"
Böylece ayrı bir sonuç/fikstür veri kaynağına gerek kalmaz; skorlar ayrıca
form ve tahmin takibi (geçmiş performans) için de kullanılır.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from dualodds.data import entities
from dualodds.video.channels import Channel
from dualodds.video.finder import NEGATIVE_WORDS
from dualodds.video.youtube import YouTube

_SCORE = re.compile(r"^(?P<home>.+?)\s+(?P<hg>\d{1,2})\s*[-–]\s*(?P<ag>\d{1,2})\s+(?P<away>.+?)$")
_VERSUS = re.compile(r"^(?P<home>.+?)\s+(?:v|vs|vs\.)\s+(?P<away>.+?)$", re.IGNORECASE)


@dataclass(frozen=True)
class HighlightRef:
    video_id: str
    title: str
    channel: str
    published_at: datetime
    home: str
    away: str
    home_goals: int | None
    away_goals: int | None

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    @property
    def score(self) -> str | None:
        return None if self.home_goals is None else f"{self.home_goals}-{self.away_goals}"

    @property
    def date(self) -> str:
        return self.published_at.date().isoformat()


def parse_highlight_title(title: str) -> tuple[str, str, int | None, int | None] | None:
    """Başlıktan (ev, deplasman, ev golü, deplasman golü); maç özeti değilse None."""
    t = entities.normalize(title)
    if "highlights" not in t or any(w in t for w in NEGATIVE_WORDS):
        return None
    for part in (p.strip() for p in re.split(r"[|•]", title)):
        for rx in (_SCORE, _VERSUS):
            m = rx.match(part)
            if not m:
                continue
            home, away = entities.key_for(m["home"].strip()), entities.key_for(m["away"].strip())
            if home and away and home != away:
                hg = int(m["hg"]) if "hg" in m.groupdict() else None
                ag = int(m["ag"]) if "ag" in m.groupdict() else None
                return home, away, hg, ag
    return None


def season_highlights(yt: YouTube, channels: list[Channel], since: datetime) -> list[HighlightRef]:
    """Özet kanallarındaki `since` sonrası maç özetleri; her maç için tek video (uzun özet tercih)."""
    best: dict[tuple[str, str, str], HighlightRef] = {}
    for ch in (c for c in channels if c.kind == "highlights"):
        for up in yt.uploads(ch.uploads_playlist, since, max_pages=40):
            parsed = parse_highlight_title(up.title)
            if not parsed:
                continue
            ref = HighlightRef(up.video_id, up.title, ch.title, up.published_at, *parsed)
            key = (ref.home, ref.away, ref.date[:7])  # Aynı ay içinde aynı eşleşme = aynı maç
            prev = best.get(key)
            if prev is None or ("extended" in up.title.lower() and "extended" not in prev.title.lower()):
                best[key] = ref
    return sorted(best.values(), key=lambda r: r.published_at)


def team_results(refs: list[HighlightRef], team: str) -> list[dict]:
    """Takımın bu sezonki sonuçları (özet başlıklarından), eskiden yeniye."""
    out = []
    for r in refs:
        if team not in (r.home, r.away) or r.home_goals is None:
            continue
        own_home = r.home == team
        gf, ga = (r.home_goals, r.away_goals) if own_home else (r.away_goals, r.home_goals)
        out.append(
            {
                "date": r.date,
                "opponent": r.away if own_home else r.home,
                "venue": "İç saha" if own_home else "Deplasman",
                "score": r.score,
                "result": "G" if gf > ga else "B" if gf == ga else "M",
            }
        )
    return out
