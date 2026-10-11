"""Maç özeti videosunu özet kanalının (Sky Sports Premier League) yükleme
listesinden bulur.

Arama (search.list, 100 birim) yerine kanalın yükleme listesi gezilir
(sayfa başına 1 birim). Adaylar; başlıkta iki takımın da geçmesi, "highlights"
kelimesi ve yayın tarihinin maç tarihine yakınlığına göre puanlanır.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from dualodds.data import entities
from dualodds.video.channels import Channel
from dualodds.video.youtube import Upload, YouTube

HIGHLIGHT_WORDS = ("highlights", "extended highlights")
NEGATIVE_WORDS = (
    "reaction", "preview", "press conference", "fifa", "efootball", "prediction", "watchalong",
    "interview", "debate", "podcast", "every goal", "all goals", "best goals", "skills", "shorts",
)
MAX_DAYS_AFTER = 4


@dataclass(frozen=True)
class VideoCandidate:
    video_id: str
    title: str
    channel: str
    published_at: datetime
    score: float

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


def score_candidate(title: str, published_at: datetime, home: str, away: str, match_date: date) -> float:
    """Bir videonun aranan maçın özeti olma olasılığına dair sezgisel puan (0 = değil)."""
    if not (entities.mentions(title, home) and entities.mentions(title, away)):
        return 0.0
    t = entities.normalize(title)
    if any(w in t for w in NEGATIVE_WORDS):
        return 0.0
    days = (published_at.date() - match_date).days
    if days < 0 or days > MAX_DAYS_AFTER:
        return 0.0
    score = 1.0
    if any(w in t for w in HIGHLIGHT_WORDS):
        score += 1.0
    if "extended" in t:
        score += 0.5  # Uzun özet = daha fazla taktik bilgi
    return score - 0.2 * days


class HighlightFinder:
    """`finder(home, away, date)` arayüzü; yükleme listesini bir kez çekip önbellekler."""

    def __init__(self, yt: YouTube, channels: list[Channel]):
        self.yt = yt
        self.channels = [c for c in channels if c.kind == "highlights"]
        if not self.channels:
            raise RuntimeError("Özet kanalı tanımlı/çözülmüş değil")
        self._uploads: list[tuple[Channel, Upload]] = []
        self._since: datetime | None = None

    def _ensure(self, since: datetime) -> None:
        if self._since is not None and since >= self._since:
            return
        self._uploads = [(ch, up) for ch in self.channels for up in self.yt.uploads(ch.uploads_playlist, since, max_pages=30)]
        self._since = since

    def __call__(self, home: str, away: str, match_date: date | str) -> list[VideoCandidate]:
        if isinstance(match_date, str):
            match_date = date.fromisoformat(match_date[:10])
        self._ensure(datetime.combine(match_date, time.min, tzinfo=timezone.utc) - timedelta(days=1))
        out = []
        for ch, up in self._uploads:
            s = score_candidate(up.title, up.published_at, home, away, match_date)
            if s > 0:
                out.append(VideoCandidate(up.video_id, up.title, ch.title, up.published_at, s))
        return sorted(out, key=lambda c: c.score, reverse=True)
