"""YouTube'da maç özeti videosunu bulur (YouTube Data API v3).

YOUTUBE_API_KEY ortam değişkeni gerekir. Adaylar; başlıkta iki takımın da
geçmesi, "highlights" vb. kelimeler, kanalın resmî/yayıncı kanalı olması ve
yayın tarihinin maç tarihine yakınlığına göre puanlanır.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from dualodds.data import entities

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

# Yayın hakkı sahibi veya resmî kanallar (başlık eşleşmesi, küçük/büyük harf duyarsız)
PREFERRED_CHANNELS = (
    "premier league",
    "nbc sports",
    "sky sports premier league",
    "sky sports football",
    "match of the day",
    "bein sports",
)
HIGHLIGHT_WORDS = ("highlights", "extended highlights", "özet", "maç özeti")
NEGATIVE_WORDS = ("reaction", "preview", "press conference", "fifa", "efootball", "fc 2", "prediction", "watchalong")


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


def score_candidate(title: str, channel: str, published_at: datetime, home: str, away: str, match_date: date) -> float:
    """Bir videonun aranan maçın özeti olma olasılığına dair sezgisel puan."""
    if not (entities.mentions(title, home) and entities.mentions(title, away)):
        return 0.0
    t = entities.normalize(title)
    if any(w in t for w in NEGATIVE_WORDS):
        return 0.0
    score = 1.0
    if any(w in t for w in HIGHLIGHT_WORDS):
        score += 1.0
    if "extended" in t:
        score += 0.5  # Uzun özet = daha fazla taktik bilgi
    ch = channel.lower()
    if any(p in ch for p in PREFERRED_CHANNELS):
        score += 1.5
    elif any(entities.mentions(channel, k) for k in (home, away)):
        score += 1.0  # Kulüp kanalı
    days = (published_at.date() - match_date).days
    if days < 0 or days > 4:
        return 0.0
    return score - 0.2 * days


def find_highlights(
    home: str,
    away: str,
    match_date: date | str,
    api_key: str | None = None,
    max_results: int = 15,
    fetch=None,
) -> list[VideoCandidate]:
    """Puana göre sıralı aday listesi (en iyisi ilk)."""
    api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY tanımlı değil; videoyu --url ile elle verebilirsiniz")
    if isinstance(match_date, str):
        match_date = date.fromisoformat(match_date[:10])

    h, a = entities.team(home), entities.team(away)
    params = {
        "part": "snippet",
        "type": "video",
        "q": f"{h.name} vs {a.name} highlights",
        "maxResults": max_results,
        "publishedAfter": f"{match_date.isoformat()}T00:00:00Z",
        "publishedBefore": f"{(match_date + timedelta(days=5)).isoformat()}T00:00:00Z",
        "key": api_key,
    }
    url = f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"
    if fetch is None:
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read())
    else:
        payload = fetch(url)

    out = []
    for item in payload.get("items", []):
        sn = item["snippet"]
        published = datetime.fromisoformat(sn["publishedAt"].replace("Z", "+00:00"))
        s = score_candidate(sn["title"], sn["channelTitle"], published, home, away, match_date)
        if s > 0:
            out.append(VideoCandidate(item["id"]["videoId"], sn["title"], sn["channelTitle"], published, s))
    return sorted(out, key=lambda c: c.score, reverse=True)
