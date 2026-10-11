"""Küçük YouTube Data API v3 istemcisi.

Kota maliyetleri (günlük ücretsiz kota 10.000 birim):
    channels.list       1 birim
    playlistItems.list  1 birim (50 video/sayfa)
    videos.list         1 birim (50 video/çağrı)
    search.list       100 birim  → yalnızca kanal ID'si bulunamazsa, bir kez
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime

API = "https://www.googleapis.com/youtube/v3"

Fetch = Callable[[str], dict]


def _default_fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


@dataclass(frozen=True)
class Upload:
    video_id: str
    title: str
    description: str
    channel_id: str
    channel_title: str
    published_at: datetime
    duration_s: int | None = None

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


def parse_duration(iso: str) -> int:
    """'PT1H2M3S' → 3723 saniye."""
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m:
        return 0
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + s


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class YouTube:
    def __init__(self, api_key: str | None = None, fetch: Fetch | None = None):
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
        if not self.api_key and fetch is None:
            raise RuntimeError("YOUTUBE_API_KEY tanımlı değil")
        self._fetch = fetch or _default_fetch
        self.units_used = 0

    def _get(self, endpoint: str, cost: int, **params) -> dict:
        params["key"] = self.api_key
        self.units_used += cost
        return self._fetch(f"{API}/{endpoint}?{urllib.parse.urlencode(params)}")

    def channel_by_handle(self, handle: str) -> dict | None:
        handle = handle if handle.startswith("@") else f"@{handle}"
        items = self._get("channels", 1, part="snippet,contentDetails", forHandle=handle).get("items", [])
        return items[0] if items else None

    def channel_by_id(self, channel_id: str) -> dict | None:
        items = self._get("channels", 1, part="snippet,contentDetails", id=channel_id).get("items", [])
        return items[0] if items else None

    def search_channels(self, query: str, max_results: int = 5) -> list[dict]:
        res = self._get("search", 100, part="snippet", type="channel", q=query, maxResults=max_results)
        return res.get("items", [])

    def uploads(self, playlist_id: str, since: datetime, max_pages: int = 10) -> Iterator[Upload]:
        """Kanalın yükleme listesini yeniden eskiye gezer, `since`'den eskiye gelince durur."""
        token = None
        for _ in range(max_pages):
            params = {"part": "snippet,contentDetails", "playlistId": playlist_id, "maxResults": 50}
            if token:
                params["pageToken"] = token
            res = self._get("playlistItems", 1, **params)
            stop = False
            for item in res.get("items", []):
                sn = item["snippet"]
                published = _ts(item.get("contentDetails", {}).get("videoPublishedAt") or sn["publishedAt"])
                if published < since:
                    stop = True
                    continue
                yield Upload(
                    video_id=sn["resourceId"]["videoId"],
                    title=sn.get("title", ""),
                    description=sn.get("description", ""),
                    channel_id=sn.get("channelId", ""),
                    channel_title=sn.get("channelTitle", ""),
                    published_at=published,
                )
            token = res.get("nextPageToken")
            if stop or not token:
                return

    def durations(self, video_ids: list[str]) -> dict[str, int]:
        out = {}
        for i in range(0, len(video_ids), 50):
            chunk = video_ids[i : i + 50]
            res = self._get("videos", 1, part="contentDetails", id=",".join(chunk))
            for item in res.get("items", []):
                out[item["id"]] = parse_duration(item["contentDetails"].get("duration", ""))
        return out
