"""Testler için sahte YouTube Data API (fetch katmanını taklit eder)."""

import urllib.parse

from dualodds.video.youtube import YouTube


class FakeYouTube(YouTube):
    def __init__(self, uploads=None, channels=None, durations=None, search=None):
        """
        uploads:   {playlist_id: [(video_id, title, published_iso[, description])]}
        channels:  {handle_or_id: (channel_id, title, uploads_playlist)}
        durations: {video_id: iso_duration}
        search:    {query: [(channel_id, title)]}
        """
        self.uploads_data = uploads or {}
        self.channels_data = channels or {}
        self.durations_data = durations or {}
        self.search_data = search or {}
        self.calls = []
        super().__init__(api_key="fake", fetch=self._fetch_impl)

    def _fetch_impl(self, url):
        endpoint = url.split("/v3/")[1].split("?")[0]
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        self.calls.append((endpoint, q))
        if endpoint == "channels":
            key = q.get("forHandle") or q.get("id")
            hit = self.channels_data.get(key)
            if not hit:
                return {"items": []}
            cid, title, upl = hit
            return {"items": [{"id": cid, "snippet": {"title": title}, "contentDetails": {"relatedPlaylists": {"uploads": upl}}}]}
        if endpoint == "search":
            return {"items": [{"id": {"channelId": cid}, "snippet": {"title": t}} for cid, t in self.search_data.get(q["q"], [])]}
        if endpoint == "playlistItems":
            vids = sorted(self.uploads_data.get(q["playlistId"], []), key=lambda v: v[2], reverse=True)
            page = int(q.get("pageToken", 0))
            chunk = vids[page * 50 : (page + 1) * 50]
            items = [
                {
                    "snippet": {
                        "title": v[1],
                        "description": v[3] if len(v) > 3 else "",
                        "publishedAt": v[2],
                        "channelId": q["playlistId"].replace("UU", "UC", 1),
                        "channelTitle": "",
                        "resourceId": {"videoId": v[0]},
                    },
                    "contentDetails": {"videoPublishedAt": v[2]},
                }
                for v in chunk
            ]
            res = {"items": items}
            if (page + 1) * 50 < len(vids):
                res["nextPageToken"] = str(page + 1)
            return res
        if endpoint == "videos":
            ids = q["id"].split(",")
            return {"items": [{"id": i, "contentDetails": {"duration": self.durations_data.get(i, "PT10M")}} for i in ids]}
        raise AssertionError(f"Beklenmeyen uç nokta: {endpoint}")
