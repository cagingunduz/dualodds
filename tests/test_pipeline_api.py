import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from dualodds.api.app import create_app
from dualodds.config import Settings
from dualodds.pipeline import load_matches, refresh_matches, season_code, video_scan
from dualodds.video import VideoStore
from dualodds.video.watcher import watch_highlights

from test_video import _FakeGemini, _analysis, _event


def test_season_code():
    assert season_code(date(2026, 10, 11)) == "2627"
    assert season_code(date(2027, 2, 1)) == "2627"
    assert season_code(date(2026, 7, 1)) == "2627"
    assert season_code(date(2026, 6, 30)) == "2526"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("DUALODDS_DATA_DIR", "/data")
    monkeypatch.setenv("SCHEDULER_ENABLED", "true")
    monkeypatch.setenv("SCAN_LIMIT", "5")
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    s = Settings.from_env()
    assert s.data_dir == Path("/data") and s.video_dir == Path("/data/lake/video")
    assert s.scheduler_enabled and s.scan_limit == 5 and s.admin_token is None


class _FakeSource:
    def __init__(self, df):
        self.df = df
        self.calls = []

    def fetch(self, league, season):
        self.calls.append(season)
        if season == "2627":
            return self.df
        raise OSError("404")


def _fake_finder(home, away, d):
    return [SimpleNamespace(url=f"https://www.youtube.com/watch?v={home}{away}", title=f"{home} v {away} highlights", channel="Premier League")]


def _fake_watcher(url, home, away, d, score=None, video_title="", channel="", fps=None):
    matches = not home.endswith("05")  # Takim05 ev maçları "yanlış video"
    return watch_highlights(url, home, away, d, score=score, client=_FakeGemini(_analysis([_event("home")], matches=matches)), model="stub")


@pytest.fixture
def settings(tmp_path, league):
    df, _ = league
    s = Settings(data_dir=tmp_path, admin_token="secret", scan_days=30, scan_limit=0)
    refresh_matches(s, today=date(2026, 10, 11), source=_FakeSource(df), log=lambda _: None)
    return s


def test_refresh_and_load(settings, league):
    df, _ = league
    loaded = load_matches(settings)
    assert len(loaded) == len(df)
    assert (settings.raw_dir / "E0_2627.csv").exists()


def test_video_scan_skips_existing_and_respects_limit(settings):
    df = load_matches(settings)
    store = VideoStore(settings.video_dir)
    last_day = df["date"].max()
    since = last_day - __import__("pandas").Timedelta(days=14)
    s1 = video_scan(df, store, since=since, limit=3, finder=_fake_finder, watcher=_fake_watcher, log=lambda _: None)
    assert len(s1["analyzed"]) + len(s1["mismatched"]) == 3
    s2 = video_scan(df, store, since=since, finder=_fake_finder, watcher=_fake_watcher, log=lambda _: None)
    assert s2["skipped_existing"] == 3
    assert not s2["errors"]

    def broken(*a, **k):
        raise RuntimeError("Gemini kapalı")

    store2 = VideoStore(settings.data_dir / "other")
    s3 = video_scan(df, store2, since=since, limit=2, finder=_fake_finder, watcher=broken, log=lambda _: None)
    assert s3["errors"] and "Gemini kapalı" in s3["errors"][0]["error"]


def _wait(client, job_id, headers, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("iş bitmedi")


def test_api_auth(settings):
    client = TestClient(create_app(settings))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/status").status_code == 401
    assert client.get("/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/status", headers={"Authorization": "Bearer secret"}).status_code == 200

    no_token = TestClient(create_app(Settings(data_dir=settings.data_dir)))
    assert no_token.get("/status", headers={"Authorization": "Bearer x"}).status_code == 503


def test_api_end_to_end(settings, league):
    df, _ = league
    h = {"Authorization": "Bearer secret"}
    client = TestClient(
        create_app(settings, finder=_fake_finder, watcher=_fake_watcher, source=_FakeSource(df))
    )

    st = client.get("/status", headers=h).json()
    assert st["matches"] == len(df) and st["video_analyses"] == 0

    p = client.get("/predict", params={"home": "Takim00", "away": "Takim01"}, headers=h).json()
    assert p["home"] + p["draw"] + p["away"] == pytest.approx(1.0)
    assert client.get("/predict", params={"home": "Takim00", "away": "Yok"}, headers=h).status_code == 404
    assert len(client.get("/teams", headers=h).json()) == 12
    assert len(client.get("/matches", params={"team": "Takim00", "limit": 3}, headers=h).json()) == 3

    since = str((df["date"].max() - __import__("pandas").Timedelta(days=14)).date())
    r = client.post("/jobs/video-scan", json={"since": since, "limit": 4}, headers=h).json()
    job = _wait(client, r["job_id"], h)
    assert job["status"] == "done", job
    assert len(job["result"]["analyzed"]) + len(job["result"]["mismatched"]) == 4
    assert job["log"]

    vm = client.get("/video/matches", headers=h).json()
    assert len(vm) == 4
    first = vm[0]
    detail = client.get("/video/match", params={"date": first["date"], "home": first["home"], "away": first["away"]}, headers=h)
    assert detail.status_code == 200 and detail.json()["analysis"]["events"]
    prof = client.get(f"/video/profile/{first['home']}", headers=h).json()
    assert "matches_analyzed" in prof

    r = client.post("/jobs/refresh-data", headers=h).json()
    assert _wait(client, r["job_id"], h)["status"] == "done"
    assert client.get("/jobs", headers=h).json()


def test_api_analyze_with_fake_claude(settings):
    from test_agents import _FakeRunnerClient

    fake = _FakeRunnerClient()
    client = TestClient(create_app(settings, analyst_client=fake))
    h = {"Authorization": "Bearer secret"}
    r = client.post("/analyze", json={"home": "Takim00", "away": "Takim01", "odds": [2.1, 3.4, 3.6]}, headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["report"]["lean"] == "1" and "Değer analizi" in body["markdown"]
    tool_names = [t.name for t in fake.kwargs["tools"]]
    assert "get_video_scouting" in tool_names


def test_video_scan_fails_fast_without_keys(settings, monkeypatch):
    for k in ("YOUTUBE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="YOUTUBE_API_KEY"):
        video_scan(load_matches(settings), VideoStore(settings.video_dir), since="2020-01-01", log=lambda _: None)
    monkeypatch.setenv("YOUTUBE_API_KEY", "x")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        video_scan(load_matches(settings), VideoStore(settings.video_dir), since="2020-01-01", log=lambda _: None)
