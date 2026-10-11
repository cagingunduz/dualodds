import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from dualodds import brief as brief_mod
from dualodds.brief import Evidence, GeminiBrief, MarketView, filter_signals, market_probabilities
from dualodds.config import Settings
from dualodds.data.sources.odds_api import parse_event, upcoming_with_odds
from dualodds.video.highlight_index import parse_highlight_title, season_highlights, team_results
from dualodds.video.watcher import watch_highlights
from dualodds.weekly import run_week, season_start

from fake_youtube import FakeYouTube
from test_video import _FakeGemini, _analysis, _event


def _event_json(home, away, when, books):
    return {
        "id": f"{home}-{away}",
        "commence_time": when,
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {
                "key": key,
                "markets": [
                    {"key": "h2h", "outcomes": [{"name": home, "price": h}, {"name": "Draw", "price": d}, {"name": away, "price": a}]},
                    {"key": "totals", "outcomes": [{"name": "Over", "price": o, "point": 2.5}, {"name": "Under", "price": u, "point": 2.5}]},
                ],
            }
            for key, (h, d, a, o, u) in books.items()
        ],
    }


def test_parse_event_prefers_pinnacle_and_maps_names():
    ev = _event_json("Manchester City", "Brighton and Hove Albion", "2026-10-18T14:00:00Z",
                     {"williamhill": (1.5, 4.5, 6.0, 1.7, 2.1), "pinnacle": (1.55, 4.6, 6.2, 1.75, 2.15)})
    f = parse_event(ev)
    assert (f.home, f.away) == ("Man City", "Brighton")
    assert f.h2h == (1.55, 4.6, 6.2) and f.source == "pinnacle" and f.totals_25 == (1.75, 2.15)


def test_parse_event_median_without_sharp_books():
    ev = _event_json("Arsenal", "Chelsea", "2026-10-18T14:00:00Z",
                     {"a": (2.0, 3.4, 3.8, 1.9, 1.9), "b": (2.1, 3.5, 3.6, 2.0, 1.8), "c": (2.2, 3.3, 3.5, 1.95, 1.85)})
    f = parse_event(ev)
    assert f.h2h == (2.1, 3.4, 3.6) and f.source == "median(3)"


def test_upcoming_with_odds_requires_key(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        upcoming_with_odds()


def test_parse_highlight_titles():
    assert parse_highlight_title("Jesus bags LATE winner! 💥 | Aston Villa 1-2 Nottingham Forest | Premier League Highlights") == (
        "Aston Villa", "Nott'm Forest", 1, 2)
    assert parse_highlight_title("Arsenal v Chelsea | Extended Highlights") == ("Arsenal", "Chelsea", None, None)
    assert parse_highlight_title("Arteta press conference | Arsenal v Chelsea") is None
    assert parse_highlight_title("EVERY goal from Matchweek 5 | Premier League Highlights") is None


def test_season_highlights_dedupes_and_results():
    from dualodds.video.channels import Channel

    yt = FakeYouTube(uploads={"UUsky": [
        ("a", "Arsenal 2-0 Chelsea | Premier League Highlights", "2026-09-20T19:00:00Z"),
        ("a2", "Arsenal 2-0 Chelsea | Extended Highlights", "2026-09-21T09:00:00Z"),
        ("b", "Chelsea 1-1 Liverpool | Premier League Highlights", "2026-09-27T19:00:00Z"),
        ("x", "Pochettino press conference", "2026-09-26T10:00:00Z"),
    ]})
    refs = season_highlights(yt, [Channel("Sky", "highlights", "UCsky", "Sky", "UUsky")], datetime(2026, 7, 1, tzinfo=timezone.utc))
    assert [r.video_id for r in refs] == ["a2", "b"]
    res = team_results(refs, "Chelsea")
    assert [(x["opponent"], x["result"], x["venue"]) for x in res] == [("Arsenal", "M", "Deplasman"), ("Liverpool", "B", "İç saha")]


def _brief(views):
    return GeminiBrief(
        headline="h", summary="s", home_style="hs", away_style="as", key_matchups=["m"],
        market_views=views, risks=["r"], data_gaps=[], confidence="medium",
    )


def test_filter_signals_requires_two_real_videos_and_strength():
    market = market_probabilities((2.0, 3.5, 4.0), (1.9, 1.9))
    allowed = {"https://youtu.be/1", "https://youtu.be/2"}
    ev = lambda u: Evidence(video_url=u, timestamp="01:00", note="n")  # noqa: E731
    views = [
        MarketView(market="OU25", selection="over", stance="supports_higher", strength="strong", reasoning="iki takım da açık",
                   evidence=[ev("https://youtu.be/1"), ev("https://youtu.be/2")]),
        MarketView(market="1X2", selection="home", stance="supports_higher", strength="strong", reasoning="tek video",
                   evidence=[ev("https://youtu.be/1"), ev("https://youtu.be/1")]),
        MarketView(market="1X2", selection="away", stance="supports_lower", strength="moderate", reasoning="uydurma link",
                   evidence=[ev("https://youtu.be/1"), ev("https://youtu.be/FAKE")]),
        MarketView(market="1X2", selection="draw", stance="supports_higher", strength="weak", reasoning="zayıf",
                   evidence=[ev("https://youtu.be/1"), ev("https://youtu.be/2")]),
        MarketView(market="1X2", selection="home", stance="in_line", strength="strong", reasoning="uyumlu",
                   evidence=[ev("https://youtu.be/1"), ev("https://youtu.be/2")]),
    ]
    sig = filter_signals(_brief(views), market, allowed)
    assert [s["selection"] for s in sig] == ["over"]
    assert sig[0]["market_prob"] == pytest.approx(0.5) and sig[0]["direction"] == "yükseğe"


def test_season_start():
    assert season_start(datetime(2026, 10, 11).date()) == datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert season_start(datetime(2027, 3, 1).date()) == datetime(2026, 7, 1, tzinfo=timezone.utc)


def test_run_week_end_to_end(tmp_path):
    from dualodds.video.channels import ChannelSpec  # noqa: F401

    now = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
    odds_payload = [
        _event_json("Arsenal", "Chelsea", "2026-10-18T14:00:00Z", {"pinnacle": (2.0, 3.5, 4.0, 1.9, 1.95)}),
        _event_json("Liverpool", "Everton", "2026-11-30T14:00:00Z", {"pinnacle": (1.3, 5.5, 9.0, 1.6, 2.3)}),  # Pencere dışı
    ]
    yt = FakeYouTube(
        channels={
            "@SkySportsPremierLeague": ("UCsky", "Sky Sports Premier League", "UUsky"),
            "@tifo": ("UCtifo", "Tifo Football", "UUtifo"),
        },
        uploads={
            "UUsky": [
                ("h1", "Arsenal 2-0 Fulham | Premier League Highlights", "2026-10-04T19:00:00Z"),
                ("h2", "Wolves 1-3 Chelsea | Premier League Highlights", "2026-10-05T19:00:00Z"),
                ("h3", "Everton 0-0 Spurs | Premier League Highlights", "2026-10-05T19:00:00Z"),  # Bu hafta oynamayan takımlar
            ],
            "UUtifo": [("t1", "How Arsenal press", "2026-10-06T10:00:00Z"), ("t2", "Barcelona tactics", "2026-10-06T10:00:00Z")],
        },
    )
    watched = []

    def watcher(url, home, away, d, score=None, video_title="", channel="", fps=None):
        watched.append((home, away))
        return watch_highlights(url, home, away, d, score=score, client=_FakeGemini(_analysis([_event("home")])), model="m")

    from test_tactical import _FakeGemini as TacticalGemini, _tactics
    from dualodds.video.tactical import analyze_tactical_video

    def tactical_analyzer(up, ch):
        return analyze_tactical_video(up, ch, client=TacticalGemini([_tactics("Arsenal", ["high_press"])]), model="m")

    contexts = []

    def briefer(home, away, kickoff, context):
        contexts.append(context)
        urls = sorted(brief_mod.allowed_video_urls(context))
        return _brief([
            MarketView(market="1X2", selection="home", stance="supports_higher", strength="moderate", reasoning="pres",
                       evidence=[Evidence(video_url=u, timestamp=None, note="n") for u in urls[:2]]),
        ])

    settings = Settings(data_dir=tmp_path)
    summary = run_week(
        settings, days=7, now=now, yt=yt, odds_fetch=lambda url: (odds_payload, {"x-requests-remaining": "498"}),
        watcher=watcher, tactical_analyzer=tactical_analyzer, briefer=briefer, log=lambda _: None,
    )
    assert summary["fixtures"] == 1
    assert sorted(watched) == [("Arsenal", "Fulham"), ("Wolves", "Chelsea")]  # Yalnızca bu haftanın takımları
    assert summary["tactical_analyzed"] == ["How Arsenal press"]
    ctx = contexts[0]
    assert ctx["home"]["season_results"][0]["result"] == "G"
    assert ctx["home"]["tactical_notes"] and ctx["market_fair_probabilities"]["home"] > 0.45
    sig = summary["signals"]["Arsenal-Chelsea"]
    assert sig and sig[0]["selection"] == "home"
    md = (tmp_path / "reports" / "week_2026-10-15" / "2026-10-18_arsenal_chelsea.md").read_text()
    assert "Süzgeç sonrası sonuç" in md and "1 (ev sahibi)" in md
    data = json.loads((tmp_path / "reports" / "week_2026-10-15" / "2026-10-18_arsenal_chelsea.json").read_text())
    assert data["brief"]["headline"] == "h"
    # İkinci çalıştırmada aynı videolar tekrar izlenmez
    watched.clear()
    run_week(settings, days=7, now=now, yt=yt, odds_fetch=lambda url: (odds_payload, {}), watcher=watcher,
             tactical_analyzer=tactical_analyzer, briefer=briefer, log=lambda _: None)
    assert watched == []


def test_run_week_budget_spreads_across_teams(tmp_path):
    now = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
    odds_payload = [_event_json("Arsenal", "Chelsea", "2026-10-18T14:00:00Z", {"pinnacle": (2.0, 3.5, 4.0, 1.9, 1.95)})]
    uploads = [(f"a{i}", f"Arsenal {i}-0 Fulham | Premier League Highlights", f"2026-09-{10 + i:02d}T19:00:00Z") for i in range(5)]
    uploads += [("c0", "Chelsea 1-0 Wolves | Premier League Highlights", "2026-09-01T19:00:00Z")]
    yt = FakeYouTube(
        channels={"@SkySportsPremierLeague": ("UCsky", "Sky Sports Premier League", "UUsky")},
        uploads={"UUsky": uploads},
    )
    watched = []

    def watcher(url, home, away, d, score=None, video_title="", channel="", fps=None):
        watched.append(video_title)
        return watch_highlights(url, home, away, d, score=score, client=_FakeGemini(_analysis([_event("home")])), model="m")

    run_week(Settings(data_dir=tmp_path), now=now, max_videos=3, yt=yt, odds_fetch=lambda url: (odds_payload, {}),
             watcher=watcher, tactical_analyzer=lambda up, ch: None, briefer=lambda *a: _brief([]), log=lambda _: None)
    # Bütçenin %70'i (2 video): Arsenal'in en yeni maçı ve Chelsea'nin tek maçı; Arsenal'in tüm maçları değil
    assert len(watched) == 2 and any(t.startswith("Chelsea") for t in watched)


def test_daily_quota_stops_run_without_retrying(tmp_path):
    from dualodds.video.watcher import DailyQuotaExceeded, with_retries

    class Quota(Exception):
        code = 429

        def __str__(self):
            return "429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier"

    slept = []
    with pytest.raises(DailyQuotaExceeded):
        with_retries(lambda: (_ for _ in ()).throw(Quota()), delays=(1, 2), sleep=slept.append)
    assert slept == []  # Günlük kota için bekleme/tekrar yok

    now = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
    odds_payload = [_event_json("Arsenal", "Chelsea", "2026-10-18T14:00:00Z", {"pinnacle": (2.0, 3.5, 4.0, 1.9, 1.95)})]
    yt = FakeYouTube(
        channels={"@SkySportsPremierLeague": ("UCsky", "Sky Sports Premier League", "UUsky")},
        uploads={"UUsky": [(f"a{i}", f"Arsenal {i}-0 Fulham | Premier League Highlights", f"2026-09-{10 + i:02d}T19:00:00Z") for i in range(3)]},
    )
    calls = []

    def watcher(*a, **k):
        calls.append(1)
        raise DailyQuotaExceeded("kota")

    briefs = []
    s = run_week(Settings(data_dir=tmp_path), now=now, yt=yt, odds_fetch=lambda url: (odds_payload, {}), watcher=watcher,
                 tactical_analyzer=lambda up, ch: None, briefer=lambda *a: briefs.append(1), log=lambda _: None)
    assert calls == [1] and briefs == [] and s["gemini_quota_exhausted"]
    md = (tmp_path / "reports" / "week_2026-10-15" / "2026-10-18_arsenal_chelsea.md").read_text()
    assert "Piyasa (marjı temizlenmiş)" in md
