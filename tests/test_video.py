from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from dualodds.agents import AnalysisContext
from dualodds.agents import tools
from dualodds.agents.analyst import _build_tools
from dualodds.data import entities
from dualodds.video import MatchVideoAnalysis, VideoStore
from dualodds.video.finder import HighlightFinder, score_candidate
from dualodds.video.profile import team_profile
from dualodds.video.watcher import watch_highlights


def _team_obs(**kw):
    base = {
        "pressing": "high",
        "build_up": "short",
        "main_attack_channels": ["left"],
        "observed_formation": "4-3-3",
        "strengths": ["Sol kanat kombinasyonları"],
        "weaknesses": ["Arka direk savunması"],
        "set_piece_notes": None,
    }
    return {**base, **kw}


def _event(team, origin="cross", event_type="goal", channel="left", issue=None):
    return {
        "video_timestamp": "01:23",
        "match_minute": 12,
        "team": team,
        "event_type": event_type,
        "origin": origin,
        "attack_channel": channel,
        "players": [],
        "description": "Ortaya çevrilen top kafayla gol oldu.",
        "defensive_issue": issue,
        "chance_quality": "high",
    }


def _analysis(events, matches=True):
    return MatchVideoAnalysis.model_validate(
        {
            "video_matches_fixture": matches,
            "observed_score": "2-1",
            "home": _team_obs(),
            "away": _team_obs(pressing="low", build_up="long", main_attack_channels=["center"]),
            "events": events,
            "tactical_summary": "Ev sahibi sol kanattan üstünlük kurdu.",
            "limitations": "Özet 8 dakika; pres evreleri az görünüyor.",
        }
    )


class _FakeGemini:
    def __init__(self, analysis):
        self.analysis = analysis
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed=self.analysis, text=self.analysis.model_dump_json())


def test_entities_alias_matching():
    assert entities.mentions("Manchester City vs Man Utd | Premier League Highlights", "Man City")
    assert entities.mentions("Manchester City vs Man Utd | Premier League Highlights", "Man United")
    assert entities.mentions("Spurs 2-0 Wolves", "Tottenham")
    assert not entities.mentions("Manchester City vs Arsenal", "Man United")
    assert entities.team("Nott'm Forest").name == "Nottingham Forest"


def test_score_candidate():
    d = date(2026, 9, 20)
    pub = datetime(2026, 9, 20, 19, tzinfo=timezone.utc)
    extended = score_candidate("Arsenal v Chelsea | Extended Highlights", pub, "Arsenal", "Chelsea", d)
    plain = score_candidate("Arsenal 2-1 Chelsea | Premier League Highlights", pub, "Arsenal", "Chelsea", d)
    late = score_candidate("Arsenal 2-1 Chelsea | Highlights", datetime(2026, 9, 23, tzinfo=timezone.utc), "Arsenal", "Chelsea", d)
    assert extended > plain > late > 0
    assert score_candidate("Arsenal v Chelsea REACTION", pub, "Arsenal", "Chelsea", d) == 0
    assert score_candidate("Arsenal v Spurs | Highlights", pub, "Arsenal", "Chelsea", d) == 0
    assert score_candidate("Arteta press conference | Arsenal v Chelsea", pub, "Arsenal", "Chelsea", d) == 0
    assert score_candidate("Arsenal v Chelsea | Highlights", datetime(2026, 9, 19, tzinfo=timezone.utc), "Arsenal", "Chelsea", d) == 0


def test_highlight_finder_uses_highlights_channel_only():
    from fake_youtube import FakeYouTube

    yt = FakeYouTube(
        uploads={
            "UUsky": [
                ("pl1", "Arsenal 2-1 Chelsea | Premier League Highlights", "2026-09-20T19:00:00Z"),
                ("pc", "Arteta press conference ahead of Chelsea", "2026-09-18T12:00:00Z"),
                ("old", "Arsenal v Chelsea | Highlights", "2025-03-01T12:00:00Z"),
            ],
            "UUtifo": [("t1", "Arsenal vs Chelsea highlights", "2026-09-20T20:00:00Z")],
        }
    )
    from dualodds.video.channels import Channel

    channels = [
        Channel("Sky Sports Premier League", "highlights", "UCsky", "Sky Sports Premier League", "UUsky"),
        Channel("Tifo", "tactical", "UCtifo", "Tifo Football", "UUtifo"),
    ]
    finder = HighlightFinder(yt, channels)
    cands = finder("Arsenal", "Chelsea", "2026-09-20")
    assert [c.video_id for c in cands] == ["pl1"]
    units = yt.units_used
    finder("Liverpool", "Everton", "2026-09-21")  # Önbellekten, yeni API çağrısı yok
    assert yt.units_used == units


def test_watch_highlights_builds_record():
    fake = _FakeGemini(_analysis([_event("home")]))
    rec = watch_highlights("https://www.youtube.com/watch?v=abc", "Man City", "Arsenal", "2026-09-20", score="2-1", client=fake, model="test-model")
    assert rec.analysis.observed_score == "2-1"
    assert rec.analyzer_model == "test-model"
    call = fake.calls[0]
    parts = call["contents"].parts
    assert parts[0].file_data.file_uri.endswith("v=abc")
    assert "Manchester City" in parts[1].text and "2-1" in parts[1].text
    assert call["config"].response_schema is MatchVideoAnalysis


def _store_with_matches(tmp_path):
    store = VideoStore(tmp_path)
    fake_records = [
        ("2026-08-22", "Arsenal", "Chelsea", [_event("home"), _event("away", "counter_attack", channel="right", issue="Bek önde yakalandı")]),
        ("2026-08-29", "Fulham", "Arsenal", [_event("away", "corner"), _event("home", "counter_attack", channel="right", issue="Sağ bek arkası boş")]),
        ("2026-09-05", "Arsenal", "Everton", [_event("home", "cutback")]),
        ("2026-09-12", "Arsenal", "Wolves", [_event("home")]),  # Yanlış video
    ]
    for d, h, a, evs in fake_records:
        rec = watch_highlights("u", h, a, d, client=_FakeGemini(_analysis(evs, matches=(a != "Wolves"))), model="m")
        store.save(rec)
    return store


def test_store_and_profile(tmp_path):
    store = _store_with_matches(tmp_path)
    assert store.has("2026-08-22", "Arsenal", "Chelsea")
    recs = store.for_team("Arsenal", last_n=10)
    assert len(recs) == 3  # Uyuşmayan video hariç
    prof = team_profile("Arsenal", recs)
    assert prof["matches_analyzed"] == 3
    assert prof["goals_seen"] == {"for": 3, "against": 2}
    assert prof["chances_conceded_by_origin"] == {"counter_attack": 2}
    assert prof["chances_conceded_by_channel"] == {"right": 2}
    assert len(prof["recurring_defensive_issues"]) == 2
    # Deplasman maçında takımın kendi gözlemleri "away" tarafından okunur
    assert prof["pressing_observed"] == {"high": 2, "low": 1}
    assert "caveat" in prof


def test_profile_respects_before_date(tmp_path):
    store = _store_with_matches(tmp_path)
    assert len(store.for_team("Arsenal", before="2026-09-01")) == 2


def test_analyst_gets_video_tool_only_with_store(tmp_path, league):
    df, _ = league
    ctx = AnalysisContext.build(df)
    names = [t.name for t in _build_tools(ctx, "Takim00", "Takim01", {})]
    assert "get_video_scouting" not in names

    ctx_v = AnalysisContext.build(df, video_store=VideoStore(tmp_path))
    names_v = [t.name for t in _build_tools(ctx_v, "Takim00", "Takim01", {})]
    assert "get_video_scouting" in names_v and names_v[-1] == "submit_report"
    assert tools.video_scouting(ctx_v, "Takim00")["highlights"]["matches_analyzed"] == 0
