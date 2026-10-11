from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from dualodds.video.channels import ChannelSpec, resolve_channels
from dualodds.video.tactical import (
    TacticalStore,
    TacticalVideoAnalysis,
    analyze_tactical_video,
    mentioned_teams,
    tactical_matchup,
    tactical_scan,
)
from dualodds.video.youtube import Upload, parse_duration

from fake_youtube import FakeYouTube


def test_parse_duration():
    assert parse_duration("PT1H2M3S") == 3723
    assert parse_duration("PT45S") == 45
    assert parse_duration("PT12M") == 720


def test_resolve_channels_handle_then_search_and_cache(tmp_path):
    yt = FakeYouTube(
        channels={
            "@tifo": ("UCtifo", "Tifo Football", "UUtifo"),
            "@wrong": ("UCx", "Some Other Channel", "UUx"),
            "UCcv": ("UCcv", "The Coaches' Voice", "UUcv"),
        },
        search={"The Coaches' Voice": [("UCfan", "Coaches Voice Fans"), ("UCcv", "The Coaches' Voice")]},
    )
    specs = (
        ChannelSpec("Tifo Football by The Athletic", "tactical", ("@tifo",)),
        ChannelSpec("The Coaches' Voice", "tactical", ("@wrong",)),
        ChannelSpec("Missing Channel", "tactical", ()),
    )
    logs = []
    cache = tmp_path / "channels.json"
    out = resolve_channels(yt, cache, specs, log=logs.append)
    assert [c.channel_id for c in out] == ["UCtifo", "UCcv"]
    assert any("Missing Channel" in line for line in logs)
    # İkinci çağrı önbellekten: yalnızca bulunamayan kanal için tekrar arama yapılır
    yt.calls.clear()
    resolve_channels(yt, cache, specs[:2], log=logs.append)
    assert yt.calls == []


def _tactics(team, style, vulnerable=(), effective=()):
    return {
        "team": team,
        "manager": "Hoca",
        "period_covered": "2026/27",
        "style_tags": list(style),
        "effective_against": list(effective),
        "vulnerable_to": list(vulnerable),
        "shape_in_possession": "3-2-5",
        "shape_out_of_possession": "4-4-2",
        "attacking_patterns": ["Sol kanatta üçgenler"],
        "defensive_weaknesses": ["Bek arkası"],
        "key_player_roles": [],
        "summary": "Özet",
        "evidence_timestamps": ["02:10"],
    }


class _FakeGemini:
    def __init__(self, teams):
        self.teams = teams
        self.models = SimpleNamespace(generate_content=self._gen)
        self.prompts = []

    def _gen(self, **kw):
        self.prompts.append(kw["contents"].parts[1].text)
        a = TacticalVideoAnalysis.model_validate(
            {"video_type": "team_analysis", "about_premier_league": True, "teams": self.teams, "key_takeaways": ["x"], "confidence": "medium"}
        )
        return SimpleNamespace(parsed=a, text=a.model_dump_json())


def _upload(vid, title, when="2026-09-15T10:00:00+00:00"):
    return Upload(vid, title, "", "UCtifo", "Tifo Football", datetime.fromisoformat(when))


def test_analyze_tactical_video_maps_unknown_team_names():
    fake = _FakeGemini([_tactics("Manchester City", ["high_press"]), _tactics("Real Madrid", ["low_block"])])
    rec = analyze_tactical_video(_upload("v1", "How Man City press"), "Tifo Football", client=fake, model="m")
    assert [t.team for t in rec.analysis.teams] == ["Man City"]  # Tam ad anahtara çevrildi, PL dışı takım atıldı
    assert "Man City (Manchester City)" in fake.prompts[0]


def test_mentioned_teams():
    assert set(mentioned_teams("Why Arsenal and Spurs struggle")) == {"Arsenal", "Tottenham"}
    assert mentioned_teams("Barcelona's new pressing scheme") == []


def test_tactical_scan_filters_and_stores(tmp_path):
    from dualodds.video.channels import Channel

    yt = FakeYouTube(
        uploads={
            "UUtifo": [
                ("a1", "How Arsenal build up under pressure", "2026-09-20T10:00:00Z"),
                ("b1", "Barcelona tactics explained", "2026-09-19T10:00:00Z"),
                ("s1", "Liverpool in 30 seconds #shorts", "2026-09-18T10:00:00Z"),
                ("old", "Chelsea tactics", "2026-01-01T10:00:00Z"),
            ],
            "UUsky": [("h1", "Arsenal 2-1 Chelsea | Highlights", "2026-09-20T19:00:00Z")],
        },
        durations={"a1": "PT14M", "s1": "PT30S"},
    )
    channels = [
        Channel("Tifo", "tactical", "UCtifo", "Tifo Football", "UUtifo"),
        Channel("Sky", "highlights", "UCsky", "Sky Sports Premier League", "UUsky"),
    ]
    store = TacticalStore(tmp_path)

    def analyzer(up, channel):
        return analyze_tactical_video(up, channel, client=_FakeGemini([_tactics("Arsenal", ["possession_build_up"])]), model="m")

    since = datetime(2026, 9, 1, tzinfo=timezone.utc)
    s = tactical_scan(yt, channels, store, since, analyzer=analyzer, log=lambda _: None)
    assert [a["title"] for a in s["analyzed"]] == ["How Arsenal build up under pressure"]
    assert len(s["skipped_no_pl_team"]) == 1 and len(s["skipped_duration"]) == 1
    assert store.has("a1")
    # Tekrar taramada analiz edilmiş video atlanır
    s2 = tactical_scan(yt, channels, store, since, analyzer=analyzer, log=lambda _: None)
    assert s2["skipped_existing"] == 1 and not s2["analyzed"]

    notes = store.for_team("Arsenal", as_of=datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert notes and notes[0]["channel"] == "Tifo Football" and notes[0]["style_tags"] == ["possession_build_up"]
    # Analizden önceki bir tarih için not dönmez (geleceğe sızıntı yok)
    assert store.for_team("Arsenal", as_of=datetime(2026, 9, 10, tzinfo=timezone.utc)) == []


def test_tactical_matchup_edges():
    home_notes = [_tactics("Liverpool", ["high_press", "counter_attack"], vulnerable=["direct_long_ball"])]
    away_notes = [
        _tactics("Everton", ["low_block", "direct_long_ball"], vulnerable=["high_press"]),
        _tactics("Everton", ["low_block"], vulnerable=["high_press", "crossing"]),
    ]
    m = tactical_matchup("Liverpool", "Everton", home_notes, away_notes)
    assert m["home_edges"] == [{"team": "Liverpool", "style": "high_press", "label": "önde yoğun pres", "sources": 1}]
    assert m["away_edges"][0]["style"] == "direct_long_ball"
    assert m["away_profile"]["vulnerable_to"]["high_press"] == 2
