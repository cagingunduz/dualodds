"""Takım bazında video izci profili: birden çok maçın gözlemlerini birleştirir.

Tekil maç gözlemleri gürültülüdür; tekrar eden örüntüler ("son 5 maçın 4'ünde
arka direkten gol yedi") asıl değerli sinyaldir.
"""

from __future__ import annotations

from collections import Counter

from dualodds.video.schema import StoredVideoAnalysis

HIGHLIGHT_BIAS_NOTE = (
    "Özet videolar maçın küçük bir kısmını ve ağırlıkla gol/pozisyon anlarını gösterir; "
    "pres ve oyun kurma gözlemleri düşük güvenlidir. Sayımlar eğilim gösterir, istatistik yerine geçmez."
)


def _side(record: StoredVideoAnalysis, team: str) -> tuple[str, str]:
    return ("home", "away") if record.home == team else ("away", "home")


def team_profile(team: str, records: list[StoredVideoAnalysis]) -> dict:
    """`records` takımın maçları olmalı (VideoStore.for_team çıktısı)."""
    if not records:
        return {"team": team, "matches_analyzed": 0, "note": "Bu takım için video analizi yok."}

    created_origin, conceded_origin = Counter(), Counter()
    created_channel, conceded_channel = Counter(), Counter()
    pressing, build_up = Counter(), Counter()
    defensive_issues, strengths, weaknesses, matches = [], [], [], []
    goals_for = goals_against = 0

    for r in records:
        own, opp = _side(r, team)
        obs = getattr(r.analysis, own)
        pressing[obs.pressing] += 1
        build_up[obs.build_up] += 1
        strengths += obs.strengths
        weaknesses += obs.weaknesses
        opponent = r.away if own == "home" else r.home
        matches.append(
            {
                "date": r.match_date,
                "opponent": opponent,
                "venue": "İç saha" if own == "home" else "Deplasman",
                "observed_score": r.analysis.observed_score,
                "summary": r.analysis.tactical_summary,
                "video_url": r.video_url,
            }
        )
        for ev in r.analysis.events:
            if ev.event_type in ("goal", "big_chance", "woodwork", "shot_saved", "penalty_awarded"):
                if ev.team == own:
                    created_origin[ev.origin] += 1
                    created_channel[ev.attack_channel] += 1
                else:
                    conceded_origin[ev.origin] += 1
                    conceded_channel[ev.attack_channel] += 1
                    if ev.defensive_issue:
                        defensive_issues.append(f"{r.match_date} vs {opponent}: {ev.defensive_issue}")
            if ev.event_type == "goal":
                if ev.team == own:
                    goals_for += 1
                else:
                    goals_against += 1

    return {
        "team": team,
        "matches_analyzed": len(records),
        "goals_seen": {"for": goals_for, "against": goals_against},
        "chances_created_by_origin": dict(created_origin.most_common()),
        "chances_created_by_channel": dict(created_channel.most_common()),
        "chances_conceded_by_origin": dict(conceded_origin.most_common()),
        "chances_conceded_by_channel": dict(conceded_channel.most_common()),
        "pressing_observed": dict(pressing),
        "build_up_observed": dict(build_up),
        "recurring_defensive_issues": defensive_issues[-10:],
        "observed_strengths": strengths[-10:],
        "observed_weaknesses": weaknesses[-10:],
        "matches": matches,
        "caveat": HIGHLIGHT_BIAS_NOTE,
    }
