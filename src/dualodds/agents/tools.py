"""Ajanların kullandığı araçlar (saf Python, LLM'den bağımsız test edilebilir).

Her araç JSON'a çevrilebilir bir sözlük döndürür. Yeni veri kaynakları
(sakatlıklar, taktik profiller, hava durumu, haberler) buraya yeni araçlar
olarak eklenir; ajan hangi aracı ne zaman çağıracağına kendisi karar verir.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone

import numpy as np
import pandas as pd

from dualodds.data.schema import played
from dualodds.evaluation.odds import devig_shin, expected_value, kelly_fraction, overround
from dualodds.models.dixon_coles import DixonColesModel
from dualodds.video.profile import team_profile
from dualodds.video.store import VideoStore
from dualodds.video.tactical import TacticalStore, tactical_matchup

OUTCOME_LABELS = ("1", "X", "2")


@dataclass
class AnalysisContext:
    matches: pd.DataFrame
    model: DixonColesModel
    video_store: VideoStore | None = None
    tactical_store: TacticalStore | None = None

    @classmethod
    def build(
        cls,
        matches: pd.DataFrame,
        as_of: pd.Timestamp | str | None = None,
        video_store: VideoStore | None = None,
        tactical_store: TacticalStore | None = None,
    ) -> "AnalysisContext":
        hist = matches if as_of is None else matches[matches["date"] < pd.Timestamp(as_of)]
        return cls(matches=hist, model=DixonColesModel().fit(hist), video_store=video_store, tactical_store=tactical_store)

    @property
    def has_video(self) -> bool:
        return self.video_store is not None or self.tactical_store is not None

    def known_teams(self) -> list[str]:
        return list(self.model.teams)


def _result_for(team: str, row: pd.Series) -> str:
    gf, ga = (row.home_goals, row.away_goals) if row.home == team else (row.away_goals, row.home_goals)
    return "G" if gf > ga else "B" if gf == ga else "M"


def model_probabilities(ctx: AnalysisContext, home: str, away: str) -> dict:
    probs = ctx.model.predict(home, away).to_dict()
    probs["model_as_of"] = str(ctx.model.fitted_as_of.date())
    probs["ratings"] = {
        team: {"attack": round(ctx.model.attack[team], 3), "defence": round(ctx.model.defence[team], 3)}
        for team in (home, away)
    }
    probs["home_advantage"] = round(ctx.model.home_advantage, 3)
    return probs


def team_form(ctx: AnalysisContext, team: str, last_n: int = 6) -> dict:
    df = played(ctx.matches)
    games = df[(df["home"] == team) | (df["away"] == team)].tail(last_n)
    if games.empty:
        raise KeyError(f"Bu takım için maç yok: {team}")
    items = []
    for _, r in games.iterrows():
        venue = "İç saha" if r.home == team else "Deplasman"
        opp = r.away if r.home == team else r.home
        items.append(
            {
                "date": str(r.date.date()),
                "venue": venue,
                "opponent": opp,
                "score": f"{r.home_goals}-{r.away_goals}",
                "result": _result_for(team, r),
            }
        )
    results = [i["result"] for i in items]
    gf = sum(int(r.home_goals if r.home == team else r.away_goals) for _, r in games.iterrows())
    ga = sum(int(r.away_goals if r.home == team else r.home_goals) for _, r in games.iterrows())
    return {
        "team": team,
        "matches": items,
        "points_per_game": round((3 * results.count("G") + results.count("B")) / len(results), 2),
        "goals_for": gf,
        "goals_against": ga,
    }


def head_to_head(ctx: AnalysisContext, home: str, away: str, last_n: int = 6) -> dict:
    df = played(ctx.matches)
    mask = ((df["home"] == home) & (df["away"] == away)) | ((df["home"] == away) & (df["away"] == home))
    games = df[mask].tail(last_n)
    return {
        "matches": [
            {"date": str(r.date.date()), "home": r.home, "away": r.away, "score": f"{r.home_goals}-{r.away_goals}"}
            for _, r in games.iterrows()
        ],
        "note": "Az sayıda maç; tek başına güçlü bir sinyal değildir." if len(games) < 4 else "",
    }


def market_comparison(ctx: AnalysisContext, home: str, away: str, odds_home: float, odds_draw: float, odds_away: float) -> dict:
    odds = np.array([odds_home, odds_draw, odds_away], dtype=float)
    if np.any(odds <= 1.0):
        raise ValueError("Oranlar 1.0'dan büyük olmalı")
    model_p = ctx.model.predict(home, away).one_x_two
    fair = devig_shin(odds)
    rows = []
    for i, label in enumerate(OUTCOME_LABELS):
        rows.append(
            {
                "outcome": label,
                "odds": float(odds[i]),
                "model_prob": round(float(model_p[i]), 4),
                "market_fair_prob": round(float(fair[i]), 4),
                "edge": round(float(model_p[i] - fair[i]), 4),
                "expected_value": round(expected_value(model_p[i], odds[i]), 4),
                "kelly_stake_pct": round(kelly_fraction(model_p[i], odds[i]), 4),
            }
        )
    return {"bookmaker_margin": round(overround(odds), 4), "outcomes": rows}


def _as_of(ctx: AnalysisContext):
    return ctx.model.fitted_as_of.to_pydatetime().replace(tzinfo=timezone.utc) if ctx.model.fitted_as_of is not None else None


def video_scouting(ctx: AnalysisContext, team: str, last_n: int = 5) -> dict:
    """Takımın maç özeti profili + taktik analiz videolarından notlar."""
    out: dict = {"team": team}
    if ctx.video_store is not None:
        as_of = str(ctx.model.fitted_as_of.date()) if ctx.model.fitted_as_of is not None else None
        out["highlights"] = team_profile(team, ctx.video_store.for_team(team, last_n=last_n, before=as_of))
    if ctx.tactical_store is not None:
        out["tactical_notes"] = ctx.tactical_store.for_team(team, as_of=_as_of(ctx))
    if len(out) == 1:
        out["note"] = "Video deposu bağlı değil."
    return out


def matchup(ctx: AnalysisContext, home: str, away: str) -> dict:
    """İki takımın taktik etiketlerinin karşılaştırması (kim kimin zorlandığı tarzda oynuyor)."""
    if ctx.tactical_store is None:
        return {"note": "Taktik video deposu bağlı değil."}
    as_of = _as_of(ctx)
    return tactical_matchup(
        home, away, ctx.tactical_store.for_team(home, as_of=as_of), ctx.tactical_store.for_team(away, as_of=as_of)
    )
