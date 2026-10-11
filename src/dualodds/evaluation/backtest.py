"""İleriye doğru yürüyen (walk-forward) backtest.

Model her adımda yalnızca o tarihten önceki maçlarla yeniden eğitilir, böylece
geleceğe ait bilgi sızmaz. Model olasılıkları piyasa (marjı temizlenmiş kapanış)
olasılıklarıyla karşılaştırılır ve basit bir bahis simülasyonu yapılır.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dualodds.data.schema import CLOSING_COLUMNS, ODDS_COLUMNS, played
from dualodds.evaluation.metrics import brier_score, log_loss, outcome_index, ranked_probability_score
from dualodds.evaluation.odds import closing_line_value, devig_shin, expected_value
from dualodds.models.dixon_coles import DixonColesModel

OUTCOMES = ("home", "draw", "away")


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    summary: dict

    def __str__(self) -> str:
        return "\n".join(f"{k:>24}: {v:.4f}" if isinstance(v, float) else f"{k:>24}: {v}" for k, v in self.summary.items())


def _devig_rows(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    out = np.full((len(df), 3), np.nan)
    vals = df[cols].to_numpy(dtype=float)
    for i, row in enumerate(vals):
        if np.all(np.isfinite(row)) and np.all(row > 1.0):
            out[i] = devig_shin(row)
    return out


def walk_forward(
    matches: pd.DataFrame,
    start: pd.Timestamp | str,
    model_factory: Callable[[], DixonColesModel] = DixonColesModel,
    refit_every_days: int = 7,
    edge_threshold: float = 0.03,
    stake: float = 1.0,
) -> BacktestResult:
    df = played(matches).reset_index(drop=True)
    start = pd.Timestamp(start)
    test = df[df["date"] >= start]
    if test.empty:
        raise ValueError("Başlangıç tarihinden sonra maç yok")

    records = []
    model: DixonColesModel | None = None
    next_refit = start
    for date, day in test.groupby("date", sort=True):
        if model is None or date >= next_refit:
            model = model_factory().fit(df, as_of=date)
            next_refit = date + pd.Timedelta(days=refit_every_days)
        for _, m in day.iterrows():
            try:
                p = model.predict(m["home"], m["away"]).one_x_two
            except KeyError:
                continue  # Eğitim verisinde görülmemiş takım
            records.append({**m.to_dict(), "p_home": p[0], "p_draw": p[1], "p_away": p[2]})

    pred = pd.DataFrame(records)
    pred["outcome"] = outcome_index(pred["home_goals"], pred["away_goals"])
    model_p = pred[["p_home", "p_draw", "p_away"]].to_numpy()
    y = pred["outcome"].to_numpy()

    summary: dict = {
        "n_matches": len(pred),
        "model_log_loss": log_loss(model_p, y),
        "model_brier": brier_score(model_p, y),
        "model_rps": ranked_probability_score(model_p, y),
    }

    close_p = _devig_rows(pred, CLOSING_COLUMNS)
    has_close = np.all(np.isfinite(close_p), axis=1)
    if has_close.any():
        summary["market_log_loss"] = log_loss(close_p[has_close], y[has_close])
        summary["model_log_loss_same_set"] = log_loss(model_p[has_close], y[has_close])
        summary["market_rps"] = ranked_probability_score(close_p[has_close], y[has_close])

    # Bahis simülasyonu: maç başına en yüksek beklenen değerli tek seçim
    odds = pred[ODDS_COLUMNS].to_numpy(dtype=float)
    bets = []
    for i in range(len(pred)):
        if not np.all(np.isfinite(odds[i])):
            continue
        ev = model_p[i] * odds[i] - 1.0
        k = int(np.argmax(ev))
        if ev[k] <= edge_threshold:
            continue
        won = y[i] == k
        bets.append(
            {
                "idx": i,
                "pick": OUTCOMES[k],
                "odds": odds[i, k],
                "model_p": model_p[i, k],
                "ev": expected_value(model_p[i, k], odds[i, k]),
                "profit": stake * (odds[i, k] - 1.0) if won else -stake,
                "clv": closing_line_value(odds[i, k], close_p[i, k]) if has_close[i] else np.nan,
            }
        )
    bets_df = pd.DataFrame(bets)
    summary["n_bets"] = len(bets_df)
    if len(bets_df):
        summary["roi"] = float(bets_df["profit"].sum() / (stake * len(bets_df)))
        summary["avg_clv"] = float(bets_df["clv"].mean())
        summary["pct_positive_clv"] = float((bets_df["clv"] > 0).mean())

    pred = pred.join(bets_df.set_index("idx")[["pick", "ev", "profit", "clv"]], how="left")
    return BacktestResult(predictions=pred, summary=summary)
