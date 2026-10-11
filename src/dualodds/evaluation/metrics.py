"""Olasılıksal tahmin skorları. Doğruluk (accuracy) yanıltıcıdır;
olasılık kalitesini ölçen metrikler kullanılır."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _onehot(outcomes, k: int) -> np.ndarray:
    outcomes = np.asarray(outcomes, dtype=int)
    y = np.zeros((len(outcomes), k))
    y[np.arange(len(outcomes)), outcomes] = 1.0
    return y


def log_loss(probs, outcomes) -> float:
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    idx = np.asarray(outcomes, dtype=int)
    return float(-np.log(p[np.arange(len(idx)), idx]).mean())


def brier_score(probs, outcomes) -> float:
    p = np.asarray(probs, dtype=float)
    return float(((p - _onehot(outcomes, p.shape[1])) ** 2).sum(axis=1).mean())


def ranked_probability_score(probs, outcomes) -> float:
    """Sıralı sonuçlar (ev > beraberlik > deplasman) için RPS; futbolda
    1X2 tahminleri için standart metrik. Düşük daha iyi."""
    p = np.asarray(probs, dtype=float)
    y = _onehot(outcomes, p.shape[1])
    cp, cy = np.cumsum(p, axis=1)[:, :-1], np.cumsum(y, axis=1)[:, :-1]
    return float(((cp - cy) ** 2).sum(axis=1).mean() / (p.shape[1] - 1))


def calibration_table(prob, hit, bins: int = 10) -> pd.DataFrame:
    """Tahmin edilen olasılık kovalarına göre gerçekleşme oranı."""
    df = pd.DataFrame({"prob": np.asarray(prob, dtype=float), "hit": np.asarray(hit, dtype=float)})
    df["bin"] = pd.cut(df["prob"], np.linspace(0, 1, bins + 1), include_lowest=True)
    return (
        df.groupby("bin", observed=True)
        .agg(n=("hit", "size"), mean_pred=("prob", "mean"), observed=("hit", "mean"))
        .reset_index()
    )


def outcome_index(home_goals, away_goals) -> np.ndarray:
    """0 = ev sahibi, 1 = beraberlik, 2 = deplasman."""
    hg = np.asarray(home_goals, dtype=float)
    ag = np.asarray(away_goals, dtype=float)
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
