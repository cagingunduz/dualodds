"""Gerçek gücü bilinen sentetik lig üretici.

Testlerde modelin parametreleri geri kazanıp kazanmadığını doğrulamak ve
gerçek veri olmadan demo yapmak için kullanılır. Bahisçi oranları gerçek
olasılıklara gürültü + marj eklenerek üretilir; kapanış oranları açılıştan
daha az gürültülüdür (piyasa maça yaklaştıkça bilgi toplar).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import poisson

from dualodds.data.schema import validate_matches


def _outcome_probs(lam: float, mu: float, max_goals: int = 10) -> np.ndarray:
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m /= m.sum()
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


def _odds_from_probs(p: np.ndarray, margin: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    noisy = p * np.exp(rng.normal(0.0, noise, size=3))
    noisy /= noisy.sum()
    return 1.0 / (noisy * (1.0 + margin))


def generate_league(
    n_teams: int = 20,
    seasons: int = 2,
    seed: int = 7,
    home_advantage: float = 0.25,
    base_rate: float = 0.15,
    margin: float = 0.05,
    open_noise: float = 0.12,
    close_noise: float = 0.04,
    start: datetime = datetime(2023, 8, 12),
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Çift devreli lig sezonları üretir.

    Returns:
        (maçlar, gerçek güçler) — güçler {takım: (atak, savunma)}.
    """
    rng = np.random.default_rng(seed)
    teams = [f"Takim{i:02d}" for i in range(n_teams)]
    attack = rng.normal(0, 0.3, n_teams)
    defence = rng.normal(0, 0.3, n_teams)
    attack -= attack.mean()
    defence -= defence.mean()

    rows = []
    date = start
    for season_idx in range(seasons):
        fixtures = [(h, a) for h in range(n_teams) for a in range(n_teams) if h != a]
        rng.shuffle(fixtures)
        per_round = n_teams // 2
        for i, (h, a) in enumerate(fixtures):
            if i and i % per_round == 0:
                date += timedelta(days=7)
            lam = np.exp(base_rate + home_advantage + attack[h] - defence[a])
            mu = np.exp(base_rate + attack[a] - defence[h])
            p = _outcome_probs(lam, mu)
            open_odds = _odds_from_probs(p, margin, open_noise, rng)
            close_odds = _odds_from_probs(p, margin * 0.6, close_noise, rng)
            rows.append(
                {
                    "date": date,
                    "home": teams[h],
                    "away": teams[a],
                    "home_goals": rng.poisson(lam),
                    "away_goals": rng.poisson(mu),
                    "odds_home": open_odds[0],
                    "odds_draw": open_odds[1],
                    "odds_away": open_odds[2],
                    "close_home": close_odds[0],
                    "close_draw": close_odds[1],
                    "close_away": close_odds[2],
                    "league": "SYN",
                    "season": str(start.year + season_idx),
                    "source": "synthetic",
                }
            )
        date += timedelta(days=60)  # sezon arası

    truth = {t: (float(attack[i]), float(defence[i])) for i, t in enumerate(teams)}
    return validate_matches(pd.DataFrame(rows)), truth
