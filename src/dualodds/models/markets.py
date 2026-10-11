"""Skor olasılık matrisinden bahis pazarlarının olasılıklarını türetir."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MarketProbabilities:
    home: float
    draw: float
    away: float
    exp_home_goals: float
    exp_away_goals: float
    over: dict[float, float]  # {2.5: P(toplam gol > 2.5), ...}
    btts: float  # Karşılıklı gol
    top_scores: list[tuple[str, float]]

    @property
    def one_x_two(self) -> np.ndarray:
        return np.array([self.home, self.draw, self.away])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["over"] = {str(k): v for k, v in self.over.items()}
        return d


def markets_from_matrix(m: np.ndarray, lines: tuple[float, ...] = (1.5, 2.5, 3.5), top_n: int = 5) -> MarketProbabilities:
    g = np.arange(m.shape[0])
    total = g[:, None] + g[None, :]
    flat = np.argsort(m, axis=None)[::-1][:top_n]
    tops = [(f"{i}-{j}", float(m[i, j])) for i, j in zip(*np.unravel_index(flat, m.shape))]
    return MarketProbabilities(
        home=float(np.tril(m, -1).sum()),
        draw=float(np.trace(m)),
        away=float(np.triu(m, 1).sum()),
        exp_home_goals=float((m.sum(axis=1) * g).sum()),
        exp_away_goals=float((m.sum(axis=0) * g).sum()),
        over={line: float(m[total > line].sum()) for line in lines},
        btts=float(m[1:, 1:].sum()),
        top_scores=tops,
    )
