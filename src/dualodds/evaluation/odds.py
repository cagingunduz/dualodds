"""Oran matematiği: marj temizleme, beklenen değer, Kelly, CLV.

Piyasa (özellikle kapanış oranları) en güçlü rakibimiz ve en iyi
referansımızdır. Bir modelin işe yarayıp yaramadığının en dürüst ölçüsü,
kapanış oranını sistematik olarak yenip yenemediğidir (CLV).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def implied_probabilities(odds) -> np.ndarray:
    return 1.0 / np.asarray(odds, dtype=float)


def overround(odds) -> float:
    """Bahisçi marjı: ima edilen olasılıkların toplamı - 1."""
    return float(implied_probabilities(odds).sum() - 1.0)


def devig_proportional(odds) -> np.ndarray:
    q = implied_probabilities(odds)
    return q / q.sum()


def devig_shin(odds) -> np.ndarray:
    """Shin (1993) yöntemi: marjın içeriden bilgili bahisçilere karşı
    konduğunu varsayar; favori-sürpriz yanlılığını orantısal yönteme göre
    daha iyi düzeltir."""
    q = implied_probabilities(odds)
    s = q.sum()
    if s <= 1.0:
        return q / s

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z**2 + 4 * (1 - z) * q**2 / s) - z) / (2 * (1 - z))

    z = brentq(lambda z: probs(z).sum() - 1.0, 0.0, 0.999)
    p = probs(z)
    return p / p.sum()


def expected_value(prob: float, odds: float) -> float:
    """1 birim bahsin beklenen kârı."""
    return prob * odds - 1.0


def kelly_fraction(prob: float, odds: float, multiplier: float = 0.25, cap: float = 0.05) -> float:
    """Kasa yüzdesi olarak önerilen bahis. Model hatasına karşı varsayılan
    olarak çeyrek Kelly ve %5 üst sınır uygulanır."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (prob * b - (1.0 - prob)) / b
    return float(min(max(f * multiplier, 0.0), cap))


def closing_line_value(taken_odds: float, closing_fair_prob: float) -> float:
    """Alınan oranın, marjı temizlenmiş kapanış olasılığına göre beklenen
    değeri. Pozitif CLV uzun vadede kârlılığın en güvenilir öncü göstergesidir."""
    return taken_odds * closing_fair_prob - 1.0
