import numpy as np
import pytest

from dualodds.evaluation import (
    brier_score,
    closing_line_value,
    devig_proportional,
    devig_shin,
    kelly_fraction,
    log_loss,
    overround,
    ranked_probability_score,
)


def test_devig_sums_to_one():
    odds = [1.8, 3.6, 4.8]
    assert overround(odds) > 0
    assert devig_proportional(odds).sum() == pytest.approx(1.0)
    assert devig_shin(odds).sum() == pytest.approx(1.0)


def test_shin_shades_longshot_more_than_proportional():
    odds = [1.25, 6.0, 13.0]
    assert devig_shin(odds)[2] < devig_proportional(odds)[2]


def test_kelly():
    assert kelly_fraction(0.5, 1.9) == 0.0  # Negatif beklenen değer
    assert kelly_fraction(0.6, 2.0, multiplier=1.0, cap=1.0) == pytest.approx(0.2)
    assert kelly_fraction(0.9, 3.0) == 0.05  # Üst sınır


def test_clv_sign():
    assert closing_line_value(2.2, 1 / 2.0) > 0
    assert closing_line_value(1.9, 1 / 2.0) < 0


def test_scores_perfect_vs_uniform():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    uniform = np.full((3, 3), 1 / 3)
    assert log_loss(perfect, y) < log_loss(uniform, y)
    assert brier_score(perfect, y) == pytest.approx(0.0)
    assert ranked_probability_score(perfect, y) == pytest.approx(0.0)
    # RPS sıralamaya duyarlı: ev sahibi kazanırken deplasmana verilen olasılık beraberliğe göre daha çok cezalandırılır
    assert ranked_probability_score([[0.5, 0.0, 0.5]], [0]) > ranked_probability_score([[0.5, 0.5, 0.0]], [0])
