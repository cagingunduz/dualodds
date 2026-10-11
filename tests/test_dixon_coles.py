import numpy as np
import pytest

from dualodds.models import DixonColesModel


def test_recovers_true_strengths(league, fitted):
    _, truth = league
    teams = sorted(truth)
    est_att = np.array([fitted.attack[t] for t in teams])
    true_att = np.array([truth[t][0] for t in teams])
    est_def = np.array([fitted.defence[t] for t in teams])
    true_def = np.array([truth[t][1] for t in teams])
    assert np.corrcoef(est_att, true_att)[0, 1] > 0.8
    assert np.corrcoef(est_def, true_def)[0, 1] > 0.8
    assert 0.1 < fitted.home_advantage < 0.4


def test_score_matrix_is_distribution(fitted):
    m = fitted.score_matrix("Takim00", "Takim01")
    assert m.shape == (11, 11)
    assert np.all(m >= 0)
    assert m.sum() == pytest.approx(1.0)


def test_markets_consistent(fitted):
    p = fitted.predict("Takim00", "Takim01")
    assert p.home + p.draw + p.away == pytest.approx(1.0)
    assert p.over[1.5] >= p.over[2.5] >= p.over[3.5]
    assert 0 < p.btts < 1


def test_home_advantage_and_neutral(fitted):
    lam, mu = fitted.expected_goals("Takim00", "Takim01")
    lam_n, mu_n = fitted.expected_goals("Takim00", "Takim01", neutral=True)
    assert lam > lam_n and mu == pytest.approx(mu_n)


def test_no_future_leakage(league):
    df, _ = league
    cut = df["date"].iloc[200]
    model = DixonColesModel().fit(df, as_of=cut)
    assert model.n_matches == int((df["date"] < cut).sum())


def test_unknown_team_raises(fitted):
    with pytest.raises(KeyError):
        fitted.predict("Takim00", "Yok FK")
