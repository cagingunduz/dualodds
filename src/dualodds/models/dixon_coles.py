"""Dixon-Coles (1997) skor modeli, zaman ağırlıklı.

Her takım için atak ve savunma gücü, ortak bir ev sahibi avantajı ve düşük
skorlu maçlardaki bağımlılığı düzelten `rho` parametresi tahmin edilir.
Eski maçlar `exp(-xi * gün)` ile ağırlıklandırılır; varsayılan xi≈0.0019
yaklaşık 1 yıllık yarı ömre karşılık gelir.

Bu model platformun "L0" temel modelidir: hızlı, yorumlanabilir ve diğer
modellerin (xG tabanlı, oyuncu bazlı, taktik) karşılaştırılacağı referanstır.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

from dualodds.data.schema import played
from dualodds.models.markets import MarketProbabilities, markets_from_matrix

_RHO_BOUNDS = (-0.2, 0.2)
_EPS = 1e-10


def _tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1.0 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1.0 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1.0 - rho, t)
    return np.maximum(t, _EPS)


@dataclass
class DixonColesModel:
    xi: float = 0.0019
    max_goals: int = 10
    l2: float = 1e-3  # Az maçı olan (ör. ligden yeni çıkan) takımlar için hafif düzenleme

    teams: list[str] = field(default_factory=list, init=False)
    attack: dict[str, float] = field(default_factory=dict, init=False)
    defence: dict[str, float] = field(default_factory=dict, init=False)
    home_advantage: float = field(default=0.0, init=False)
    intercept: float = field(default=0.0, init=False)
    rho: float = field(default=0.0, init=False)
    fitted_as_of: pd.Timestamp | None = field(default=None, init=False)
    n_matches: int = field(default=0, init=False)

    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp | str | None = None) -> "DixonColesModel":
        """Modeli `as_of` tarihinden ÖNCE oynanmış maçlarla eğitir (sızıntı yok)."""
        df = played(matches)
        if as_of is not None:
            as_of = pd.Timestamp(as_of)
            df = df[df["date"] < as_of]
        else:
            as_of = df["date"].max() + pd.Timedelta(days=1)
        if len(df) < 10:
            raise ValueError("Eğitim için en az 10 oynanmış maç gerekli")

        teams = sorted(set(df["home"]) | set(df["away"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        h = df["home"].map(idx).to_numpy()
        a = df["away"].map(idx).to_numpy()
        x = df["home_goals"].to_numpy(dtype=float)
        y = df["away_goals"].to_numpy(dtype=float)
        days = (as_of - df["date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-self.xi * days)
        w /= w.sum()
        const = gammaln(x + 1) + gammaln(y + 1)

        def unpack(theta):
            att = theta[:n] - theta[:n].mean()
            dfn = theta[n : 2 * n] - theta[n : 2 * n].mean()
            return att, dfn, theta[2 * n], theta[2 * n + 1], theta[2 * n + 2]

        def nll(theta):
            att, dfn, home, rho, c = unpack(theta)
            log_lam = c + home + att[h] - dfn[a]
            log_mu = c + att[a] - dfn[h]
            lam, mu = np.exp(log_lam), np.exp(log_mu)
            ll = np.log(_tau(x, y, lam, mu, rho)) + x * log_lam - lam + y * log_mu - mu - const
            return -(w * ll).sum() + self.l2 * (att @ att + dfn @ dfn)

        theta0 = np.zeros(2 * n + 3)
        theta0[2 * n] = 0.2
        theta0[2 * n + 2] = np.log(max(df[["home_goals", "away_goals"]].to_numpy(dtype=float).mean(), 0.1))
        bounds = [(None, None)] * (2 * n) + [(None, None), _RHO_BOUNDS, (None, None)]
        res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds)
        if not res.success and not np.isfinite(res.fun):
            raise RuntimeError(f"Optimizasyon başarısız: {res.message}")

        att, dfn, home, rho, c = unpack(res.x)
        self.teams = teams
        self.attack = {t: float(att[i]) for t, i in idx.items()}
        self.defence = {t: float(dfn[i]) for t, i in idx.items()}
        self.home_advantage = float(home)
        self.rho = float(rho)
        self.intercept = float(c)
        self.fitted_as_of = as_of
        self.n_matches = len(df)
        return self

    def _check(self, team: str) -> None:
        if not self.teams:
            raise RuntimeError("Model henüz eğitilmedi")
        if team not in self.attack:
            raise KeyError(f"Model bu takımı tanımıyor: {team}")

    def expected_goals(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        self._check(home)
        self._check(away)
        home_adv = 0.0 if neutral else self.home_advantage
        lam = np.exp(self.intercept + home_adv + self.attack[home] - self.defence[away])
        mu = np.exp(self.intercept + self.attack[away] - self.defence[home])
        return float(lam), float(mu)

    def score_matrix(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        """P(ev=i, deplasman=j) matrisi, boyut (max_goals+1)²."""
        lam, mu = self.expected_goals(home, away, neutral)
        g = np.arange(self.max_goals + 1)
        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        m[0, 0] *= 1 - lam * mu * self.rho
        m[0, 1] *= 1 + lam * self.rho
        m[1, 0] *= 1 + mu * self.rho
        m[1, 1] *= 1 - self.rho
        m = np.maximum(m, 0)
        return m / m.sum()

    def predict(self, home: str, away: str, neutral: bool = False) -> MarketProbabilities:
        return markets_from_matrix(self.score_matrix(home, away, neutral))

    def ratings(self) -> pd.DataFrame:
        """Takım güç tablosu (atak + savunma ile sıralı)."""
        r = pd.DataFrame({"attack": self.attack, "defence": self.defence})
        r["overall"] = r["attack"] + r["defence"]
        return r.sort_values("overall", ascending=False)
