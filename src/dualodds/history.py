"""Tarihsel benzer oranlar: "Bu oranlarla açılan maçlar geçmişte nasıl bitti?"

Kaynak: football-data.co.uk arşivi (ücretsiz). Büyük liglerin son sezonlarında
oynanmış maçların sonuçları, 1X2 ve 2.5 Alt/Üst oranları. Ham dosyalar bir kez
indirilip önbelleğe alınır.

Not: football-data.co.uk verisinin ticari üründe gösterilmesi için site sahibinin
iznini almak gerekir; şimdilik iç analiz amaçlı kullanılıyor.
"""

from __future__ import annotations

import io
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
DEFAULT_LEAGUES = ("E0", "E1", "SP1", "I1", "D1", "F1")
# Öncelik: Pinnacle (en keskin piyasa) > piyasa ortalaması > Bet365
_H2H = [("PSH", "PSD", "PSA"), ("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"), ("B365H", "B365D", "B365A")]
_OU = [("P>2.5", "P<2.5"), ("Avg>2.5", "Avg<2.5"), ("BbAv>2.5", "BbAv<2.5"), ("B365>2.5", "B365<2.5")]

SELECTIONS = ("home", "draw", "away", "over", "under")


def last_seasons(current_start_year: int, n: int = 10) -> list[str]:
    """Tamamlanmış son n sezon kodu (ör. 2026 → '2526', '2425', ...)."""
    return [f"{(y - 1) % 100:02d}{y % 100:02d}" for y in range(current_start_year, current_start_year - n, -1)]


def _first(raw: pd.DataFrame, candidates) -> pd.DataFrame | None:
    for cols in candidates:
        if all(c in raw.columns for c in cols):
            block = raw[list(cols)].apply(pd.to_numeric, errors="coerce")
            if block.notna().all(axis=1).mean() > 0.5:
                return block
    return None


def parse_history_csv(raw: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    out = pd.DataFrame(
        {
            "league": league,
            "season": season,
            "home": raw["HomeTeam"].astype(str),
            "away": raw["AwayTeam"].astype(str),
            "hg": pd.to_numeric(raw["FTHG"], errors="coerce"),
            "ag": pd.to_numeric(raw["FTAG"], errors="coerce"),
        }
    )
    h2h = _first(raw, _H2H)
    out[["o_home", "o_draw", "o_away"]] = h2h.to_numpy() if h2h is not None else np.nan
    ou = _first(raw, _OU)
    out[["o_over", "o_under"]] = ou.to_numpy() if ou is not None else np.nan
    return out.dropna(subset=["hg", "ag"]).reset_index(drop=True)


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:  # urllib yönlendirmeleri izler
        return resp.read()


def load_history(cache_dir: Path, leagues=DEFAULT_LEAGUES, seasons: list[str] | None = None, fetch=None, log=print) -> pd.DataFrame:
    """Arşivi önbellekten yükler, eksik dosyaları indirir."""
    seasons = seasons or last_seasons(2026)
    fetch = fetch or _download
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for league in leagues:
        for season in seasons:
            path = cache_dir / f"{league}_{season}.csv"
            if not path.exists():
                try:
                    path.write_bytes(fetch(BASE_URL.format(season=season, league=league)))
                except Exception as e:
                    log(f"Arşiv indirilemedi {league} {season}: {type(e).__name__}: {e}")
                    continue
            try:
                raw = pd.read_csv(io.BytesIO(path.read_bytes()), encoding="latin-1", on_bad_lines="skip")
                frames.append(parse_history_csv(raw, league, season))
            except Exception as e:
                log(f"Arşiv okunamadı {path.name}: {type(e).__name__}: {e}")
    if not frames:
        raise RuntimeError("Tarihsel oran arşivi yüklenemedi")
    return pd.concat(frames, ignore_index=True)


def _fair(odds: np.ndarray) -> np.ndarray:
    inv = 1.0 / odds
    return inv / inv.sum(axis=-1, keepdims=True)


def _roi(odds: np.ndarray, won: np.ndarray) -> float:
    """Her maça 1 birim, o maçın kendi oranından oynansaydı ortalama getiri."""
    return float(np.mean(np.where(won, odds - 1.0, -1.0)))


@dataclass
class SimilarOdds:
    n_h2h: int
    n_ou: int
    rates: dict  # Gerçekleşme oranları: home/draw/away/over/under/btts
    roi: dict  # Sabit bahis getirisi: home/draw/away/over/under
    ranges: dict  # Benzer maçların oran aralıkları
    top_scores: list[tuple[str, float]]
    leagues: dict

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def similar_odds(history: pd.DataFrame, h2h: tuple | None, totals: tuple | None, k: int = 600) -> SimilarOdds | None:
    """Bu maçın oranlarına en yakın k geçmiş maçın sonuç istatistikleri."""
    if h2h is None:
        return None
    hh = history.dropna(subset=["o_home", "o_draw", "o_away"])
    hh = hh[(hh[["o_home", "o_draw", "o_away"]] > 1.0).all(axis=1)]
    if len(hh) < 50:
        return None
    target = _fair(np.array(h2h, dtype=float))
    fair = _fair(hh[["o_home", "o_draw", "o_away"]].to_numpy(dtype=float))
    dist = np.hypot(fair[:, 0] - target[0], fair[:, 2] - target[2])
    near = hh.iloc[np.argsort(dist)[: min(k, len(hh))]]
    hg, ag = near["hg"].to_numpy(), near["ag"].to_numpy()
    res = {"home": hg > ag, "draw": hg == ag, "away": hg < ag}
    rates = {s: float(v.mean()) for s, v in res.items()}
    rates["btts"] = float(((hg > 0) & (ag > 0)).mean())
    roi = {s: _roi(near[f"o_{s}"].to_numpy(dtype=float), res[s]) for s in ("home", "draw", "away")}
    ranges = {s: (round(float(near[f"o_{s}"].quantile(0.1)), 2), round(float(near[f"o_{s}"].quantile(0.9)), 2)) for s in ("home", "draw", "away")}
    scores = Counter(f"{int(a)}-{int(b)}" for a, b in zip(hg, ag))
    top = [(s, round(c / len(near), 3)) for s, c in scores.most_common(5)]

    n_ou = 0
    if totals is not None:
        ou = history.dropna(subset=["o_over", "o_under"])
        ou = ou[(ou[["o_over", "o_under"]] > 1.0).all(axis=1)]
        if len(ou) >= 50:
            t_over = _fair(np.array(totals, dtype=float))[0]
            f_over = _fair(ou[["o_over", "o_under"]].to_numpy(dtype=float))[:, 0]
            near_ou = ou.iloc[np.argsort(np.abs(f_over - t_over))[: min(k, len(ou))]]
            over = (near_ou["hg"] + near_ou["ag"]).to_numpy() > 2.5
            rates["over"], rates["under"] = float(over.mean()), float(1 - over.mean())
            roi["over"] = _roi(near_ou["o_over"].to_numpy(dtype=float), over)
            roi["under"] = _roi(near_ou["o_under"].to_numpy(dtype=float), ~over)
            for s in ("over", "under"):
                ranges[s] = (round(float(near_ou[f"o_{s}"].quantile(0.1)), 2), round(float(near_ou[f"o_{s}"].quantile(0.9)), 2))
            n_ou = len(near_ou)

    return SimilarOdds(
        n_h2h=len(near),
        n_ou=n_ou,
        rates={k_: round(v, 4) for k_, v in rates.items()},
        roi={k_: round(v, 4) for k_, v in roi.items()},
        ranges=ranges,
        top_scores=top,
        leagues=dict(Counter(near["league"]).most_common()),
    )
