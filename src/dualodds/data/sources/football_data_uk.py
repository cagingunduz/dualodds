"""football-data.co.uk bağlayıcısı: maç sonuçları + bahis oranları (CSV).

Ücretsiz, geniş geçmişe sahip ve kapanış oranlarını (Pinnacle kapanışı dahil)
içerdiği için ilk backtest ve kalibrasyon kaynağı olarak idealdir.

Lig kodları örnekleri: E0 (Premier League), SP1 (La Liga), I1 (Serie A),
D1 (Bundesliga), F1 (Ligue 1), T1 (Süper Lig), N1 (Eredivisie).
Sezon kodu: "2425" = 2024/25.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

from dualodds.data.schema import validate_matches
from dualodds.data.sources.base import DataSource, SourceInfo, register_source

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# Tercih sırası: Pinnacle (en keskin piyasa) > piyasa ortalaması > Bet365
_OPEN_ODDS = [("PSH", "PSD", "PSA"), ("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A")]
_CLOSE_ODDS = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]


def _first_available(raw: pd.DataFrame, candidates: list[tuple[str, str, str]]) -> pd.DataFrame | None:
    for cols in candidates:
        if all(c in raw.columns for c in cols):
            return raw[list(cols)].apply(pd.to_numeric, errors="coerce")
    return None


def parse_csv(raw: pd.DataFrame, league: str | None = None, season: str | None = None) -> pd.DataFrame:
    """football-data.co.uk CSV'sini kanonik şemaya çevirir."""
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"])
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
            "home": raw["HomeTeam"],
            "away": raw["AwayTeam"],
            "home_goals": raw["FTHG"],
            "away_goals": raw["FTAG"],
        }
    )
    opening = _first_available(raw, _OPEN_ODDS)
    if opening is not None:
        out[["odds_home", "odds_draw", "odds_away"]] = opening.to_numpy()
    closing = _first_available(raw, _CLOSE_ODDS)
    if closing is not None:
        out[["close_home", "close_draw", "close_away"]] = closing.to_numpy()
    out["league"] = league if league is not None else raw.get("Div")
    out["season"] = season
    out["source"] = FootballDataUK.info.name
    return validate_matches(out)


@register_source
class FootballDataUK(DataSource):
    info = SourceInfo(
        name="football-data.co.uk",
        kind="results_odds",
        license="Ücretsiz; ticari kullanım öncesi site sahibinden yazılı teyit alın",
        commercial_use=False,
        url="https://www.football-data.co.uk",
        notes=["Kapanış oranları CLV ölçümü için kullanılır"],
    )

    def fetch(self, league: str, season: str, cache_dir: str | Path | None = None, **_) -> pd.DataFrame:
        cache_path = Path(cache_dir) / f"{league}_{season}.csv" if cache_dir else None
        if cache_path is not None and cache_path.exists():
            content = cache_path.read_bytes()
        else:
            url = BASE_URL.format(season=season, league=league)
            with urllib.request.urlopen(url, timeout=30) as resp:
                content = resp.read()
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(content)
        raw = pd.read_csv(io.BytesIO(content), encoding="latin-1")
        return parse_csv(raw, league=league, season=season)

    @staticmethod
    def from_file(path: str | Path, league: str | None = None, season: str | None = None) -> pd.DataFrame:
        raw = pd.read_csv(path, encoding="latin-1")
        return parse_csv(raw, league=league, season=season)
