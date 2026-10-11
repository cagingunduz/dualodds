"""Kanonik maç tablosu.

Her veri kaynağı kendi formatını bu tabloya çevirir. Modeller, backtest ve
ajanlar yalnızca bu şemayı bilir; böylece yeni bir kaynak eklemek modelleri
etkilemez.
"""

from __future__ import annotations

import pandas as pd

# Zorunlu kolonlar
REQUIRED_COLUMNS = ["date", "home", "away", "home_goals", "away_goals"]

# Opsiyonel kolonlar: açılış/alınabilir oranlar ve kapanış oranları (1X2)
ODDS_COLUMNS = ["odds_home", "odds_draw", "odds_away"]
CLOSING_COLUMNS = ["close_home", "close_draw", "close_away"]
META_COLUMNS = ["league", "season", "source"]

MATCH_COLUMNS = REQUIRED_COLUMNS + ODDS_COLUMNS + CLOSING_COLUMNS + META_COLUMNS


def validate_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Şemayı doğrular, tipleri normalize eder ve tarihe göre sıralar.

    Eksik opsiyonel kolonlar NaN ile eklenir.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik zorunlu kolonlar: {missing}")

    out = df.copy()
    for col in MATCH_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA if col in META_COLUMNS else float("nan")

    out["date"] = pd.to_datetime(out["date"])
    out["home"] = out["home"].astype(str).str.strip()
    out["away"] = out["away"].astype(str).str.strip()
    for col in ODDS_COLUMNS + CLOSING_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Oynanmamış maçlar (gol bilgisi yok) fikstür olarak tutulabilir
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce").astype("Int64")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce").astype("Int64")

    return out[MATCH_COLUMNS].sort_values("date", kind="stable").reset_index(drop=True)


def played(df: pd.DataFrame) -> pd.DataFrame:
    """Skoru belli olan maçlar."""
    return df[df["home_goals"].notna() & df["away_goals"].notna()]
