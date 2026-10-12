"""The Odds API bağlayıcısı: önümüzdeki Premier League maçları + oranlar.

Fikstür kaynağı da budur (ayrı bir veri sağlayıcısına gerek kalmaz).
Kota: /odds çağrısı (bölge × pazar) kredi harcar; ücretsiz planda ayda 500 kredi.
Haftalık bir çağrı (1 bölge × 2 pazar) = 2 kredi.

Ortam değişkeni: ODDS_API_KEY
"""

from __future__ import annotations

import json
import os
import statistics
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime

from dualodds.data import entities

API = "https://api.the-odds-api.com/v4"
SPORT = "soccer_epl"
# Keskin piyasa önce: Pinnacle ve Betfair borsası en bilgili fiyatları verir
PREFERRED_BOOKS = ("pinnacle", "betfair_ex_eu", "betfair_ex_uk", "matchbook")


@dataclass
class FixtureOdds:
    event_id: str
    commence_time: datetime
    home: str  # Takım anahtarı (entities)
    away: str
    h2h: tuple[float, float, float] | None = None  # ev, beraberlik, deplasman
    totals_25: tuple[float, float] | None = None  # 2.5 üst, alt
    source: str = ""  # Hangi bahisçi / "median(n)"
    n_books: int = 0
    raw_names: tuple[str, str] = ("", "")
    best: dict = field(default_factory=dict)  # {"home": (oran, bahisçi), ..., "over": ..., "under": ...}
    extra: dict = field(default_factory=dict)


def _default_fetch(url: str) -> tuple[list | dict, dict]:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read()), dict(resp.headers)


def _pick(prices_by_book: dict[str, tuple]) -> tuple[tuple | None, str]:
    for book in PREFERRED_BOOKS:
        if book in prices_by_book:
            return prices_by_book[book], book
    if not prices_by_book:
        return None, ""
    cols = list(zip(*prices_by_book.values()))
    return tuple(round(statistics.median(c), 3) for c in cols), f"median({len(prices_by_book)})"


def parse_event(ev: dict) -> FixtureOdds | None:
    home = entities.key_for(ev["home_team"]) or ev["home_team"]
    away = entities.key_for(ev["away_team"]) or ev["away_team"]
    h2h, totals = {}, {}
    for bm in ev.get("bookmakers", []):
        for m in bm.get("markets", []):
            outs = m.get("outcomes", [])
            if m["key"] == "h2h":
                by = {o["name"]: o["price"] for o in outs}
                if {ev["home_team"], ev["away_team"], "Draw"} <= by.keys():
                    h2h[bm["key"]] = (by[ev["home_team"]], by["Draw"], by[ev["away_team"]])
            elif m["key"] == "totals":
                by = {o["name"]: o["price"] for o in outs if abs(float(o.get("point", 0)) - 2.5) < 1e-9}
                if {"Over", "Under"} <= by.keys():
                    totals[bm["key"]] = (by["Over"], by["Under"])
    h2h_price, src = _pick(h2h)
    tot_price, _ = _pick(totals)
    best: dict = {}
    for prices, names in ((h2h, ("home", "draw", "away")), (totals, ("over", "under"))):
        for book, tup in prices.items():
            for name, price in zip(names, tup):
                if name not in best or price > best[name][0]:
                    best[name] = (float(price), book)
    return FixtureOdds(
        event_id=ev["id"],
        commence_time=datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")),
        home=home,
        away=away,
        h2h=h2h_price,
        totals_25=tot_price,
        source=src,
        n_books=len(h2h),
        raw_names=(ev["home_team"], ev["away_team"]),
        best=best,
    )


def upcoming_with_odds(
    api_key: str | None = None,
    regions: str = "eu",
    markets: str = "h2h,totals",
    fetch=None,
) -> tuple[list[FixtureOdds], dict]:
    """Önümüzdeki maçlar ve oranları. İkinci değer kota bilgisi."""
    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key and fetch is None:
        raise RuntimeError("ODDS_API_KEY tanımlı değil")
    params = {"apiKey": api_key or "", "regions": regions, "markets": markets, "oddsFormat": "decimal", "dateFormat": "iso"}
    url = f"{API}/sports/{SPORT}/odds?{urllib.parse.urlencode(params)}"
    data, headers = (fetch or _default_fetch)(url)
    quota = {k: headers.get(k) for k in ("x-requests-remaining", "x-requests-used", "x-requests-last")}
    fixtures = [f for f in (parse_event(ev) for ev in data) if f is not None]
    return sorted(fixtures, key=lambda f: f.commence_time), quota
