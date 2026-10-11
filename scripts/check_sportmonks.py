"""Sportmonks Premier League kapsam testi (salt okunur).

Planımızın Premier League için hangi verileri gerçekten verdiğini ölçer:
son oynanan bir maç ve sıradaki bir maç üzerinde her `include`'u tek tek dener.
Örnek yanıtları incelemek için `data/samples/sportmonks/` altına kaydeder.

Kullanım:
    SPORTMONKS_API_TOKEN=... python scripts/check_sportmonks.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

BASE = "https://api.sportmonks.com/v3/football"
OUT = Path("data/samples/sportmonks")

# Oynanmış maçta denenecekler
PLAYED_INCLUDES = [
    "scores", "statistics.type", "events", "lineups", "formations", "referees", "venue",
    "weatherReport", "sidelined", "xGFixture", "pressure", "trends", "odds", "premiumOdds",
    "postmatchNews", "matchfacts",
]
# Henüz oynanmamış maçta denenecekler (maç öncesi analiz için kritik olanlar)
UPCOMING_INCLUDES = [
    "sidelined", "expectedLineups", "predictedLineups", "odds", "premiumOdds", "prematchNews",
    "weatherReport", "predictions", "venue", "referees",
]


def get(path: str, token: str, **params) -> tuple[int, dict]:
    url = f"{BASE}{urllib.parse.quote(path)}"
    if params:
        url += "?" + urllib.parse.urlencode(params, safe=":;,.")
    req = urllib.request.Request(url, headers={"Authorization": token, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except ValueError:
            return e.code, {"message": e.reason}


def size(value) -> str:
    if value is None:
        return "boş"
    if isinstance(value, list):
        return f"{len(value)} kayıt" if value else "boş liste"
    return "var"


def save(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def probe(fixture_id: int, includes: list[str], token: str, tag: str) -> dict[str, str]:
    results = {}
    for inc in includes:
        status, body = get(f"/fixtures/{fixture_id}", token, include=inc)
        key = inc.split(".")[0]
        if status == 200:
            results[inc] = size(body.get("data", {}).get(key))
            save(f"{tag}_{key}", body)
        else:
            results[inc] = f"ERİŞİM YOK ({status}: {body.get('message', '')[:80]})"
    return results


def main() -> int:
    token = os.environ.get("SPORTMONKS_API_TOKEN")
    if not token:
        print("SPORTMONKS_API_TOKEN tanımlı değil")
        return 1

    status, body = get("/leagues/search/Premier League", token, include="country")
    if status != 200:
        print(f"Lig araması başarısız ({status}): {body.get('message')}")
        return 1
    leagues = [lg for lg in body.get("data", []) if (lg.get("country") or {}).get("name") == "England" and lg["name"] == "Premier League"]
    if not leagues:
        names = [(lg["name"], (lg.get("country") or {}).get("name")) for lg in body.get("data", [])]
        print(f"İngiltere Premier League planda görünmüyor. Erişilebilen eşleşmeler: {names}")
        return 1
    league = leagues[0]
    print(f"Premier League id={league['id']}")
    rate = body.get("rate_limit")
    if rate:
        print(f"Kota: {rate}")

    status, body = get(f"/leagues/{league['id']}", token, include="currentSeason")
    season = (body.get("data") or {}).get("currentseason") or (body.get("data") or {}).get("currentSeason") or {}
    print(f"Güncel sezon: {season.get('name')} (id={season.get('id')})")

    today = date.today()
    status, body = get(
        f"/fixtures/between/{today - timedelta(days=45)}/{today + timedelta(days=30)}", token,
        filters=f"fixtureLeagues:{league['id']}", include="participants;state", per_page=50,
    )
    if status != 200:
        print(f"Fikstür sorgusu başarısız ({status}): {body.get('message')}")
        return 1
    fixtures = body.get("data", [])
    save("fixtures_window", body)
    finished = [f for f in fixtures if (f.get("state") or {}).get("short_name") in ("FT", "AET", "FT_PEN")]
    upcoming = [f for f in fixtures if (f.get("state") or {}).get("short_name") == "NS"]
    print(f"Son 45 gün + önümüzdeki 30 gün: {len(fixtures)} maç ({len(finished)} bitmiş, {len(upcoming)} oynanmamış)")

    report = {}
    if finished:
        fx = finished[-1]
        print(f"\nOynanmış maç örneği: {fx['name']} ({fx['starting_at']})")
        report["played"] = probe(fx["id"], PLAYED_INCLUDES, token, "played")
        for inc, res in report["played"].items():
            print(f"  {inc:<18} {res}")
    if upcoming:
        fx = upcoming[0]
        print(f"\nOynanmamış maç örneği: {fx['name']} ({fx['starting_at']})")
        report["upcoming"] = probe(fx["id"], UPCOMING_INCLUDES, token, "upcoming")
        for inc, res in report["upcoming"].items():
            print(f"  {inc:<18} {res}")

    save("summary", report)
    print(f"\nÖrnek yanıtlar: {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
