"""Maç analizi: Gemini video bulgularından analiz yazar, oranlar süzgeç olur.

Akış:
1. Bağlam (koddan): iki takımın bu sezonki sonuçları (özet başlıkları), maç
   özeti analizleri, taktik video notları, taktik eşleşme ve piyasanın marjı
   temizlenmiş olasılıkları.
2. Gemini (metin): analizi yazar ve her pazar için video kanıtının piyasa
   olasılığından DAHA YÜKSEK / UYUMLU / DAHA DÜŞÜK bir ihtimale işaret edip
   etmediğini, gücünü ve kanıtını (video + an) söyler. Olasılık uydurmaz.
3. Süzgeç (koddan): yalnızca en az orta güçte ve en az 2 FARKLI videoya
   dayanan, kanıt linkleri gerçekten verdiğimiz videolardan olan görüşler
   sinyal olarak kalır. Gerisi "piyasayla uyumlu".
4. En uygun seçenek (koddan): video sinyali ile tarihsel benzer oran
   istatistiği aynı seçeneği destekliyorsa o seçenek, en iyi oranıyla birlikte
   önerilir; desteklemiyorsa "Pas".
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from dualodds.data import entities
from dualodds.evaluation.odds import devig_proportional, devig_shin

Selection = Literal["home", "draw", "away", "over", "under"]
SELECTION_TR = {"home": "1 (ev sahibi)", "draw": "X (beraberlik)", "away": "2 (deplasman)", "over": "2.5 Üst", "under": "2.5 Alt"}
MIN_VIDEOS = 2


class Evidence(BaseModel):
    video_url: str = Field(description="Kanıtın geldiği video; yalnızca bağlamda verilen URL'lerden biri")
    timestamp: str | None = Field(description="Videodaki an (mm:ss) biliniyorsa")
    note: str = Field(description="Kanıtın kısa açıklaması, Türkçe")


class MarketView(BaseModel):
    market: Literal["1X2", "OU25"]
    selection: Selection
    stance: Literal["supports_higher", "in_line", "supports_lower"] = Field(
        description="Video kanıtı bu seçeneğin olasılığının piyasanınkinden yüksek mi, uyumlu mu, düşük mü olduğuna işaret ediyor?"
    )
    strength: Literal["weak", "moderate", "strong"]
    reasoning: str = Field(description="Gerekçe, Türkçe")
    evidence: list[Evidence]


class GeminiBrief(BaseModel):
    headline: str
    summary: str = Field(description="2-4 paragraf maç analizi, Türkçe")
    home_style: str = Field(description="Ev sahibinin bu sezonki oyun tarzı, kısa")
    away_style: str = Field(description="Deplasmanın bu sezonki oyun tarzı, kısa")
    key_matchups: list[str] = Field(description="Maçı belirleyecek eşleşmeler (ör. 'X'in sağ kanadı vs Y'nin sol beki')")
    market_views: list[MarketView]
    risks: list[str]
    data_gaps: list[str] = Field(description="Bu analizde eksik kalan veya eski olan bilgiler")
    confidence: Literal["low", "medium", "high"]


PROMPT = """Sen bir futbol analiz platformunun baş analistisin. Aşağıdaki bağlam, iki takımın BU SEZONKİ maç özeti videolarından ve taktik analiz videolarından çıkarılmış gözlemleri ve bahis piyasasının olasılıklarını içerir.

Görevin:
1. Maçın analizini yaz: iki takımın oyun tarzı, birbirlerine karşı avantaj/dezavantajları, maçı belirleyecek eşleşmeler.
2. Her pazar seçeneği için (1, X, 2, 2.5 Üst, 2.5 Alt) video kanıtının, piyasanın olasılığına göre DAHA YÜKSEK, UYUMLU ya da DAHA DÜŞÜK bir ihtimale işaret edip etmediğini belirt.

Kurallar:
- Metinlerde piyasa olasılığını yüzde olarak YAZMA ("piyasa %58" gibi). Fiyattan bahsetmen gerekirse oranı yaz (ör. "1.72").
- Bağlamdaki "historical_similar_odds", bu maça benzer oranlarla oynanmış geçmiş maçların sonuçlarıdır; gerektiğinde bunlara atıf yapabilirsin.
- Piyasa çok bilgili kabul edilir. Varsayılan cevap "in_line"dır; ondan ancak somut ve tekrarlayan video kanıtı varsa ayrıl.
- Olasılık veya oran uydurma. Sayısal olasılık yazma; yalnızca yön (stance) ve güç (strength) ver.
- Her kanıt bağlamdaki bir video URL'sine dayanmalı. Bağlamda olmayan URL yazma.
- Geçen sezona veya farklı teknik direktör dönemine ait notları kanıt sayma; data_gaps'te belirt.
- Özet videoların gol anlarına yanlı olduğunu unutma; tek maçlık gözlemi örüntü sayma.
- Serbest metinleri Türkçe yaz.

Maç: {home} – {away} ({kickoff})

BAĞLAM (JSON):
{context}"""


def market_probabilities(h2h: tuple | None, totals: tuple | None) -> dict:
    out = {}
    if h2h:
        p = devig_shin(h2h)
        out.update(home=float(p[0]), draw=float(p[1]), away=float(p[2]))
    if totals:
        p = devig_proportional(totals)
        out.update(over=float(p[0]), under=float(p[1]))
    return out


def allowed_video_urls(context: dict) -> set[str]:
    urls = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "video_url" and isinstance(v, str):
                    urls.add(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(context)
    return urls


_OPPOSITE = {"over": "under", "under": "over"}
_STRENGTH = {"weak": 0, "moderate": 1, "strong": 2}
STRENGTH_TR = {"weak": "zayıf", "moderate": "orta", "strong": "güçlü"}


def filter_signals(brief: GeminiBrief, market: dict, allowed_urls: set[str], min_videos: int = MIN_VIDEOS) -> list[dict]:
    """Süzgeç: güçlü/orta, ≥2 farklı gerçek videoya dayanan, piyasadan ayrışan görüşler.

    Alt/Üst iki yönlü pazardır: "Üst düşüğe" = "Alt yükseğe". Aynı görüşün iki
    yüzü tek sinyal sayılır; her seçenek için en güçlü görüş kalır.
    """
    best: dict[str, dict] = {}
    for v in brief.market_views:
        if v.stance == "in_line" or v.strength == "weak" or v.selection not in market:
            continue
        videos = {e.video_url for e in v.evidence if e.video_url in allowed_urls}
        if len(videos) < min_videos:
            continue
        selection, stance = v.selection, v.stance
        if selection in _OPPOSITE and stance == "supports_lower":
            selection, stance = _OPPOSITE[selection], "supports_higher"
        sig = {
            "selection": selection,
            "label": SELECTION_TR[selection],
            "direction": "lehte" if stance == "supports_higher" else "aleyhte",
            "strength": v.strength,
            "market_prob": round(market.get(selection, 0.0), 4),
            "videos": sorted(videos),
            "reasoning": v.reasoning,
        }
        prev = best.get(selection)
        if prev is None or (_STRENGTH[v.strength], len(videos)) > (_STRENGTH[prev["strength"]], len(prev["videos"])):
            best[selection] = sig
    order = ["home", "draw", "away", "over", "under"]
    return [best[k] for k in order if k in best]


def recommend(signals: list[dict], market: dict, hist: dict | None, best_prices: dict) -> dict:
    """En uygun seçenek: video sinyali lehte + tarihsel benzer oranlar çelişmiyor.

    Puan: video gücü (orta=1, güçlü=2) + tarihsel gerçekleşme şimdiki adil
    olasılığın belirgin üstündeyse +1 + benzer oranlarda sabit bahis getirisi
    pozitifse +1. Tarihsel veri açıkça aleyhteyse seçenek elenir.
    """
    candidates = []
    for s in signals:
        if s["direction"] != "lehte":
            continue
        sel = s["selection"]
        score = _STRENGTH[s["strength"]]
        reasons = [f"video kanıtı lehte ({STRENGTH_TR[s['strength']]}, {len(s['videos'])} video)"]
        fair = market.get(sel)
        if hist and sel in hist.get("rates", {}) and fair is not None:
            rate, roi = hist["rates"][sel], hist["roi"].get(sel)
            n = hist["n_ou"] if sel in ("over", "under") else hist["n_h2h"]
            if rate < fair - 0.01:
                continue  # Tarihsel benzer oranlar bu seçeneğin aleyhine
            if rate > fair + 0.02:
                score += 1
                reasons.append(f"benzer oranlı {n} maçta gerçekleşme %{rate * 100:.0f} (oranın gerektirdiğinden yüksek)")
            else:
                reasons.append(f"benzer oranlı {n} maçta gerçekleşme %{rate * 100:.0f} (oranla uyumlu)")
            if roi is not None and roi > 0:
                score += 1
                reasons.append(f"bu oran aralığında sabit bahis getirisi %{roi * 100:+.1f}")
        price = best_prices.get(sel)
        candidates.append({"selection": sel, "label": s["label"], "score": score, "reasons": reasons,
                           "best_odds": price[0] if price else None, "bookmaker": price[1] if price else None})
    if not candidates:
        reason = "Video kanıtı ile tarihsel oran istatistiği aynı seçeneği desteklemiyor." if signals else "Belirgin sinyal yok."
        return {"selection": None, "label": "Pas", "reasons": [reason], "confidence": None}
    top = max(candidates, key=lambda c: c["score"])
    top["confidence"] = "yüksek" if top["score"] >= 4 else "orta" if top["score"] >= 2 else "düşük"
    return top


def gemini_brief(home: str, away: str, kickoff: datetime, context: dict, client=None, model: str | None = None) -> GeminiBrief:
    from google import genai
    from google.genai import types

    from dualodds.video.watcher import DEFAULT_MODEL, with_retries

    model = model or os.environ.get("DUALODDS_GEMINI_MODEL", DEFAULT_MODEL)
    client = client or genai.Client()
    prompt = PROMPT.format(
        home=entities.team(home).name,
        away=entities.team(away).name,
        kickoff=kickoff.strftime("%Y-%m-%d %H:%M UTC"),
        context=json.dumps(context, ensure_ascii=False, default=str),
    )
    resp = with_retries(
        lambda: client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=GeminiBrief),
        )
    )
    return resp.parsed if isinstance(resp.parsed, GeminiBrief) else GeminiBrief.model_validate_json(resp.text)


BOOK_NAMES = {"pinnacle": "Pinnacle", "betfair_ex_eu": "Betfair Exchange", "betfair_ex_uk": "Betfair Exchange", "matchbook": "Matchbook"}


def _book(key: str | None) -> str:
    if not key:
        return ""
    return BOOK_NAMES.get(key, key.replace("_", " ").title())


def _odds_section(report: dict) -> list[str]:
    odds, best = report["odds"], report.get("best_prices") or {}
    if not odds.get("h2h"):
        return ["## Oranlar", "Oran bulunamadı.", ""]
    ref = dict(zip(("home", "draw", "away"), odds["h2h"]))
    if odds.get("totals_25"):
        ref.update(zip(("over", "under"), odds["totals_25"]))
    rows = ["| Seçenek | Oran (" + (report["odds_source"] or "") + ") | En iyi oran |", "|---|---|---|"]
    for sel in ("home", "draw", "away", "over", "under"):
        if sel not in ref:
            continue
        b = best.get(sel)
        best_txt = f"{b[0]:.2f} ({_book(b[1])})" if b else "-"
        rows.append(f"| {SELECTION_TR[sel]} | {ref[sel]:.2f} | {best_txt} |")
    return ["## Oranlar", *rows, ""]


def _history_section(hist: dict | None) -> list[str]:
    if not hist:
        return ["## Tarihsel benzer oranlar", "Arşiv verisi yok.", ""]
    r, roi, rg = hist["rates"], hist["roi"], hist["ranges"]
    lines = [
        "## Tarihsel benzer oranlar",
        f"Son 10 sezonda 6 büyük ligde bu maça en yakın oranlarla oynanan **{hist['n_h2h']} maç** "
        f"(ev sahibi oranı {rg['home'][0]:.2f}–{rg['home'][1]:.2f}, deplasman {rg['away'][0]:.2f}–{rg['away'][1]:.2f}):",
        "",
        "| Sonuç | Gerçekleşme | Sabit bahis getirisi* |",
        "|---|---|---|",
    ]
    for sel in ("home", "draw", "away", "over", "under"):
        if sel in r:
            lines.append(f"| {SELECTION_TR[sel]} | %{r[sel] * 100:.0f} | %{roi.get(sel, 0) * 100:+.1f} |")
    lines.append(f"| Karşılıklı gol var | %{r['btts'] * 100:.0f} | - |")
    if hist.get("n_ou"):
        lines.append(f"\n2.5 Alt/Üst satırları benzer Alt/Üst oranlı {hist['n_ou']} maçtan.")
    lines.append("En sık skorlar: " + ", ".join(f"{sc} (%{p * 100:.0f})" for sc, p in hist["top_scores"]))
    lines += ["", "_*O maçlara kendi oranlarından 1'er birim oynansaydı ortalama getiri._", ""]
    return lines


def render_markdown(report: dict) -> str:
    b = report["brief"]
    lines = [f"# {report['home_name']} – {report['away_name']}", f"_{report['kickoff']}_", ""]
    if b:
        lines += [f"**{b['headline']}**", "", b["summary"], ""]
        lines += [f"**{report['home_name']}:** {b['home_style']}", "", f"**{report['away_name']}:** {b['away_style']}", ""]
        if b["key_matchups"]:
            lines += ["## Belirleyici eşleşmeler", *[f"- {x}" for x in b["key_matchups"]], ""]
    lines += _odds_section(report)
    lines += _history_section(report.get("history"))
    lines += ["## Video sinyalleri (süzgeç sonrası)"]
    if report["signals"]:
        for s in report["signals"]:
            lines.append(f"- **{s['label']}**: video kanıtı {s['direction']} ({STRENGTH_TR[s['strength']]}, {len(s['videos'])} video). {s['reasoning']}")
    else:
        lines.append("- Videolarda orandan ayrışan, tekrarlayan ve güçlü bir kanıt yok.")
    rec = report.get("recommendation") or {}
    lines += ["", "## En uygun seçenek"]
    if rec.get("selection"):
        price = f" — en iyi oran **{rec['best_odds']:.2f}** ({_book(rec['bookmaker'])})" if rec.get("best_odds") else ""
        lines.append(f"**{rec['label']}**{price} · güven: **{rec['confidence']}**")
        lines += [f"- {x}" for x in rec["reasons"]]
    else:
        lines.append(f"**Pas** — {rec.get('reasons', ['Belirgin sinyal yok.'])[0]}")
    if b:
        if b["risks"]:
            lines += ["", "## Riskler", *[f"- {r}" for r in b["risks"]]]
        if b["data_gaps"]:
            lines += ["", "## Eksik / eski bilgi", *[f"- {g}" for g in b["data_gaps"]]]
        lines += ["", f"Analiz güveni: **{b['confidence']}**"]
    if report.get("error"):
        lines += ["", f"_Not: {report['error']}_"]
    lines += ["", f"Kaynak videolar: {report['n_videos']} · _Analizdir, kazanç garantisi yoktur. Sistemin isabeti henüz ölçülmedi; küçük ve sabit miktarlarla kullanın. 18+_"]
    return "\n".join(lines)
