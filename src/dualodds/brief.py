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


def filter_signals(brief: GeminiBrief, market: dict, allowed_urls: set[str], min_videos: int = MIN_VIDEOS) -> list[dict]:
    """Süzgeç: güçlü/orta, ≥2 farklı gerçek videoya dayanan, piyasadan ayrışan görüşler."""
    signals = []
    for v in brief.market_views:
        if v.stance == "in_line" or v.strength == "weak" or v.selection not in market:
            continue
        videos = {e.video_url for e in v.evidence if e.video_url in allowed_urls}
        if len(videos) < min_videos:
            continue
        signals.append(
            {
                "selection": v.selection,
                "label": SELECTION_TR[v.selection],
                "direction": "yükseğe" if v.stance == "supports_higher" else "düşüğe",
                "strength": v.strength,
                "market_prob": round(market[v.selection], 4),
                "videos": sorted(videos),
                "reasoning": v.reasoning,
            }
        )
    return signals


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


def render_markdown(report: dict) -> str:
    b = report["brief"]
    m = report["market"]
    lines = [f"# {report['home_name']} – {report['away_name']}", f"_{report['kickoff']} · oranlar: {report['odds_source']}_", ""]
    if b:
        lines += [f"**{b['headline']}**", "", b["summary"], ""]
        lines += [f"**{report['home_name']}:** {b['home_style']}", "", f"**{report['away_name']}:** {b['away_style']}", ""]
        if b["key_matchups"]:
            lines += ["## Belirleyici eşleşmeler", *[f"- {x}" for x in b["key_matchups"]], ""]
    if m:
        parts = [f"{SELECTION_TR[k]} %{v * 100:.1f}" for k, v in m.items()]
        lines += ["## Piyasa (marjı temizlenmiş)", " · ".join(parts), ""]
    lines += ["## Süzgeç sonrası sonuç"]
    if report["signals"]:
        for s in report["signals"]:
            lines.append(
                f"- **{s['label']}**: piyasa %{s['market_prob'] * 100:.1f}; video kanıtı daha {s['direction']} "
                f"bir ihtimale işaret ediyor ({s['strength']}, {len(s['videos'])} video). {s['reasoning']}"
            )
    else:
        lines.append("- Piyasa ile uyumlu; videolarda piyasadan ayrışan güçlü ve tekrarlayan bir kanıt yok.")
    if b:
        if b["risks"]:
            lines += ["", "## Riskler", *[f"- {r}" for r in b["risks"]]]
        if b["data_gaps"]:
            lines += ["", "## Eksik / eski bilgi", *[f"- {g}" for g in b["data_gaps"]]]
        lines += ["", f"Güven: **{b['confidence']}**"]
    lines += ["", f"Kaynak videolar: {report['n_videos']} · _İstatistiksel analizdir, bahis tavsiyesi değildir. 18+_"]
    return "\n".join(lines)
