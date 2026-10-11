"""Maç analisti ajanı (Claude).

Ajan; model olasılıklarını, form ve geçmiş karşılaşma verisini, piyasa
oranlarını araçlar üzerinden toplar, kullanıcının verdiği nitel notları
(sakatlık, rotasyon, motivasyon vb.) değerlendirir ve sonunda
`submit_report` aracıyla yapılandırılmış rapor teslim eder.

Sayılar modelden gelir; ajan sayı üretmez, yorumlar.
"""

from __future__ import annotations

import json

import anthropic
from anthropic import beta_tool

from dualodds.agents import tools
from dualodds.agents.report import MatchReport, ValueAssessment

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """Sen DualOdds platformunun kıdemli futbol analistisin. Abonelere maç öncesi analiz raporu hazırlıyorsun.

Çalışma ilkeleri:
- Olasılıklar ve beklenen goller istatistiksel modelden gelir. Rakam uydurma; araçların döndürdüğü sayıları kullan.
- Senin katkın yorumdur: modelin neyi yakalayıp neyi kaçırabileceğini (sakatlık, rotasyon, motivasyon, stil eşleşmesi, takvim yoğunluğu) açıkla.
- Model ile piyasa arasında büyük fark varsa önce piyasanın bildiği bir şey olabileceğini düşün; farkı otomatik olarak "değer" sayma.
- Belirsizliği dürüstçe yaz. Kesin sonuç vaat eden ifadeler kullanma.
- Raporu Türkçe yaz ve işin sonunda mutlaka submit_report aracını çağır."""


class AnalystError(RuntimeError):
    pass


def _build_tools(ctx: tools.AnalysisContext, home: str, away: str, sink: dict):
    @beta_tool
    def get_model_probabilities() -> str:
        """Bu maç için Dixon-Coles modelinin olasılıklarını döndürür: 1X2, beklenen goller, alt/üst, karşılıklı gol, en olası skorlar ve takım güç puanları."""
        return json.dumps(tools.model_probabilities(ctx, home, away), ensure_ascii=False)

    @beta_tool
    def get_team_form(team: str, last_n: int = 6) -> str:
        """Bir takımın son maçlarını, puan ortalamasını ve gol sayılarını döndürür.

        Args:
            team: Takım adı (veri setindeki yazımıyla).
            last_n: Kaç maç geriye bakılacağı.
        """
        try:
            return json.dumps(tools.team_form(ctx, team, last_n), ensure_ascii=False)
        except KeyError as e:
            return json.dumps({"error": str(e), "known_teams": ctx.known_teams()}, ensure_ascii=False)

    @beta_tool
    def get_head_to_head(last_n: int = 6) -> str:
        """İki takım arasındaki son karşılaşmaları döndürür.

        Args:
            last_n: Kaç maç geriye bakılacağı.
        """
        return json.dumps(tools.head_to_head(ctx, home, away, last_n), ensure_ascii=False)

    @beta_tool
    def submit_report(headline: str, summary: str, key_factors: list[str], risks: list[str], confidence: str, lean: str) -> str:
        """Nihai analiz raporunu teslim eder. Analizi bitirdiğinde bir kez çağır.

        Args:
            headline: Tek cümlelik başlık.
            summary: 2-4 paragraflık analiz; modelin sayılarına ve topladığın verilere dayansın.
            key_factors: Maçı belirleyecek 3-6 faktör.
            risks: Analizi boşa çıkarabilecek 2-4 risk.
            confidence: "düşük", "orta" veya "yüksek".
            lean: Eğilimin (ör. "1", "X", "2", "2.5 Üst", "Pas"). Değer yoksa "Pas".
        """
        sink.update(
            headline=headline, summary=summary, key_factors=key_factors, risks=risks, confidence=confidence, lean=lean
        )
        return "Rapor alındı."

    return [get_model_probabilities, get_team_form, get_head_to_head, submit_report]


def analyze_match(
    ctx: tools.AnalysisContext,
    home: str,
    away: str,
    odds: tuple[float, float, float] | None = None,
    notes: str | None = None,
    client: anthropic.Anthropic | None = None,
    effort: str = "high",
) -> MatchReport:
    """Bir maç için yapılandırılmış analiz raporu üretir."""
    probs = tools.model_probabilities(ctx, home, away)  # Bilinmeyen takımda erken hata
    value: list[ValueAssessment] = []
    market = None
    if odds is not None:
        market = tools.market_comparison(ctx, home, away, *odds)
        value = [ValueAssessment(**row) for row in market["outcomes"]]

    prompt = [f"Maç: {home} (ev sahibi) – {away} (deplasman)."]
    if market is not None:
        prompt.append(f"Piyasa oranları ve modele göre karşılaştırma:\n{json.dumps(market, ensure_ascii=False)}")
    if notes:
        prompt.append(f"Editörden nitel notlar (doğrulanmamış olabilir):\n{notes}")
    prompt.append("Gerekli verileri araçlarla topla, analiz et ve submit_report ile raporu teslim et.")

    sink: dict = {}
    client = client or anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        # Güvenlik sınıflandırıcısı bir isteği reddederse sunucu tarafında yedek modele geçilir
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        cache_control={"type": "ephemeral"},
        tools=_build_tools(ctx, home, away, sink),
        messages=[{"role": "user", "content": "\n\n".join(prompt)}],
        max_iterations=12,
    )
    last = None
    for message in runner:
        last = message
    if last is not None and last.stop_reason == "refusal":
        raise AnalystError("Model isteği reddetti")
    if not sink:
        raise AnalystError("Ajan raporu teslim etmedi (submit_report çağrılmadı)")

    return MatchReport(
        home=home,
        away=away,
        model_as_of=probs["model_as_of"],
        probabilities={k: probs[k] for k in ("home", "draw", "away", "exp_home_goals", "exp_away_goals", "over", "btts", "top_scores")},
        value=value,
        **sink,
    )
