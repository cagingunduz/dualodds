"""Abonelere sunulan maç raporunun şeması.

Sayısal alanlar (olasılıklar, beklenen goller, değer analizi) her zaman
kodla, modelden doldurulur. LLM yalnızca yorum, gerekçe ve risk alanlarını
yazar. Böylece raporlardaki hiçbir sayı "uydurulmuş" olamaz.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ValueAssessment(BaseModel):
    outcome: str
    odds: float
    model_prob: float
    market_fair_prob: float
    edge: float
    expected_value: float
    kelly_stake_pct: float


class MatchReport(BaseModel):
    home: str
    away: str
    model_as_of: str
    probabilities: dict = Field(description="Model çıktısı (1X2, alt/üst, KG, olası skorlar)")
    value: list[ValueAssessment] = Field(default_factory=list)

    # LLM tarafından yazılan alanlar
    headline: str = ""
    summary: str = ""
    key_factors: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: str = ""  # "düşük" | "orta" | "yüksek"
    lean: str = ""  # Analistin eğilimi; modelle çelişiyorsa gerekçe summary'de olmalı

    def to_markdown(self) -> str:
        p = self.probabilities
        lines = [
            f"# {self.home} – {self.away}",
            f"**{self.headline}**" if self.headline else "",
            "",
            f"Model (veri: {self.model_as_of}): 1 %{p['home'] * 100:.1f} · X %{p['draw'] * 100:.1f} · 2 %{p['away'] * 100:.1f}",
            f"Beklenen gol: {p['exp_home_goals']:.2f} – {p['exp_away_goals']:.2f} · "
            f"2.5 Üst %{p['over']['2.5'] * 100:.1f} · KG Var %{p['btts'] * 100:.1f}",
            "",
            self.summary,
        ]
        if self.key_factors:
            lines += ["", "## Öne çıkan faktörler", *[f"- {f}" for f in self.key_factors]]
        if self.risks:
            lines += ["", "## Riskler", *[f"- {r}" for r in self.risks]]
        if self.value:
            lines += ["", "## Değer analizi", "| Sonuç | Oran | Model | Piyasa (adil) | Avantaj | Kelly |", "|---|---|---|---|---|---|"]
            for v in self.value:
                lines.append(
                    f"| {v.outcome} | {v.odds:.2f} | %{v.model_prob * 100:.1f} | %{v.market_fair_prob * 100:.1f} | "
                    f"{v.edge * 100:+.1f} puan | %{v.kelly_stake_pct * 100:.1f} |"
                )
        if self.confidence:
            lines += ["", f"Güven: **{self.confidence}**" + (f" · Eğilim: **{self.lean}**" if self.lean else "")]
        lines += ["", "_Bu içerik istatistiksel analizdir, kesin sonuç garantisi vermez. 18+ · Sorumlu oynayın._"]
        return "\n".join(lines)
