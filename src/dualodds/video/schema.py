"""Video gözlem şeması.

Kod alanları (enum) İngilizce ve sabittir, böylece maçlar arasında sayılabilir
ve modele özellik olarak girebilir. Serbest metin alanları Türkçedir.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Side = Literal["home", "away"]
Channel = Literal["left", "center", "right", "unknown"]

EventType = Literal[
    "goal",
    "disallowed_goal",
    "big_chance",
    "shot_saved",
    "shot_off_target",
    "woodwork",
    "penalty_awarded",
    "red_card",
    "other",
]

ChanceOrigin = Literal[
    "build_up",  # Geriden sabırlı oyun kurma
    "counter_attack",  # Hızlı hücum / kontra
    "high_press_turnover",  # Önde baskıyla top kazanma
    "cross",
    "cutback",  # Çizgiden geriye pas
    "through_ball",
    "long_ball",
    "corner",
    "free_kick",
    "penalty",
    "individual_dribble",
    "long_shot",
    "defensive_error",
    "rebound",
    "other",
]


class ChanceEvent(BaseModel):
    video_timestamp: str = Field(description="Videodaki an, mm:ss")
    match_minute: int | None = Field(description="Skor tabelasında görünüyorsa maç dakikası, yoksa null")
    team: Side = Field(description="Pozisyonu üreten takım")
    event_type: EventType
    origin: ChanceOrigin = Field(description="Pozisyonun nasıl doğduğu")
    attack_channel: Channel = Field(description="Atağın geliştiği koridor (hücum eden takımın bakışıyla)")
    players: list[str] = Field(description="Ekranda/tabelada/spikerden tanınabilen oyuncular; emin değilsen boş bırak")
    description: str = Field(description="Pozisyonun kısa Türkçe anlatımı")
    defensive_issue: str | None = Field(description="Savunan takımın hatası/zaafı (ör. 'arka direk boş kaldı'), yoksa null")
    chance_quality: Literal["low", "medium", "high"]


class TeamObservations(BaseModel):
    pressing: Literal["high", "mid", "low", "unclear"]
    build_up: Literal["short", "long", "mixed", "unclear"]
    main_attack_channels: list[Literal["left", "center", "right"]]
    observed_formation: str | None = Field(description="Görüntüden makul güvenle çıkarılabiliyorsa diziliş, yoksa null")
    strengths: list[str] = Field(description="Videoda kanıtı olan güçlü yönler (Türkçe)")
    weaknesses: list[str] = Field(description="Videoda kanıtı olan zaaflar (Türkçe)")
    set_piece_notes: str | None


class MatchVideoAnalysis(BaseModel):
    video_matches_fixture: bool = Field(description="Video gerçekten belirtilen iki takımın belirtilen maçı mı?")
    observed_score: str | None = Field(description="Videoda görülen son skor 'ev-deplasman', ör. '2-1'; görülemiyorsa null")
    home: TeamObservations
    away: TeamObservations
    events: list[ChanceEvent]
    tactical_summary: str = Field(description="Maçın taktik hikâyesi, 1-2 paragraf, Türkçe")
    limitations: str = Field(description="Bu özetten çıkarılamayanlar ve güvenilirlik notu, Türkçe")


class StoredVideoAnalysis(BaseModel):
    """Saklanan kayıt: analiz + kaynak metadatası (izlenebilirlik için)."""

    match_date: str
    home: str  # Kanonik takım anahtarı
    away: str
    video_url: str
    video_title: str = ""
    channel: str = ""
    analyzer_model: str
    analyzed_at: str
    analysis: MatchVideoAnalysis
