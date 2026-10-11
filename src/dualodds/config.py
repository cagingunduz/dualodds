"""Ortam değişkenlerinden okunan ayarlar (Railway, yerel, test)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    league: str = "E0"  # football-data.co.uk lig kodu; E0 = Premier League
    admin_token: str | None = None
    scheduler_enabled: bool = False
    schedule_cron: str = "0 6 * * 2,5"  # Salı (hafta sonu maçları) ve Cuma (hafta içi maçları) 06:00
    timezone: str = "Europe/Istanbul"
    scan_days: int = 8  # Zamanlanmış taramada geriye kaç gün bakılacağı
    scan_limit: int = 20  # Tek taramada en fazla kaç yeni maç özeti (maliyet sınırı)
    tactical_limit: int = 10  # Tek taramada en fazla kaç yeni taktik video

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def video_dir(self) -> Path:
        return self.data_dir / "lake" / "video"

    @property
    def tactical_dir(self) -> Path:
        return self.data_dir / "lake" / "tactical"

    @property
    def channels_cache(self) -> Path:
        return self.data_dir / "lake" / "channels.json"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        data_dir = env.get("DUALODDS_DATA_DIR") or env.get("RAILWAY_VOLUME_MOUNT_PATH") or "data"
        return cls(
            data_dir=Path(data_dir),
            league=env.get("DUALODDS_LEAGUE", "E0"),
            admin_token=env.get("ADMIN_TOKEN") or None,
            scheduler_enabled=_bool(env.get("SCHEDULER_ENABLED"), False),
            schedule_cron=env.get("SCHEDULE_CRON", cls.schedule_cron),
            timezone=env.get("SCHEDULE_TZ", cls.timezone),
            scan_days=int(env.get("SCAN_DAYS", cls.scan_days)),
            scan_limit=int(env.get("SCAN_LIMIT", cls.scan_limit)),
            tactical_limit=int(env.get("TACTICAL_LIMIT", cls.tactical_limit)),
        )
