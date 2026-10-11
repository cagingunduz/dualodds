"""Veri kaynağı arayüzü ve kayıt defteri.

Platformun değeri, çok sayıda kaynağın tek bir omurgada birleşmesinden gelir.
Her kaynak bir `DataSource` olarak yazılır ve lisans/kullanım koşulları
metadatası ile birlikte kaydedilir; böylece hangi verinin ticari üründe
kullanılabileceği kod düzeyinde izlenebilir.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class SourceInfo:
    name: str
    kind: str  # "results_odds", "events", "tracking", "news", "weather", ...
    license: str  # Kısa lisans/kullanım koşulu notu
    commercial_use: bool  # Ücretli üründe kullanılabilir mi (sözleşmeye göre)?
    url: str = ""
    notes: list[str] = field(default_factory=list)


class DataSource(ABC):
    info: SourceInfo

    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        """Ham veriyi çekip kanonik şemaya çevrilmiş DataFrame döndürür."""


_REGISTRY: dict[str, type[DataSource]] = {}


def register_source(cls: type[DataSource]) -> type[DataSource]:
    _REGISTRY[cls.info.name] = cls
    return cls


def get_source(name: str) -> type[DataSource]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Bilinmeyen kaynak: {name}. Kayıtlı: {sorted(_REGISTRY)}") from None


def registered_sources() -> list[SourceInfo]:
    return [cls.info for cls in _REGISTRY.values()]
