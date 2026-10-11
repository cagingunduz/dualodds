"""Varlık eşleştirme: farklı kaynaklardaki takım adlarını tek kimliğe bağlar.

Anahtar = football-data.co.uk yazımı (kanonik veri setimizdeki ad).
`name` tam/resmi ad, `aliases` başlık/metin eşleştirmede kullanılan diğer yazımlar.
Yeni kaynak eklendikçe bu tablo büyür; ileride veritabanına taşınacak.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    key: str
    name: str
    aliases: tuple[str, ...] = ()

    def all_names(self) -> tuple[str, ...]:
        return (self.key, self.name, *self.aliases)


_PREMIER_LEAGUE = [
    Team("Arsenal", "Arsenal"),
    Team("Aston Villa", "Aston Villa", ("Villa",)),
    Team("Bournemouth", "AFC Bournemouth"),
    Team("Brentford", "Brentford"),
    Team("Brighton", "Brighton & Hove Albion", ("Brighton and Hove Albion",)),
    Team("Burnley", "Burnley"),
    Team("Chelsea", "Chelsea"),
    Team("Coventry", "Coventry City"),
    Team("Crystal Palace", "Crystal Palace", ("Palace",)),
    Team("Everton", "Everton"),
    Team("Fulham", "Fulham"),
    Team("Hull", "Hull City"),
    Team("Ipswich", "Ipswich Town"),
    Team("Leeds", "Leeds United"),
    Team("Leicester", "Leicester City"),
    Team("Liverpool", "Liverpool"),
    Team("Luton", "Luton Town"),
    Team("Man City", "Manchester City", ("Man City", "Manchester C")),
    Team("Man United", "Manchester United", ("Man Utd", "Man United", "Manchester Utd")),
    Team("Newcastle", "Newcastle United"),
    Team("Nott'm Forest", "Nottingham Forest", ("Forest", "Nottm Forest", "Nott'm Forest")),
    Team("Sheffield United", "Sheffield United", ("Sheff Utd",)),
    Team("Southampton", "Southampton"),
    Team("Sunderland", "Sunderland"),
    Team("Tottenham", "Tottenham Hotspur", ("Spurs",)),
    Team("West Ham", "West Ham United"),
    Team("Wolves", "Wolverhampton Wanderers", ("Wolverhampton",)),
]

TEAMS: dict[str, Team] = {t.key: t for t in _PREMIER_LEAGUE}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def team(key: str) -> Team:
    """Tanınmayan takım için adı olduğu gibi kullanan bir kayıt döndürür."""
    return TEAMS.get(key, Team(key, key))


def key_for(name: str) -> str | None:
    """Herhangi bir kaynaktaki takım adını anahtara çevirir (tam eşleşme, takma adlar dahil)."""
    n = normalize(name)
    for t in TEAMS.values():
        if any(normalize(x) == n for x in t.all_names()):
            return t.key
    return None


def mentions(text: str, key: str) -> bool:
    """Metin (ör. video başlığı) bu takımdan bahsediyor mu?"""
    norm = f" {normalize(text)} "
    return any(f" {normalize(n)} " in norm for n in team(key).all_names())
