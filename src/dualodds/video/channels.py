"""İzlenen YouTube kanalları.

İki tür kanal var:
- highlights: Maç özetleri (maç başına olay analizi, watcher.py)
- tactical:   Taktik analiz videoları (takım/teknik direktör profili, tactical.py)

Kanallar adla değil, değişmeyen kanal ID'siyle tanınır. ID'ler ilk çalıştırmada
handle (ya da gerekirse arama) ile bulunur ve önbelleğe yazılır. Doğrulamak için:
`dualodds video channels`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from dualodds.data.entities import normalize
from dualodds.video.youtube import YouTube


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    kind: str  # "highlights" | "tactical"
    handles: tuple[str, ...] = ()  # Denenecek handle tahminleri; tutmazsa adla aranır
    channel_id: str | None = None  # Biliniyorsa sabitlenir (en güvenilir)


@dataclass(frozen=True)
class Channel:
    name: str
    kind: str
    channel_id: str
    title: str
    uploads_playlist: str


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("Sky Sports Premier League", "highlights", ("@SkySportsPremierLeague", "@skysportspl")),
    ChannelSpec("Tifo Football by The Athletic", "tactical", ("@tifo", "@TifoFootball")),
    ChannelSpec("Football Made Simple", "tactical", ("@FootballMadeSimple", "@footballmadesimple1")),
    ChannelSpec("Football Meta", "tactical", ("@footballmeta",)),
    ChannelSpec("The Coaches' Voice", "tactical", ("@TheCoachesVoice", "@coachesvoice")),
    ChannelSpec("Adam Clery", "tactical", ("@AdamClery", "@adamclery")),
)


def _title_matches(expected: str, title: str) -> bool:
    e, t = normalize(expected), normalize(title)
    # "Tifo Football by The Athletic" ↔ "Tifo Football" gibi kısmi eşleşmelere izin ver
    core = e.split(" by ")[0]
    return core in t or t in e


def _to_channel(spec: ChannelSpec, item: dict) -> Channel:
    return Channel(
        name=spec.name,
        kind=spec.kind,
        channel_id=item["id"],
        title=item["snippet"]["title"],
        uploads_playlist=item["contentDetails"]["relatedPlaylists"]["uploads"],
    )


def resolve_channel(spec: ChannelSpec, yt: YouTube) -> Channel:
    if spec.channel_id:
        item = yt.channel_by_id(spec.channel_id)
        if item:
            return _to_channel(spec, item)
    for handle in spec.handles:
        item = yt.channel_by_handle(handle)
        if item and _title_matches(spec.name, item["snippet"]["title"]):
            return _to_channel(spec, item)
    # Son çare: adla ara (100 birim), başlığı eşleşen ilk kanalı al
    for hit in yt.search_channels(spec.name):
        if _title_matches(spec.name, hit["snippet"]["title"]):
            item = yt.channel_by_id(hit["id"]["channelId"])
            if item:
                return _to_channel(spec, item)
    raise LookupError(f"Kanal bulunamadı: {spec.name}")


def resolve_channels(yt: YouTube, cache_path: Path, specs: tuple[ChannelSpec, ...] = CHANNELS, log=print) -> list[Channel]:
    """Önbellekteki kanalları kullanır, eksikleri çözer ve önbelleği günceller."""
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    out = []
    for spec in specs:
        if spec.name in cache:
            out.append(Channel(**cache[spec.name]))
            continue
        try:
            ch = resolve_channel(spec, yt)
        except LookupError as e:
            log(str(e))
            continue
        log(f"Kanal bulundu: {spec.name} → '{ch.title}' ({ch.channel_id})")
        cache[spec.name] = asdict(ch)
        out.append(ch)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
