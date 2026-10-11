"""Taktik analiz videoları (Tifo, Football Made Simple, Football Meta,
The Coaches' Voice, Adam Clery...) → takım bazlı taktik profiller.

Bu kanallar maç özeti değil, uzman yorumu yayınlar: oyun kurma, pres, teknik
direktörün sistemi, takımın hangi oyun tarzlarına karşı zorlandığı. Gemini
videoyu izler ve videoda ele alınan her Premier League takımı için
yapılandırılmış bir profil çıkarır.

Notlar:
- Bunlar ikincil kaynak (yorumcu görüşü) ve tarihe bağlıdır: taktikler teknik
  direktör ve sezonla değişir, bu yüzden yayın tarihi ve teknik direktör saklanır.
- İçerik yeniden yayınlanmaz; yalnızca çıkarılan bilgi ve kaynak linki tutulur.
"""

from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from dualodds.data import entities
from dualodds.video.channels import Channel
from dualodds.video.watcher import DEFAULT_MODEL, DailyQuotaExceeded, gemini_video_json, with_retries
from dualodds.video.youtube import Upload, YouTube

StyleTag = Literal[
    "high_press",
    "man_to_man_press",
    "mid_block",
    "low_block",
    "possession_build_up",
    "direct_long_ball",
    "counter_attack",
    "wide_overloads",
    "crossing",
    "central_combinations",
    "inverted_fullbacks",
    "high_defensive_line",
    "set_piece_threat",
    "transition_vulnerable",
]

STYLE_LABELS_TR = {
    "high_press": "önde yoğun pres",
    "man_to_man_press": "adam adama pres",
    "mid_block": "orta blok",
    "low_block": "derin blok",
    "possession_build_up": "geriden sabırlı oyun kurma",
    "direct_long_ball": "direkt / uzun top",
    "counter_attack": "hızlı hücum / kontra",
    "wide_overloads": "kanatlarda sayısal üstünlük",
    "crossing": "orta ağırlıklı hücum",
    "central_combinations": "merkezden kombinasyon",
    "inverted_fullbacks": "içe kat eden bekler",
    "high_defensive_line": "yüksek savunma hattı",
    "set_piece_threat": "duran top tehdidi",
    "transition_vulnerable": "geçişlerde açık veren",
}


class TeamTactics(BaseModel):
    team: str = Field(description="Verilen listedeki takım anahtarı (aynen)")
    manager: str | None = Field(description="Videoda ele alınan teknik direktör, yoksa null")
    period_covered: str | None = Field(description="Analizin kapsadığı dönem (ör. '2026/27 ilk 6 hafta'), belirsizse null")
    style_tags: list[StyleTag] = Field(description="Takımın oyun tarzını tanımlayan etiketler")
    effective_against: list[StyleTag] = Field(description="Videoya göre bu takımın iyi sonuç aldığı rakip tarzları")
    vulnerable_to: list[StyleTag] = Field(description="Videoya göre bu takımı zorlayan rakip tarzları")
    shape_in_possession: str | None
    shape_out_of_possession: str | None
    attacking_patterns: list[str] = Field(description="Hücum örüntüleri, Türkçe")
    defensive_weaknesses: list[str] = Field(description="Savunma zaafları, Türkçe")
    key_player_roles: list[str] = Field(description="Kritik oyuncu rolleri, Türkçe ('Oyuncu: rol')")
    summary: str = Field(description="Bu takım için 2-4 cümlelik özet, Türkçe")
    evidence_timestamps: list[str] = Field(description="İddiaları destekleyen video anları, mm:ss")


class TacticalVideoAnalysis(BaseModel):
    video_type: Literal["team_analysis", "manager_profile", "match_review", "match_preview", "player_role", "concept", "other"]
    about_premier_league: bool = Field(description="Video Premier League takımlarını esaslı biçimde ele alıyor mu?")
    teams: list[TeamTactics] = Field(description="Videoda esaslı biçimde analiz edilen Premier League takımları; geçerken anılanlar hariç")
    key_takeaways: list[str] = Field(description="Videonun ana fikirleri, Türkçe")
    confidence: Literal["low", "medium", "high"]


class StoredTacticalAnalysis(BaseModel):
    video_id: str
    video_url: str
    title: str
    channel: str
    published_at: str
    analyzer_model: str
    analyzed_at: str
    analysis: TacticalVideoAnalysis


PROMPT = """Sen elit bir futbol taktik analistisin. Bu video "{channel}" kanalında {published} tarihinde yayınlandı. Başlık: "{title}".

Görevin: Videoda esaslı biçimde analiz edilen her Premier League takımı için taktik profili çıkar.

Takım anahtarları (team alanına aynen bunlardan birini yaz): {teams}

Kurallar:
- Yalnızca videoda söylenen veya gösterilenlere dayan. Videoda olmayanı ekleme; bilmiyorsan alanı boş liste ya da null bırak.
- Geçerken anılan takımları ekleme; yalnızca üzerine analiz yapılanları ekle.
- Taktikler teknik direktöre ve döneme bağlıdır: teknik direktörü ve analiz edilen dönemi mutlaka belirt.
- effective_against / vulnerable_to alanlarını yalnızca video bu konuda bir şey söylüyorsa veya gösterdiği örneklerden açıkça çıkıyorsa doldur.
- Video Premier League ile ilgili değilse about_premier_league=false ve teams=[] döndür.
- Serbest metin alanlarını Türkçe yaz."""


def mentioned_teams(text: str) -> list[str]:
    return [key for key in entities.TEAMS if entities.mentions(text, key)]


def keyword_relevant(up: Upload) -> bool:
    """Yedek filtre: başlık/açıklamada PL takımı veya 'Premier League' geçiyor mu?"""
    text = f"{up.title}\n{up.description}"
    return bool(mentioned_teams(text)) or "premier league" in entities.normalize(text)


class VideoRelevance(BaseModel):
    video_id: str
    tactical: bool = Field(description="Takım/teknik direktör taktiği veya oyun tarzı üzerine esaslı analiz mi?")
    premier_league: bool = Field(description="Premier League takımlarını ele alıyor mu?")
    teams: list[str] = Field(description="Ele alınan PL takım anahtarları (teknik direktör adından da çıkar)")


class RelevanceBatch(BaseModel):
    items: list[VideoRelevance]


CLASSIFY_PROMPT = """Aşağıdaki YouTube videolarını yalnızca başlık ve açıklamalarına bakarak sınıflandır.

tactical=true yalnızca video bir takımın, teknik direktörün veya maçın TAKTİK/oyun tarzı analiziyse.
tactical=false: bilgi yarışmaları, oyuncu hikâyeleri, transfer, hukuk/ceza davaları, finans, canlı yayın sohbetleri, sponsorlu içerik, genel futbol kültürü.
premier_league=true: video Premier League takımlarını ele alıyorsa (teknik direktör adıyla anılsa bile, ör. "Arteta's" → Arsenal).
teams: ele alınan takımlar, yalnızca şu anahtarlardan: {teams}

Videolar (JSON):
{videos}"""


def classify_uploads(uploads: list[Upload], client=None, model: str | None = None, batch_size: int = 40) -> dict[str, VideoRelevance]:
    """Video izletmeden önce ucuz bir metin çağrısıyla alakasızları eler (video başına birkaç yüz token)."""
    import json

    from google import genai
    from google.genai import types

    model = model or os.environ.get("DUALODDS_GEMINI_MODEL", DEFAULT_MODEL)
    client = client or genai.Client()
    out: dict[str, VideoRelevance] = {}
    for i in range(0, len(uploads), batch_size):
        chunk = uploads[i : i + batch_size]
        videos = [{"video_id": u.video_id, "title": u.title, "description": u.description[:400]} for u in chunk]
        prompt = CLASSIFY_PROMPT.format(teams=", ".join(entities.TEAMS), videos=json.dumps(videos, ensure_ascii=False))
        resp = with_retries(
            lambda: client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=RelevanceBatch),
            )
        )
        batch = resp.parsed if isinstance(resp.parsed, RelevanceBatch) else RelevanceBatch.model_validate_json(resp.text)
        out.update({item.video_id: item for item in batch.items})
    return out


def analyze_tactical_video(upload: Upload, channel: str, fps: float | None = None, model: str | None = None, client=None) -> StoredTacticalAnalysis:
    model = model or os.environ.get("DUALODDS_GEMINI_MODEL", DEFAULT_MODEL)
    teams = ", ".join(f"{k} ({t.name})" for k, t in entities.TEAMS.items())
    prompt = PROMPT.format(channel=channel, published=upload.published_at.date(), title=upload.title, teams=teams)
    analysis = gemini_video_json(upload.url, prompt, TacticalVideoAnalysis, fps=fps, model=model, client=client)
    # Model listede olmayan bir ad yazdıysa eşleştirmeye çalış, olmazsa at
    fixed = []
    for t in analysis.teams:
        if t.team in entities.TEAMS:
            fixed.append(t)
        else:
            match = mentioned_teams(t.team)
            if match:
                fixed.append(t.model_copy(update={"team": match[0]}))
    analysis = analysis.model_copy(update={"teams": fixed})
    return StoredTacticalAnalysis(
        video_id=upload.video_id,
        video_url=upload.url,
        title=upload.title,
        channel=channel,
        published_at=upload.published_at.isoformat(timespec="seconds"),
        analyzer_model=model,
        analyzed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        analysis=analysis,
    )


class TacticalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for(self, video_id: str) -> Path:
        return self.root / f"{video_id}.json"

    def has(self, video_id: str) -> bool:
        return self.path_for(video_id).exists()

    def save(self, record: StoredTacticalAnalysis) -> Path:
        path = self.path_for(record.video_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return path

    def all(self) -> list[StoredTacticalAnalysis]:
        if not self.root.exists():
            return []
        records = [StoredTacticalAnalysis.model_validate_json(p.read_text(encoding="utf-8")) for p in self.root.glob("*.json")]
        return sorted(records, key=lambda r: r.published_at)

    def for_team(self, team: str, max_age_days: int = 180, last_n: int = 6, as_of: datetime | None = None) -> list[dict]:
        """Takım hakkındaki en yeni taktik notlar (yeniden eskiye)."""
        as_of = as_of or datetime.now(timezone.utc)
        cutoff = as_of - timedelta(days=max_age_days)
        notes = []
        for r in reversed(self.all()):
            published = datetime.fromisoformat(r.published_at)
            if published < cutoff or published > as_of:
                continue
            for t in r.analysis.teams:
                if t.team == team:
                    notes.append(
                        {
                            "published_at": r.published_at[:10],
                            "channel": r.channel,
                            "title": r.title,
                            "video_url": r.video_url,
                            "video_type": r.analysis.video_type,
                            "confidence": r.analysis.confidence,
                            **t.model_dump(exclude={"team"}),
                        }
                    )
        return notes[:last_n]


def style_profile(notes: list[dict]) -> dict:
    """Birden çok videodaki etiketleri sayar; birden fazla kaynakta geçen etiket daha güvenilir."""
    def count(field):
        return dict(Counter(tag for n in notes for tag in n[field]).most_common())

    return {"style": count("style_tags"), "effective_against": count("effective_against"), "vulnerable_to": count("vulnerable_to")}


def tactical_matchup(home: str, away: str, home_notes: list[dict], away_notes: list[dict]) -> dict:
    """Bir takımın oyun tarzı, rakibinin zorlandığı tarzlarla örtüşüyor mu?"""
    hp, ap = style_profile(home_notes), style_profile(away_notes)

    def edges(attacker, attacker_p, defender_p):
        out = []
        for tag in attacker_p["style"]:
            if tag in defender_p["vulnerable_to"]:
                out.append(
                    {
                        "team": attacker,
                        "style": tag,
                        "label": STYLE_LABELS_TR.get(tag, tag),
                        "sources": min(attacker_p["style"][tag], defender_p["vulnerable_to"][tag]),
                    }
                )
        return out

    return {
        "home_profile": hp,
        "away_profile": ap,
        "home_edges": edges(home, hp, ap),
        "away_edges": edges(away, ap, hp),
        "sources": {"home": len(home_notes), "away": len(away_notes)},
        "caveat": "Etiketler yorumcu videolarından çıkarılmıştır; tek kaynaklı eşleşmeler zayıf sinyaldir, "
        "teknik direktör değiştiyse eski notlar geçersiz olabilir.",
    }


def tactical_scan(
    yt: YouTube,
    channels: list[Channel],
    store: TacticalStore,
    since: datetime,
    limit: int = 0,
    min_duration_s: int = 120,
    max_duration_s: int = 45 * 60,
    analyzer=None,
    classifier=None,
    teams_filter: set[str] | None = None,
    log=print,
) -> dict:
    """Taktik kanallarının yeni videolarını tara; Premier League taktik analizlerini izlet.

    `teams_filter` verilirse yalnızca bu takımları (ya da lig genelini) ele alan videolar izlenir.

    `limit` başarılı + hatalı tüm Gemini video denemelerini sayar (maliyet sınırı).
    `classifier` verilmezse anahtar kelime filtresi kullanılır.
    """
    analyzer = analyzer or analyze_tactical_video
    tactical = [c for c in channels if c.kind == "tactical"]
    summary: dict = {"analyzed": [], "skipped_existing": 0, "skipped_not_relevant": [], "skipped_duration": [], "errors": []}

    candidates: list[tuple[Channel, Upload]] = []
    for ch in tactical:
        ups = list(yt.uploads(ch.uploads_playlist, since))
        log(f"{ch.title}: {since.date()} sonrası {len(ups)} video")
        candidates += [(ch, up) for up in ups]

    fresh = []
    for ch, up in candidates:
        if store.has(up.video_id):
            summary["skipped_existing"] += 1
        else:
            fresh.append((ch, up))

    relevance = None
    if classifier is not None and fresh:
        try:
            relevance = classifier([up for _, up in fresh])
        except DailyQuotaExceeded:
            raise
        except Exception as e:  # Sınıflandırıcı çökerse anahtar kelime filtresine düş
            log(f"Sınıflandırma HATA {type(e).__name__}: {e}; anahtar kelime filtresi kullanılıyor")

    todo = []
    for ch, up in fresh:
        if relevance is not None:
            r = relevance.get(up.video_id)
            ok = bool(r and r.tactical and r.premier_league)
            teams = set(r.teams) if r else set()
        else:
            ok = keyword_relevant(up)
            teams = set(mentioned_teams(f"{up.title}\n{up.description}"))
        # Takım belirtilmemiş lig geneli videolar ("Her PL takımının zaafı") her zaman ilgili sayılır
        if ok and teams_filter and teams and not (teams & teams_filter):
            ok = False
        if ok:
            todo.append((ch, up))
        else:
            summary["skipped_not_relevant"].append(f"{ch.title}: {up.title}")

    durations = yt.durations([up.video_id for _, up in todo]) if todo else {}
    for ch, up in sorted(todo, key=lambda x: x[1].published_at):
        if limit and len(summary["analyzed"]) + len(summary["errors"]) >= limit:
            log(f"Limit ({limit}) doldu, tarama durduruldu")
            break
        dur = durations.get(up.video_id)
        if dur is not None and not (min_duration_s <= dur <= max_duration_s):
            summary["skipped_duration"].append(f"{ch.title}: {up.title} ({dur // 60} dk)")
            continue
        try:
            record = analyzer(up, ch.title)
            store.save(record)
            teams = [t.team for t in record.analysis.teams]
            summary["analyzed"].append({"channel": ch.title, "title": up.title, "url": up.url, "teams": teams})
            log(f"{ch.title}: '{up.title}' → {', '.join(teams) or 'PL takımı analizi yok'}")
        except DailyQuotaExceeded as e:
            log(f"DURDU: {e}")
            summary["quota_exhausted"] = str(e)
            break
        except Exception as e:
            log(f"{ch.title}: '{up.title}' HATA {type(e).__name__}: {e}")
            summary["errors"].append({"video": up.url, "error": f"{type(e).__name__}: {e}"})
    summary["youtube_units_used"] = yt.units_used
    return summary
