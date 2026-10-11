"""Tekrarlanan veri işleri: maç verisini yenileme ve video taraması.

CLI ve sunucu (zamanlayıcı + API) aynı fonksiyonları kullanır.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

from dualodds.config import Settings
from dualodds.data.schema import played, validate_matches
from dualodds.data.sources import FootballDataUK
from dualodds.video.store import VideoStore

DEFAULT_CHANNELS_CACHE = Path("data/lake/channels.json")

Log = Callable[[str], None]


def season_code(d: date) -> str:
    """Avrupa sezon kodu: Temmuz ve sonrası yeni sezon. 2026-10-11 → '2627'."""
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def refresh_matches(settings: Settings, today: date | None = None, seasons: int = 2, source=None, log: Log = print) -> pd.DataFrame:
    """Son `seasons` sezonun sonuç + oranlarını indirip ham dizine kaydeder."""
    today = today or date.today()
    source = source or FootballDataUK()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for back in range(seasons):
        code = season_code(date(today.year - back, today.month, min(today.day, 28)))
        try:
            df = source.fetch(league=settings.league, season=code)
        except Exception as e:  # Sezon dosyası henüz yayınlanmamış olabilir
            log(f"{settings.league} {code}: indirilemedi ({type(e).__name__}: {e})")
            continue
        df.to_csv(settings.raw_dir / f"{settings.league}_{code}.csv", index=False)
        log(f"{settings.league} {code}: {len(df)} maç kaydedildi")
        frames.append(df)
    if not frames:
        raise RuntimeError("Hiçbir sezon indirilemedi")
    return validate_matches(pd.concat(frames, ignore_index=True))


def load_matches(settings: Settings) -> pd.DataFrame:
    """Ham dizindeki tüm sezonları birleştirir."""
    files = sorted(settings.raw_dir.glob(f"{settings.league}_*.csv"))
    if not files:
        raise RuntimeError(f"Maç verisi yok: {settings.raw_dir}. Önce veri yenileme işini çalıştırın.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return validate_matches(df).drop_duplicates(subset=["date", "home", "away"], keep="last").reset_index(drop=True)


def video_scan(
    matches: pd.DataFrame,
    store: VideoStore,
    since: date | str,
    until: date | str | None = None,
    team: str | None = None,
    limit: int = 0,
    fps: float | None = None,
    finder: Callable | None = None,
    watcher: Callable | None = None,
    log: Log = print,
    channels_cache: Path = DEFAULT_CHANNELS_CACHE,
) -> dict:
    """Aralıktaki oynanmış maçların özetlerini bul → izlet → kaydet.

    Daha önce analiz edilen maçlar atlanır. Bir maçtaki hata taramayı durdurmaz.
    """
    if finder is None and not os.environ.get("YOUTUBE_API_KEY"):
        raise RuntimeError("YOUTUBE_API_KEY tanımlı değil")
    if watcher is None:
        _require_gemini()
        from dualodds.video.watcher import watch_highlights as watcher

    df = played(matches)
    df = df[df["date"] >= pd.Timestamp(since)]
    if until is not None:
        df = df[df["date"] <= pd.Timestamp(until)]
    if team:
        df = df[(df["home"] == team) | (df["away"] == team)]

    log(f"{pd.Timestamp(since).date()} sonrası {len(df)} oynanmış maç taranacak")
    todo = [m for _, m in df.iterrows() if not store.has(str(m["date"].date()), m["home"], m["away"])]
    if finder is None and todo:
        finder = default_finder(channels_cache, log)  # Yalnızca yapılacak iş varsa YouTube'a git
    summary: dict = {"matches_in_window": len(df), "analyzed": [], "skipped_existing": 0, "not_found": [], "mismatched": [], "errors": []}
    for _, m in df.iterrows():
        d = str(m["date"].date())
        label = f"{d} {m['home']} – {m['away']}"
        if store.has(d, m["home"], m["away"]):
            summary["skipped_existing"] += 1
            continue
        if limit and len(summary["analyzed"]) + len(summary["mismatched"]) + len(summary["errors"]) >= limit:
            log(f"Limit ({limit}) doldu, tarama durduruldu")
            break
        try:
            candidates = finder(m["home"], m["away"], d)
            if not candidates:
                log(f"{label}: özet bulunamadı")
                summary["not_found"].append(label)
                continue
            best = candidates[0]
            score = f"{m['home_goals']}-{m['away_goals']}"
            record = watcher(best.url, m["home"], m["away"], d, score=score, video_title=best.title, channel=best.channel, fps=fps)
            store.save(record)
            a = record.analysis
            entry = {"match": label, "video": best.url, "observed_score": a.observed_score, "events": len(a.events)}
            if a.video_matches_fixture:
                summary["analyzed"].append(entry)
                log(f"{label}: {len(a.events)} pozisyon, görülen skor {a.observed_score} ({best.channel})")
            else:
                summary["mismatched"].append(entry)
                log(f"{label}: video maçla UYUŞMUYOR, profillere dahil edilmeyecek ({best.url})")
        except Exception as e:
            log(f"{label}: HATA {type(e).__name__}: {e}")
            summary["errors"].append({"match": label, "error": f"{type(e).__name__}: {e}"})
    return summary


def _require_gemini() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError("GEMINI_API_KEY tanımlı değil")


def resolve_default_channels(channels_cache: Path, log: Log = print, yt=None):
    from dualodds.video.channels import resolve_channels
    from dualodds.video.youtube import YouTube

    yt = yt or YouTube()
    return yt, resolve_channels(yt, channels_cache, log=log)


def default_finder(channels_cache: Path, log: Log = print):
    from dualodds.video.finder import HighlightFinder

    yt, channels = resolve_default_channels(channels_cache, log)
    return HighlightFinder(yt, channels)


def tactical_job(
    settings: Settings,
    since: date | str | None = None,
    limit: int | None = None,
    yt=None,
    analyzer=None,
    log: Log = print,
) -> dict:
    """Taktik kanallarının `since` sonrası videolarını analiz eder."""
    from dualodds.video.tactical import TacticalStore, classify_uploads, tactical_scan

    if analyzer is None:
        _require_gemini()
    if yt is None and not os.environ.get("YOUTUBE_API_KEY"):
        raise RuntimeError("YOUTUBE_API_KEY tanımlı değil")
    yt, channels = resolve_default_channels(settings.channels_cache, log, yt)
    since = since or date.today() - timedelta(days=settings.scan_days)
    since_dt = datetime.combine(pd.Timestamp(since).date(), time.min, tzinfo=timezone.utc)
    return tactical_scan(
        yt, channels, TacticalStore(settings.tactical_dir), since_dt,
        limit=settings.tactical_limit if limit is None else limit, analyzer=analyzer,
        # Gerçek çalıştırmada başlıklar Gemini ile sınıflandırılır; testlerde anahtar kelime filtresi
        classifier=classify_uploads if analyzer is None else None, log=log,
    )


def weekly_job(
    settings: Settings, today: date | None = None, log: Log = print, finder=None, watcher=None, source=None,
    yt=None, tactical_analyzer=None,
) -> dict:
    """Zamanlanmış iş: veriyi yenile, son `scan_days` günün maç özetlerini ve taktik videolarını analiz et."""
    today = today or date.today()
    matches = refresh_matches(settings, today=today, source=source, log=log)
    result = {
        "highlights": video_scan(
            matches,
            VideoStore(settings.video_dir),
            since=today - timedelta(days=settings.scan_days),
            limit=settings.scan_limit,
            finder=finder,
            watcher=watcher,
            log=log,
            channels_cache=settings.channels_cache,
        )
    }
    try:
        result["tactical"] = tactical_job(settings, since=today - timedelta(days=settings.scan_days), yt=yt, analyzer=tactical_analyzer, log=log)
    except Exception as e:  # Taktik tarama hatası özet taramasını boşa çıkarmasın
        log(f"Taktik tarama HATA {type(e).__name__}: {e}")
        result["tactical"] = {"error": f"{type(e).__name__}: {e}"}
    return result
