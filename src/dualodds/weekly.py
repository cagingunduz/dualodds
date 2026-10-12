"""Haftalık analiz: önümüzdeki maçlar → bu sezonun videoları → Gemini analizi → oran süzgeci.

    dualodds week --days 7 --max-videos 40

Ücretli veri sağlayıcısı kullanılmaz:
- Fikstür + oranlar: The Odds API (ODDS_API_KEY)
- Sezon sonuçları + maç özetleri: Sky Sports PL başlıkları ve videoları (YOUTUBE_API_KEY)
- Taktik: 5 taktik kanalın videoları
- Analiz: Gemini (GEMINI_API_KEY)
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from dualodds import brief as brief_mod
from dualodds.config import Settings
from dualodds.data import entities
from dualodds.data.sources.odds_api import upcoming_with_odds
from dualodds.history import last_seasons, load_history, similar_odds
from dualodds.pipeline import resolve_default_channels
from dualodds.video.highlight_index import season_highlights, team_results
from dualodds.video.profile import team_profile
from dualodds.video.store import VideoStore
from dualodds.video.tactical import TacticalStore, classify_uploads, tactical_matchup, tactical_scan
from dualodds.video.watcher import DailyQuotaExceeded


def _rec_text(rec: dict | None) -> str:
    if not rec or not rec.get("selection"):
        return "Pas"
    return f"{rec['label']} @ {rec['best_odds']:.2f} ({rec['confidence']})" if rec.get("best_odds") else rec["label"]


def season_start(d: date) -> datetime:
    year = d.year if d.month >= 7 else d.year - 1
    return datetime(year, 7, 1, tzinfo=timezone.utc)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _compact_highlight(rec) -> dict:
    a = rec.analysis
    return {
        "date": rec.match_date,
        "match": f"{rec.home} {a.observed_score or '?'} {rec.away}",
        "video_url": rec.video_url,
        "summary": a.tactical_summary,
        "home_obs": a.home.model_dump(),
        "away_obs": a.away.model_dump(),
        "events": [
            {
                "t": e.video_timestamp,
                "min": e.match_minute,
                "team": rec.home if e.team == "home" else rec.away,
                "type": e.event_type,
                "origin": e.origin,
                "channel": e.attack_channel,
                "defensive_issue": e.defensive_issue,
                "quality": e.chance_quality,
            }
            for e in a.events
        ],
    }


def build_context(team: str, refs, vstore: VideoStore, tstore: TacticalStore, since: datetime) -> dict:
    highlights = [r for r in vstore.for_team(team, last_n=38) if r.match_date >= since.date().isoformat()]
    return {
        "team": team,
        "season_results": team_results(refs, team),
        "highlight_profile": team_profile(team, highlights[-8:]),
        "highlight_matches": [_compact_highlight(r) for r in highlights[-6:]],
        "tactical_notes": tstore.for_team(team, max_age_days=(datetime.now(timezone.utc) - since).days + 1, last_n=8),
    }


def run_week(
    settings: Settings,
    days: int = 7,
    max_videos: int = 40,
    now: datetime | None = None,
    yt=None,
    odds_fetch=None,
    watcher=None,
    tactical_analyzer=None,
    classifier=None,
    briefer=None,
    history=None,
    out_dir: Path | None = None,
    log=print,
) -> dict:
    now = now or datetime.now(timezone.utc)
    since = season_start(now.date())
    vstore, tstore = VideoStore(settings.video_dir), TacticalStore(settings.tactical_dir)
    real = watcher is None  # Gerçek çalıştırma (testlerde sahte bileşenler verilir)
    if real:
        import os

        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise RuntimeError("GEMINI_API_KEY tanımlı değil")
    if watcher is None:
        from dualodds.video.watcher import watch_highlights as watcher
    if real and classifier is None:
        classifier = classify_uploads
    briefer = briefer or brief_mod.gemini_brief

    # 1) Fikstür + oranlar
    fixtures, quota = upcoming_with_odds(fetch=odds_fetch)
    fixtures = [f for f in fixtures if now <= f.commence_time <= now + timedelta(days=days)]
    log(f"Önümüzdeki {days} gün: {len(fixtures)} maç (Odds API kalan kota: {quota.get('x-requests-remaining')})")
    teams = {t for f in fixtures for t in (f.home, f.away)}
    unknown = sorted(t for t in teams if t not in entities.TEAMS)
    if unknown:
        log(f"UYARI: tanınmayan takım adları (entities.py'ye ekleyin): {unknown}")

    # 2) Bu sezonun maç özetleri (başlıklardan maç + skor)
    yt, channels = resolve_default_channels(settings.channels_cache, log, yt)
    refs = season_highlights(yt, channels, since)
    log(f"Sezon başından beri {len(refs)} maç özeti bulundu")

    # 3) Bu haftanın takımlarının izlenmemiş özetleri. Bütçenin ~%70'i özetlere; takımlar arasında
    #    eşit dağıtılır (her takımın en yeni maçı önce, sonra ikinci en yeni maçı...)
    highlight_budget = max(1, round(max_videos * 0.7)) if max_videos else 0
    rank: dict[str, int] = {}
    order = []
    for r in reversed(refs):  # Yeniden eskiye
        involved = teams & {r.home, r.away}
        if not involved:
            continue
        k = min(rank.get(t, 0) for t in involved)
        for t in involved:
            rank[t] = rank.get(t, 0) + 1
        order.append((k, r))
    todo = [r for k, r in sorted(order, key=lambda x: (x[0], -x[1].published_at.timestamp()))
            if not vstore.has(r.date, r.home, r.away)]
    watched, errors = [], []
    used = 0
    quota_hit = None
    for r in todo:
        if used >= highlight_budget:
            log(f"Özet bütçesi doldu; {len(todo) - used} özet sonraki çalıştırmaya kaldı")
            break
        used += 1
        try:
            rec = watcher(r.url, r.home, r.away, r.date, score=r.score, video_title=r.title, channel=r.channel)
            vstore.save(rec)
            watched.append(f"{r.date} {r.home} {r.score} {r.away}")
            log(f"İzlendi: {r.title}")
        except DailyQuotaExceeded as e:
            quota_hit = str(e)
            log(f"DURDU: {quota_hit}")
            break
        except Exception as e:
            errors.append({"video": r.url, "error": f"{type(e).__name__}: {e}"})
            log(f"HATA {r.title}: {type(e).__name__}: {e}")

    # 4) Taktik videolar (yalnızca bu haftanın takımları + lig geneli), kalan bütçeyle
    budget = max_videos - used
    tactical = {"analyzed": []}
    if budget > 0 and not quota_hit:
        tactical = tactical_scan(
            yt, channels, tstore, since, limit=budget, analyzer=tactical_analyzer,
            classifier=classifier, teams_filter=teams, log=log,
        )
        errors += tactical.get("errors", [])
        quota_hit = quota_hit or tactical.get("quota_exhausted")

    # 5) Tarihsel benzer oranlar arşivi (ücretsiz; ilk seferde indirilir, sonra önbellekten)
    if history is None:
        try:
            history = load_history(settings.data_dir / "history", seasons=last_seasons(since.year), log=log)
            log(f"Tarihsel oran arşivi: {len(history)} maç")
        except Exception as e:
            log(f"Tarihsel oran arşivi yüklenemedi: {type(e).__name__}: {e}")

    # 6) Maç başına Gemini analizi + oran süzgeci + en uygun seçenek
    out_dir = out_dir or settings.data_dir / "reports" / f"week_{now.date().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for f in fixtures:
        market = brief_mod.market_probabilities(f.h2h, f.totals_25)
        hist = similar_odds(history, f.h2h, f.totals_25) if history is not None else None
        hist_d = hist.to_dict() if hist else None
        context = {
            "odds": {"h2h_home_draw_away": f.h2h, "over_under_2_5": f.totals_25},
            "market_fair_probabilities": market,
            "historical_similar_odds": hist_d,
            "home": build_context(f.home, refs, vstore, tstore, since),
            "away": build_context(f.away, refs, vstore, tstore, since),
            "tactical_matchup": tactical_matchup(
                f.home, f.away, tstore.for_team(f.home), tstore.for_team(f.away)
            ),
        }
        allowed = brief_mod.allowed_video_urls(context)
        report = {
            "home": f.home,
            "away": f.away,
            "home_name": entities.team(f.home).name,
            "away_name": entities.team(f.away).name,
            "kickoff": f.commence_time.strftime("%Y-%m-%d %H:%M UTC"),
            "odds": {"h2h": f.h2h, "totals_25": f.totals_25},
            "odds_source": brief_mod._book(f.source) if f.source and not f.source.startswith("median") else (f.source or "yok"),
            "best_prices": f.best,
            "history": hist_d,
            "recommendation": None,
            "market": market,
            "n_videos": len(allowed),
            "brief": None,
            "signals": [],
        }
        if quota_hit:
            report["error"] = "Gemini günlük kotası doldu; analiz yazılamadı (yalnızca piyasa olasılıkları)."
        else:
            try:
                b = briefer(f.home, f.away, f.commence_time, context)
                report["brief"] = b.model_dump()
                report["signals"] = brief_mod.filter_signals(b, market, allowed)
                report["recommendation"] = brief_mod.recommend(report["signals"], market, hist_d, f.best)
            except DailyQuotaExceeded as e:
                quota_hit = str(e)
                report["error"] = "Gemini günlük kotası doldu; analiz yazılamadı (yalnızca piyasa olasılıkları)."
                log(f"DURDU: {quota_hit}")
            except Exception as e:
                report["error"] = f"{type(e).__name__}: {e}"
                log(f"Analiz HATA {f.home}-{f.away}: {report['error']}")
        stem = f"{f.commence_time.date()}_{_slug(f.home)}_{_slug(f.away)}"
        (out_dir / f"{stem}.json").write_text(json.dumps({**report, "context": context}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (out_dir / f"{stem}.md").write_text(brief_mod.render_markdown(report), encoding="utf-8")
        reports.append(report)
        log(f"{report['home_name']} – {report['away_name']}: {_rec_text(report.get('recommendation'))}")

    index = ["# Haftalık analiz", "", "| Maç | Saat (UTC) | En uygun seçenek | En iyi oran | Güven |", "|---|---|---|---|---|"]
    for r in reports:
        rec = r.get("recommendation") or {}
        odds_txt = f"{rec['best_odds']:.2f} ({brief_mod._book(rec.get('bookmaker'))})" if rec.get("best_odds") else "-"
        index.append(f"| {r['home_name']} – {r['away_name']} | {r['kickoff'][:16]} | {rec.get('label', 'Pas')} | {odds_txt} | {rec.get('confidence') or '-'} |")
    index += ["", "_Analizdir, kazanç garantisi yoktur. Sistemin isabeti henüz ölçülmedi. 18+_"]
    (out_dir / "INDEX.md").write_text("\n".join(index), encoding="utf-8")

    return {
        "fixtures": len(fixtures),
        "highlights_watched": watched,
        "tactical_analyzed": [a["title"] for a in tactical.get("analyzed", [])],
        "errors": errors,
        "reports_dir": str(out_dir),
        "signals": {f"{r['home']}-{r['away']}": r["signals"] for r in reports},
        "recommendations": {f"{r['home']}-{r['away']}": r.get("recommendation") for r in reports},
        "youtube_units_used": getattr(yt, "units_used", None),
        "odds_quota": quota,
        "gemini_quota_exhausted": quota_hit,
    }
