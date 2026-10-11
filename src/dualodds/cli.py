"""Komut satırı arayüzü.

Örnekler:
    dualodds demo
    dualodds fetch --league T1 --season 2425 --out data/raw/T1_2425.csv
    dualodds backtest --csv data/raw/T1_2425.csv --start 2025-01-01
    dualodds predict --csv data/raw/T1_2425.csv --home Galatasaray --away Fenerbahce
    dualodds analyze --csv data/raw/T1_2425.csv --home Galatasaray --away Fenerbahce --odds 2.1 3.5 3.3

    # Premier League maç özeti videoları (Gemini + YouTube Data API)
    dualodds video find --home Arsenal --away Chelsea --date 2026-09-20
    dualodds video watch --url https://www.youtube.com/watch?v=... --home Arsenal --away Chelsea --date 2026-09-20
    dualodds video scan --csv data/raw/E0_2627.csv --since 2026-09-01
    dualodds video profile --team Arsenal
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from dualodds.data.schema import validate_matches
from dualodds.data.sources import FootballDataUK
from dualodds.data.synthetic import generate_league
from dualodds.evaluation import walk_forward
from dualodds.models import DixonColesModel
from dualodds.video.store import DEFAULT_DIR, VideoStore


def _load(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="latin-1")
    if "HomeTeam" in raw.columns:
        return FootballDataUK.from_file(path)
    return validate_matches(raw)


def cmd_demo(_: argparse.Namespace) -> None:
    df, _truth = generate_league(seasons=2)
    split = df["date"].iloc[len(df) // 2]
    print(f"Sentetik lig: {len(df)} maç. Backtest başlangıcı: {split.date()}\n")
    print(walk_forward(df, start=split))


def cmd_fetch(args: argparse.Namespace) -> None:
    df = FootballDataUK().fetch(league=args.league, season=args.season)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} maç kaydedildi: {args.out}")


def cmd_backtest(args: argparse.Namespace) -> None:
    df = _load(args.csv)
    result = walk_forward(df, start=args.start, edge_threshold=args.edge)
    print(result)


def cmd_predict(args: argparse.Namespace) -> None:
    model = DixonColesModel().fit(_load(args.csv))
    print(json.dumps(model.predict(args.home, args.away).to_dict(), indent=2, ensure_ascii=False))


def cmd_analyze(args: argparse.Namespace) -> None:
    from dualodds.agents.analyst import analyze_match
    from dualodds.agents.tools import AnalysisContext

    store = VideoStore(args.video_dir) if args.video_dir else None
    ctx = AnalysisContext.build(_load(args.csv), video_store=store)
    report = analyze_match(ctx, args.home, args.away, odds=tuple(args.odds) if args.odds else None, notes=args.notes)
    print(report.to_markdown())


def cmd_video_find(args: argparse.Namespace) -> None:
    from dualodds.video.finder import find_highlights

    for c in find_highlights(args.home, args.away, args.date):
        print(f"{c.score:5.2f}  {c.published_at:%Y-%m-%d}  {c.channel:<28} {c.title}\n       {c.url}")


def _watch_and_store(store: VideoStore, url: str, home: str, away: str, date: str, score: str | None, title: str = "", channel: str = "", fps: float | None = None):
    from dualodds.video.watcher import watch_highlights

    record = watch_highlights(url, home, away, date, score=score, video_title=title, channel=channel, fps=fps)
    path = store.save(record)
    a = record.analysis
    status = "doğrulandı" if a.video_matches_fixture else "UYUŞMUYOR (profillere dahil edilmeyecek)"
    print(f"{home} – {away} [{status}] skor: {a.observed_score} · {len(a.events)} pozisyon → {path}")
    return record


def cmd_video_watch(args: argparse.Namespace) -> None:
    store = VideoStore(args.video_dir)
    record = _watch_and_store(store, args.url, args.home, args.away, args.date, args.score, fps=args.fps)
    print()
    print(record.analysis.tactical_summary)


def cmd_video_scan(args: argparse.Namespace) -> None:
    from dualodds.data.schema import played
    from dualodds.video.finder import find_highlights

    store = VideoStore(args.video_dir)
    df = played(_load(args.csv))
    df = df[df["date"] >= pd.Timestamp(args.since)]
    if args.until:
        df = df[df["date"] <= pd.Timestamp(args.until)]
    if args.team:
        df = df[(df["home"] == args.team) | (df["away"] == args.team)]
    done = 0
    for _, m in df.iterrows():
        date = str(m["date"].date())
        if store.has(date, m["home"], m["away"]):
            continue
        if args.limit and done >= args.limit:
            break
        try:
            candidates = find_highlights(m["home"], m["away"], date)
            if not candidates:
                print(f"{date} {m['home']} – {m['away']}: özet bulunamadı")
                continue
            best = candidates[0]
            _watch_and_store(
                store, best.url, m["home"], m["away"], date, f"{m['home_goals']}-{m['away_goals']}",
                title=best.title, channel=best.channel, fps=args.fps,
            )
            done += 1
        except Exception as e:  # Bir maçtaki hata taramayı durdurmasın
            print(f"{date} {m['home']} – {m['away']}: HATA {type(e).__name__}: {e}")
    print(f"\n{done} yeni maç analiz edildi.")


def cmd_video_profile(args: argparse.Namespace) -> None:
    from dualodds.video.profile import team_profile

    store = VideoStore(args.video_dir)
    print(json.dumps(team_profile(args.team, store.for_team(args.team, last_n=args.last_n)), indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dualodds")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="Sentetik veriyle uçtan uca backtest").set_defaults(fn=cmd_demo)

    f = sub.add_parser("fetch", help="football-data.co.uk'tan lig sezonu indir")
    f.add_argument("--league", required=True)
    f.add_argument("--season", required=True)
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_fetch)

    b = sub.add_parser("backtest", help="Walk-forward backtest")
    b.add_argument("--csv", required=True)
    b.add_argument("--start", required=True)
    b.add_argument("--edge", type=float, default=0.03)
    b.set_defaults(fn=cmd_backtest)

    for name, fn, helptext in (
        ("predict", cmd_predict, "Model olasılıkları"),
        ("analyze", cmd_analyze, "Claude analist ajanı ile rapor (ANTHROPIC_API_KEY gerekir)"),
    ):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--csv", required=True)
        s.add_argument("--home", required=True)
        s.add_argument("--away", required=True)
        if name == "analyze":
            s.add_argument("--odds", type=float, nargs=3, metavar=("EV", "BERABERLIK", "DEPLASMAN"))
            s.add_argument("--notes", help="Sakatlık, rotasyon vb. nitel notlar")
            s.add_argument("--video-dir", help=f"Video analiz deposu (ör. {DEFAULT_DIR}); verilirse ajan video izci profilini kullanır")
        s.set_defaults(fn=fn)

    v = sub.add_parser("video", help="Maç özeti videolarını izleyen ajan (Gemini)")
    vsub = v.add_subparsers(dest="video_cmd", required=True)

    vf = vsub.add_parser("find", help="YouTube'da özet videosu ara (YOUTUBE_API_KEY)")
    vw = vsub.add_parser("watch", help="Tek bir videoyu izlet ve kaydet (GEMINI_API_KEY)")
    vs = vsub.add_parser("scan", help="CSV'deki maçların özetlerini bul, izlet, kaydet")
    vp = vsub.add_parser("profile", help="Takımın birleşik video izci profili")
    for sp in (vf, vw):
        sp.add_argument("--home", required=True)
        sp.add_argument("--away", required=True)
        sp.add_argument("--date", required=True, help="Maç tarihi, YYYY-AA-GG")
    vw.add_argument("--url", required=True)
    vw.add_argument("--score", help="Resmî skor, ör. 2-1 (doğrulama için)")
    vs.add_argument("--csv", required=True)
    vs.add_argument("--since", required=True)
    vs.add_argument("--until")
    vs.add_argument("--team")
    vs.add_argument("--limit", type=int, default=0, help="En fazla kaç yeni maç (0 = sınırsız)")
    vp.add_argument("--team", required=True)
    vp.add_argument("--last-n", type=int, default=5)
    for sp in (vw, vs):
        sp.add_argument("--fps", type=float, help="Saniyedeki kare (varsayılan 1); artırmak maliyeti artırır")
    for sp in (vw, vs, vp):
        sp.add_argument("--video-dir", default=str(DEFAULT_DIR))
    vf.set_defaults(fn=cmd_video_find)
    vw.set_defaults(fn=cmd_video_watch)
    vs.set_defaults(fn=cmd_video_scan)
    vp.set_defaults(fn=cmd_video_profile)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except RuntimeError as e:  # Eksik API anahtarı vb. kullanıcı hataları
        raise SystemExit(f"Hata: {e}") from None


if __name__ == "__main__":
    main()
