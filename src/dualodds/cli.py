"""Komut satırı arayüzü.

Örnekler:
    dualodds demo
    dualodds fetch --league T1 --season 2425 --out data/raw/T1_2425.csv
    dualodds backtest --csv data/raw/T1_2425.csv --start 2025-01-01
    dualodds predict --csv data/raw/T1_2425.csv --home Galatasaray --away Fenerbahce
    dualodds analyze --csv data/raw/T1_2425.csv --home Galatasaray --away Fenerbahce --odds 2.1 3.5 3.3
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

    ctx = AnalysisContext.build(_load(args.csv))
    report = analyze_match(ctx, args.home, args.away, odds=tuple(args.odds) if args.odds else None, notes=args.notes)
    print(report.to_markdown())


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
        s.set_defaults(fn=fn)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
