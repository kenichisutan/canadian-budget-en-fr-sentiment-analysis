#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

# Allow `python scripts/budget_corpus/cli.py` without PYTHONPATH
# scripts/ contains the budget_corpus package
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from budget_corpus.discover import load_config
from budget_corpus.download import USER_AGENT, download_year_lang
from budget_corpus.merge import merge_year_lang


def _parse_years(spec: str, cfg: dict) -> list[int]:
    skip = set(cfg.get("skip_years", []))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b, *rest = part.split("-")
            if rest:
                raise ValueError(f"Invalid year token: {part}")
            start, end = int(a), int(b)
            if start > end:
                start, end = end, start
            for y in range(start, end + 1):
                if y not in skip:
                    out.append(y)
        else:
            y = int(part)
            if y not in skip:
                out.append(y)
    return sorted(set(out))


def cmd_download(args: argparse.Namespace) -> int:
    cfg = load_config()
    years = _parse_years(args.years, cfg)
    langs = [args.lang] if args.lang else ["fr", "en"]
    client = httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=60.0,
        follow_redirects=True,
    )
    try:
        for year in years:
            for lang in langs:
                download_year_lang(year, lang, delay=args.delay, cfg=cfg, client=client)
    finally:
        client.close()
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    cfg = load_config()
    years = _parse_years(args.years, cfg)
    langs = [args.lang] if args.lang else ["fr", "en"]
    seps = not args.no_separators
    for year in years:
        if year in cfg.get("skip_years", []):
            print(f"SKIP merge year={year}")
            continue
        for lang in langs:
            hp, tp = merge_year_lang(year, lang, separators=seps)
            print(f"Wrote {hp} and {tp}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Canadian federal budget HTML corpus tools")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="Discover and download raw HTML")
    d.add_argument(
        "--years",
        required=True,
        help="Comma-separated years or ranges, e.g. 2024 or 2015-2019,2021-2025",
    )
    d.add_argument("--lang", choices=["fr", "en"], help="Single language (default: both)")
    d.add_argument("--delay", type=float, default=1.0, help="Seconds between HTTP requests")
    d.set_defaults(func=cmd_download)

    m = sub.add_parser("merge", help="Merge downloaded HTML into combined HTML + UTF-8 text")
    m.add_argument("--years", required=True, help="Years spec (same as download)")
    m.add_argument("--lang", choices=["fr", "en"])
    m.add_argument(
        "--no-separators",
        action="store_true",
        help="Omit <<< source: filename >>> markers in .txt output",
    )
    m.set_defaults(func=cmd_merge)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
