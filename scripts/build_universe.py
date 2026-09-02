#!/usr/bin/env python3
"""P1-15 (interim) — derive the capture universe from Alpaca's own asset list.

`universe.txt` decides what the snapshotter captures every night, which makes it
the single most consequential file in the repo on the free data path: a name
absent from it tonight has no history for tonight, ever, at any price.

Two rules follow from that, and both are enforced here rather than remembered.

  ADDITIVE ONLY   A symbol is never removed from the universe by this script.
                  When a company delists, its history stays in scope and the
                  snapshotter keeps asking for it; the day it stops returning
                  bars is itself a fact worth recording. Silently dropping the
                  name is the first move of survivorship bias, and it is exactly
                  the bias `truth.json` plants a trap for. Removals are printed
                  as a warning and require a human to edit the file.

  VALIDATED       Every symbol is checked against Alpaca's /v2/assets before it
                  lands, so an expired ticker or a typo fails here — loudly, in
                  a script you ran on purpose — instead of becoming a symbol
                  that quietly returns no bars for a year.

The seed committed to the repo is the S&P 500 as of the first commit. Run this
once your Alpaca keys exist to validate that seed and widen it.

    python scripts/build_universe.py --check          # report, change nothing
    python scripts/build_universe.py                  # validate + rewrite
    python scripts/build_universe.py --add-all-active # widen to all US equities

Widening is close to free: Alpaca's daily-bar endpoint is chunked at 200
symbols, so the whole active US equity list is ~30 requests a night. Capture
breadth now and narrow at research time, never the other way round.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

ASSETS_URL = "https://api.alpaca.markets/v2/assets"

# Venues whose daily bars are consolidated into the Alpaca feed. Anything else
# (OTC in particular) has price data too thin to build features on, and quietly
# poisons any liquidity-dependent CostModel that later reads it.
KEEP_EXCHANGES = frozenset({"NYSE", "NASDAQ", "ARCA", "AMEX", "BATS"})


def fetch_active_assets() -> dict[str, str]:
    """symbol -> exchange, for every tradable US equity Alpaca lists today."""
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        print(
            "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY are not set.\n"
            "This script validates against Alpaca's live asset list and cannot "
            "run without them. See P1-14b in the README.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    with httpx.Client(timeout=60.0) as client:
        r = client.get(
            ASSETS_URL,
            params={"status": "active", "asset_class": "us_equity"},
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        )
        r.raise_for_status()
        return {
            a["symbol"]: a["exchange"]
            for a in r.json()
            if a.get("tradable") and a.get("exchange") in KEEP_EXCHANGES
        }


def read_universe(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [s.strip() for s in path.read_text().split() if s.strip()]


def write_universe(path: Path, symbols: list[str]) -> None:
    """Sorted, one per line, trailing newline. Deterministic on purpose: the
    git diff of this file is the audit trail of when a name entered the
    universe, and an unstable ordering destroys that."""
    path.write_text("\n".join(sorted(set(symbols))) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--universe", type=Path, default=Path("universe.txt"))
    p.add_argument(
        "--check",
        action="store_true",
        help="report what would change and exit non-zero if anything would; write nothing",
    )
    p.add_argument(
        "--add-all-active",
        action="store_true",
        help="widen to every tradable US equity on a major venue, not just the current file",
    )
    a = p.parse_args()

    current = read_universe(a.universe)
    active = fetch_active_assets()

    unknown = sorted(set(current) - set(active))
    kept = sorted(set(current) & set(active))
    added = sorted(set(active) - set(current)) if a.add_all_active else []

    print(f"universe file:   {a.universe}  ({len(current)} symbols)")
    print(f"alpaca active:   {len(active)} tradable US equities on {sorted(KEEP_EXCHANGES)}")
    print(f"validated:       {len(kept)}")

    if unknown:
        # Not an error. A delisted name SHOULD still be here — see ADDITIVE ONLY.
        print(f"\nnot in Alpaca's active list ({len(unknown)}), kept anyway:")
        print("  " + " ".join(unknown))
        print(
            "  These are typos or delistings. Delistings belong in the universe;\n"
            "  typos do not. Decide by hand and edit the file — this script will not."
        )

    if added:
        print(f"\nwould add {len(added)} symbols")

    final = sorted(set(current) | set(added))

    if a.check:
        if added:
            print("\n--check: the file is behind Alpaca's active list.")
            return 1
        print("\n--check: nothing to add.")
        return 0

    if final == sorted(set(current)):
        print("\nno change.")
        return 0

    write_universe(a.universe, final)
    print(f"\nwrote {len(final)} symbols to {a.universe} (was {len(set(current))})")
    print("Commit this file. The diff is when each name entered the universe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
