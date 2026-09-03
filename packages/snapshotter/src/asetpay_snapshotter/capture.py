"""P1-05 / P1-08 — the daily snapshot.

Fifty lines of real work, and the only genuinely time-sensitive thing in the
project: on the free data path, history not captured today cannot be bought back
at any price. This runs before anything consumes it, and it runs tonight.

Three properties matter more than the fetching:

  IMMUTABLE   a snapshot is written once and never edited. A correction is a new
              snapshot with a later knowledge_date, never an overwrite. That is
              what makes the point-in-time store possible downstream.

  MANIFESTED  every capture records provider, parameters, row count, checksum,
              fetch time, schema version and universe coverage. Without the
              manifest you cannot tell a day with genuinely no data from a day
              the fetch silently failed — and those need very different
              responses.

  ATTRIBUTED  the manifest records WHICH source produced the rows. Volume from
              a single venue and consolidated volume are not the same number;
              a history assembled from both is fine, but only if the seam is
              visible. See sources.py.

TWO DATES, NEVER ONE
--------------------
`event_date` is when the trading happened. `knowledge_date` is when WE could
first have known about it. They are different, and collapsing them is the
lookahead this whole project exists to prevent.

The capture runs at 03:00 UTC on Wednesday and fetches Tuesday's session. Nobody
possessed Tuesday's closing price at any point during Tuesday. Stamping those
rows `knowledge_date = Tuesday` would make `as_of(Tuesday)` hand a strategy a
number that did not exist yet — a few hours of lookahead, applied uniformly, in
the direction that flatters every backtest.

So `event_date` comes from the bar itself and `knowledge_date` is the date the
capture ran. `as_of(Tuesday)` therefore cannot see Tuesday's close, and the
one-day trade lag stops being a convention someone has to remember. It is the
same reasoning as the GiST exclusion constraints: structural, not disciplinary.

The checksum is computed over canonically sorted, canonically typed rows so that
re-running on identical input produces an identical digest. That determinism is
tested (P1-08) because without it `features_hash` downstream becomes unstable,
which presents as a data bug and takes a week to find.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from asetpay_snapshotter.sources import PriceSource, get_source

# 2: added vwap and trade_count, and made the source pluggable.
# 3: knowledge_date is the capture date, not the session date. Bumped rather
# than edited in place — a reader of an old snapshot must be able to tell which
# shape they are holding, and this change alters what the rows MEAN, which is
# the version bump that matters most.
SCHEMA_VERSION = 3

# Below this share of the requested universe, the capture is treated as a failed
# fetch rather than a quiet day. Individual names halt, get acquired, or stop
# trading, so this is deliberately not 100%. A truncated response, an expired
# entitlement or a half-open market looks nothing like a few halts.
MIN_UNIVERSE_COVERAGE = 0.90


@dataclass(frozen=True)
class Manifest:
    """What was captured, from where, and proof it is intact."""

    snapshot_id: str
    knowledge_date: str  # when we could first have known this
    session: str  # when the trading actually happened
    provider: str
    parameters: dict[str, object]
    row_count: int
    universe_size: int  # every name ever, including delisted
    expected_trading: int  # those that should appear on this session
    universe_coverage: float  # measured against expected_trading, not universe_size
    missing_symbols: list[str]
    checksum: str
    fetched_at: str
    schema_version: int

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))


def canonical_checksum(df: pd.DataFrame) -> str:
    """A digest that depends only on the CONTENT, not on incidental ordering.

    Sorting the columns and rows, and forcing a fixed float repr, is what makes
    two runs over the same input agree. Without it, dict iteration order and
    BLAS-dependent float formatting make the digest wobble between runs.
    """
    if df.empty:
        return hashlib.sha256(b"").hexdigest()
    d = df.reindex(sorted(df.columns), axis=1)
    d = d.sort_values(by=list(d.columns), kind="mergesort").reset_index(drop=True)
    payload = d.to_csv(index=False, float_format="%.10g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def previous_session(today: date | None = None) -> date:
    """The most recent weekday strictly before `today`.

    A real implementation consults the exchange calendar; weekdays are enough
    until the universe includes a market with different holidays, at which point
    this becomes a genuine bug and should be replaced rather than patched.
    """
    d = today or datetime.now(tz=UTC).date()
    d = date.fromordinal(d.toordinal() - 1)
    while d.weekday() >= 5:
        d = date.fromordinal(d.toordinal() - 1)
    return d


def read_delistings(path: Path) -> dict[str, date]:
    """symbol -> last trading date, from `universe_delisted.txt`.

    Comment and blank lines ignored; the third field is a human-readable reason
    that nothing parses and everything benefits from.
    """
    if not path.exists():
        return {}
    out: dict[str, date] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [f.strip() for f in line.split(",", 2)]
        if len(parts) < 2:
            raise SystemExit(f"malformed line in {path}: {line!r}")
        out[parts[0]] = date.fromisoformat(parts[1])
    return out


def expected_trading(
    symbols: list[str], delisted: dict[str, date], session: date
) -> list[str]:
    """The universe members that should actually appear on this session.

    A name is expected until its last trading day, inclusive, and never after.
    Measuring coverage against the full universe instead would make it decay
    with every corporate action — AvalonBay and Equity Residential merging into
    VMRK cost 0.4% on the first live capture — until the floor trips for reasons
    that are not failures at all.
    """
    return [s for s in symbols if s not in delisted or session <= delisted[s]]


# A coverage number tells you something is wrong. A list of names tells you
# what. Capped because a genuinely broken night would otherwise write thousands
# of symbols into every manifest, and the first twenty are enough to recognise
# the pattern — a ticker convention mismatch, a halted name, a dead listing.
MAX_MISSING_RECORDED = 20


def missing_from(df: pd.DataFrame, symbols: list[str]) -> list[str]:
    """Universe members the response did not contain.

    Persisted in the manifest so the diagnosis survives the run. `coverage
    0.996` sent someone to go and work out which two names it was; recording
    them means the next person reads the answer instead of deriving it.
    """
    got = set(df["symbol"]) if len(df) else set()
    return sorted(set(symbols) - got)


def universe_coverage(df: pd.DataFrame, symbols: list[str]) -> float:
    """Share of the requested universe that actually came back.

    Row count alone cannot distinguish a full market day from a response that
    returned ten thousand rows and none of the names you care about — which is
    exactly what a wrong date or a changed entitlement produces.
    """
    if not symbols:
        return 0.0
    got = set(df["symbol"]) if len(df) else set()
    return len(got & set(symbols)) / len(symbols)


def capture(
    out_dir: Path,
    session: date,
    symbols: list[str],
    source: PriceSource | None = None,
    knowledge_date: date | None = None,
    delisted: dict[str, date] | None = None,
) -> Manifest:
    """Capture one trading `session`, as known on `knowledge_date`.

    `knowledge_date` defaults to today (UTC), which is the honest answer when
    the job is running now. It is an argument rather than a constant so that a
    backfill can state plainly when the data became knowable, instead of
    pretending the backfill happened years ago.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = source or get_source()

    started = datetime.now(tz=UTC)
    known_on = knowledge_date or started.date()
    if known_on < session:
        raise SystemExit(
            f"knowledge_date {known_on} precedes the session {session} it describes. "
            "That is lookahead by construction; refusing to write it."
        )

    expected = expected_trading(symbols, delisted or {}, session)
    df = src.fetch(session, symbols)

    # knowledge_date is the axis the point-in-time store partitions by, and it
    # is stamped here rather than inferred later from a filename. event_date
    # came from the bar itself, in sources.py.
    df = df.copy()
    df["knowledge_date"] = known_on.isoformat()
    df.to_parquet(out_dir / "prices.parquet", index=False)

    coverage = universe_coverage(df, expected)
    missing = missing_from(df, expected)
    m = Manifest(
        snapshot_id=started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        knowledge_date=known_on.isoformat(),
        session=session.isoformat(),
        provider=src.provider,
        parameters={**src.parameters(), "n_symbols_requested": len(symbols)},
        row_count=len(df),
        universe_size=len(symbols),
        expected_trading=len(expected),
        universe_coverage=round(coverage, 4),
        missing_symbols=missing[:MAX_MISSING_RECORDED],
        checksum=f"sha256:{canonical_checksum(df)}",
        fetched_at=started.isoformat(),
        schema_version=SCHEMA_VERSION,
    )
    m.write(out_dir / "_manifest.json")
    print(
        f"captured {m.row_count} rows for session {m.session} "
        f"(known {m.knowledge_date}) from {m.provider} "
        f"(universe coverage {coverage:.1%}) -> {out_dir}"
    )
    if missing:
        shown = " ".join(missing[:MAX_MISSING_RECORDED])
        more = (
            f" (+{len(missing) - MAX_MISSING_RECORDED} more)"
            if len(missing) > MAX_MISSING_RECORDED
            else ""
        )
        print(f"  absent from the response: {shown}{more}")
    return m


def main() -> int:
    p = argparse.ArgumentParser(description="Daily immutable market-data capture")
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument(
        "--session",
        type=date.fromisoformat,
        default=None,
        help="trading day to capture; defaults to the previous weekday",
    )
    p.add_argument(
        "--knowledge-date",
        type=date.fromisoformat,
        default=None,
        help="when this became knowable; defaults to today. Only set it for backfills",
    )
    p.add_argument(
        "--source",
        default=None,
        help="price source; overrides PRICE_SOURCE. See sources.py",
    )
    p.add_argument(
        "--symbols-file",
        type=Path,
        default=Path("universe.txt"),
        help="one symbol per line; the working universe until P1-15 lands",
    )
    p.add_argument(
        "--delisted-file",
        type=Path,
        default=Path("universe_delisted.txt"),
        help="symbol,last_trading_date,reason — excluded from the coverage denominator",
    )
    a = p.parse_args()

    # No fallback universe. A default of three symbols would publish a release
    # that looks healthy, passes the non-empty check, and contains almost no
    # history — the exact silent failure this pipeline is built to refuse. If
    # the universe is missing, the night is a loud failure, not a quiet one.
    if not a.symbols_file.exists():
        raise SystemExit(
            f"universe file not found: {a.symbols_file}\n"
            "Refusing to capture an implicit universe. Run "
            "scripts/build_universe.py, or pass --symbols-file."
        )
    symbols = [s.strip() for s in a.symbols_file.read_text().split() if s.strip()]
    if not symbols:
        raise SystemExit(f"universe file is empty: {a.symbols_file}")

    m = capture(
        a.out,
        a.session or previous_session(),
        symbols,
        get_source(a.source),
        a.knowledge_date,
        read_delistings(a.delisted_file),
    )

    # A capture that returned rows but missed most of the universe is a failed
    # fetch wearing a successful one's clothes. Fail here rather than publish it.
    if m.row_count and m.universe_coverage < MIN_UNIVERSE_COVERAGE:
        raise SystemExit(
            f"universe coverage {m.universe_coverage:.1%} is below "
            f"{MIN_UNIVERSE_COVERAGE:.0%}. Refusing to publish a partial session. "
            "Check the date is a trading day and the data entitlement is live."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
