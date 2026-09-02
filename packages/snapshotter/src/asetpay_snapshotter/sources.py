"""Where prices come from — one Protocol, several vendors, chosen by config.

The snapshotter was the only place in this codebase with a vendor name baked
into it. Everywhere else the pattern is already established: `Store` and
`CostModel` are descriptions that implementations satisfy, so the thing behind
them can be replaced without touching what reads from them. This module applies
that pattern to the price feed.

The reason is not tidiness. Two decisions that looked settled are not:

  WHICH BROKER   Broker eligibility is jurisdictional. The data feed must not
                 inherit that constraint, because the feed has a clock on it —
                 a night not captured is unbuyable — and the broker decision is
                 Phase 4.

  WHICH FEED     IEX volume is not consolidated volume. Every liquidity number
                 downstream inherits whichever feed the bar came from. That is
                 a Gate 2 dependency, and Gate 2 is where a wrong answer is
                 expensive.

`PriceSource` is deliberately NOT in `contracts`. contracts is the coordination
surface between the two rails, changing it requires both people in the room,
and P2 never touches the price feed — it reads the Store. Putting this here
keeps the meeting short.

---------------------------------------------------------------------------
Free-tier request maths, which is what actually decides this
---------------------------------------------------------------------------

For a 502-name universe captured once per session:

  polygon   ONE request per night for the entire US market. Grouped daily is
            included in all Stocks plans, end-of-day on the free tier. Volume
            is CONSOLIDATED. No per-symbol quota.

  alpaca    Three requests (chunked at 200 symbols). Free tier, no per-symbol
            quota. Volume is IEX only — a few percent of consolidated.

  tiingo    ~502 requests, one per symbol, against a free tier metered in
            unique symbols per month. Not implemented: the request maths does
            not fit a daily full-universe capture, and pretending otherwise
            would mean discovering it on night three.

  moomoo /  Gateway APIs. Both need a logged-in daemon (OpenD, IB Gateway)
  ibkr      alive on an ephemeral CI runner, and moomoo meters historical
            candles by account assets — 100 per 7 days below 10,000 HKD,
            1,000 above 500,000 HKD. A 502-name weekly capture does not fit
            without a large funded account. These are execution venues, not
            research feeds. Not implemented on purpose.

polygon is therefore the default, and it is the better instrument regardless
of jurisdiction: consolidated volume, and the whole market rather than a
universe chosen in advance by a person who did not yet know what they would
need.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Protocol, runtime_checkable

import pandas as pd

# The schema every source must produce. Fixed here rather than per-vendor, so a
# provider switch is a change of origin and not a change of shape.
#
# vwap and trade_count are captured even though nothing reads them yet. They are
# free from both providers, they are exactly the inputs a per-name cost model
# wants (P1-23), and — like everything else here — they cannot be bought back
# later. Capturing an unused column costs bytes; not capturing it costs the
# feature.
COLUMNS = [
    "event_date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trade_count",
]


def empty_frame() -> pd.DataFrame:
    """What a source returns when its credentials are absent.

    Empty rather than raising, so the pipeline can be exercised end to end
    before any account exists. The workflow refuses to publish an empty
    snapshot, so this cannot quietly become a run of blank releases.
    """
    return pd.DataFrame(columns=COLUMNS)


def _epoch_ms_to_date(ms: int) -> str:
    """Bar timestamp -> ISO event date, in UTC.

    US daily bars are stamped at 00:00 ET of the session, which is the same
    calendar day in UTC. Kept as a named function so that the day a non-US
    venue arrives, this is the one place the assumption is written down.
    """
    return datetime.fromtimestamp(ms / 1000, tz=UTC).date().isoformat()


@runtime_checkable
class PriceSource(Protocol):
    """One session of daily bars, from somewhere."""

    @property
    def provider(self) -> str:
        """Recorded in every manifest. The day the feed changed must be
        recoverable from the snapshots rather than from memory."""
        ...

    def parameters(self) -> dict[str, object]:
        """Vendor-specific settings that affect what the numbers mean."""
        ...

    def fetch(self, session: date, symbols: Sequence[str]) -> pd.DataFrame:
        """Bars for one trading session, with exactly `COLUMNS`.

        `session` is the EVENT date — the day the trading happened. It is not
        the knowledge date; see capture.py for why those must not be the same
        value.

        `symbols` is the universe of interest. A source that can cheaply return
        MORE than that should do so — breadth captured tonight is breadth you
        have forever, and narrowing is always available later.
        """
        ...


# --------------------------------------------------------------------- polygon


class PolygonSource:
    """Grouped daily bars: the entire US market for one date, in one request.

    Two properties that matter more than the convenience:

    CONSOLIDATED VOLUME  Not one venue's participation. This is what makes ADV,
                         dollar volume, the liquidity floor (P1-25) and the
                         cost model (P1-23) mean what their names say.

    WHOLE MARKET         The response is not filtered to `symbols`. It is not
                         cheaper to ask for less, and the universe you will
                         want in month six is not the universe you can name in
                         week one. `universe.txt` becomes a coverage FLOOR that
                         is checked, rather than a list that silently bounds
                         what history exists.

    Adjusted=false on purpose. A bar adjusted with today's split factors is not
    what was observable then; adjustments are applied downstream, point in time.
    """

    BASE = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks"

    provider = "polygon.v2.aggs.grouped"

    def parameters(self) -> dict[str, object]:
        return {"timeframe": "1Day", "adjusted": False, "scope": "all_us_stocks"}

    def fetch(self, session: date, symbols: Sequence[str]) -> pd.DataFrame:
        key = os.environ.get("POLYGON_API_KEY")
        if not key:
            print("POLYGON_API_KEY absent — emitting an empty frame")
            return empty_frame()

        import httpx

        url = f"{self.BASE}/{session.isoformat()}"
        with httpx.Client(timeout=120.0) as client:
            r = client.get(url, params={"adjusted": "false", "apiKey": key})
            r.raise_for_status()
            body = r.json()

        # `t` is the bar's own session timestamp in epoch ms UTC. Taking the
        # date from the DATA rather than from the label means a response for the
        # wrong day is visible as a mismatch instead of being relabelled to look
        # correct.
        rows = [
            {
                "event_date": _epoch_ms_to_date(b["t"]),
                "symbol": b["T"],
                "open": b.get("o"),
                "high": b.get("h"),
                "low": b.get("l"),
                "close": b.get("c"),
                "volume": b.get("v"),
                "vwap": b.get("vw"),
                "trade_count": b.get("n"),
            }
            for b in (body.get("results") or [])
        ]
        return pd.DataFrame(rows, columns=COLUMNS)


# ---------------------------------------------------------------------- alpaca


class AlpacaSource:
    """A free paper account gives the price feed and a Phase 4 broker at once.

    That two-in-one is only a saving if the broker is available where you live.
    Where it is not, this is simply one price feed among several — and one whose
    free tier records the IEX feed. IEX is a single venue carrying a few percent
    of consolidated US volume: prices are representative, VOLUME IS NOT.

    Set ALPACA_FEED=sip once a paid data subscription exists. The feed lands in
    every manifest either way.
    """

    BASE = "https://data.alpaca.markets/v2/stocks/bars"
    CHUNK = 200  # stays inside URL length limits; the free tier is generous on rate

    provider = "alpaca.v2.stocks.bars"

    def feed(self) -> str:
        return os.environ.get("ALPACA_FEED", "iex")

    def parameters(self) -> dict[str, object]:
        return {"timeframe": "1Day", "feed": self.feed(), "scope": "universe"}

    def fetch(self, session: date, symbols: Sequence[str]) -> pd.DataFrame:
        key = os.environ.get("ALPACA_API_KEY_ID")
        secret = os.environ.get("ALPACA_API_SECRET_KEY")
        if not key or not secret:
            print("ALPACA credentials absent — emitting an empty frame (see P1-14b)")
            return empty_frame()

        import httpx

        day = session.isoformat()
        rows: list[dict[str, object]] = []
        with httpx.Client(timeout=60.0) as client:
            for i in range(0, len(symbols), self.CHUNK):
                chunk = symbols[i : i + self.CHUNK]
                r = client.get(
                    self.BASE,
                    params={
                        "symbols": ",".join(chunk),
                        "timeframe": "1Day",
                        "start": day,
                        "end": day,
                        # Adjustments are applied downstream, point-in-time, NOT
                        # baked in here — see PolygonSource for the reasoning.
                        "adjustment": "raw",
                        "feed": self.feed(),
                    },
                    headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
                )
                r.raise_for_status()
                for sym, bars in (r.json().get("bars") or {}).items():
                    rows.extend(
                        {
                            "event_date": b["t"][:10],
                            "symbol": sym,
                            "open": b.get("o"),
                            "high": b.get("h"),
                            "low": b.get("l"),
                            "close": b.get("c"),
                            "volume": b.get("v"),
                            "vwap": b.get("vw"),
                            "trade_count": b.get("n"),
                        }
                        for b in bars
                    )
        return pd.DataFrame(rows, columns=COLUMNS)


# -------------------------------------------------------------------- registry


SOURCES: dict[str, type] = {
    "polygon": PolygonSource,
    "alpaca": AlpacaSource,
}

DEFAULT_SOURCE = "polygon"


def get_source(name: str | None = None) -> PriceSource:
    """Resolve PRICE_SOURCE to an implementation, or fail with the options.

    An unknown name is a hard error rather than a fall-through to the default.
    A typo that silently captures from a different vendor than intended is a
    discontinuity in the history that nobody would think to look for.
    """
    chosen = (name or os.environ.get("PRICE_SOURCE") or DEFAULT_SOURCE).strip().lower()
    if chosen not in SOURCES:
        raise SystemExit(f"unknown PRICE_SOURCE {chosen!r}. Available: {sorted(SOURCES)}")
    return SOURCES[chosen]()
