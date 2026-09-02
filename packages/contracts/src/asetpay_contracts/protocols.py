"""The four Protocols — the entire connection between the Truth and Judgment rails.

P2 (Judgment) writes code against THESE DESCRIPTIONS. P1 (Truth) builds the real
implementations behind them. Because P2 codes against the description, P2 is never
blocked waiting for P1: in week 1 P2 builds a FixtureStore over synthetic data and
starts immediately, and when P1's real store lands in week 3, no P2 code changes.

Changing anything in this file requires both people in the room. It is the only
mandatory meeting in the project.

pandas is imported under TYPE_CHECKING only, so this module still imports with
nothing but the standard library at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from asetpay_contracts.ids import AssetId

if TYPE_CHECKING:  # pragma: no cover - type-checking only, never imported at runtime
    import pandas as pd


@runtime_checkable
class StoreView(Protocol):
    """A frozen view of everything that was knowable at one instant.

    Obtained only from Store.as_of(). There is deliberately no way to construct
    one directly, and no method here takes a knowledge-time argument — the
    knowledge time is baked into the view.
    """

    @property
    def knowledge_time(self) -> date:
        """The instant this view is frozen at."""
        ...

    def prices(self, assets: Sequence[AssetId], start: date, end: date) -> pd.DataFrame:
        """Daily OHLCV as it was known at knowledge_time.

        Returns a frame indexed by (event_date, asset) with at least
        columns: open, high, low, close, volume.
        """
        ...

    def fundamentals(self, assets: Sequence[AssetId]) -> pd.DataFrame:
        """Fundamentals AS REPORTED at knowledge_time, not as later restated.

        This is the method that kills naive stores. A company restates Q1 in Q3;
        a single-axis store overwrites the original, and every backtest date
        before Q3 now trades on numbers nobody had. Here, a restatement is a
        second row with a later knowledge_date, and this method returns whichever
        version was current at knowledge_time.
        """
        ...

    def universe(self, universe_id: str) -> list[AssetId]:
        """Members of the universe at knowledge_time, INCLUDING names that were
        live then and have since delisted.

        A universe that silently drops delisted names is survivorship bias. A
        universe that includes them without their delisting returns is ALSO
        survivorship bias, because the losses never register.
        """
        ...


@runtime_checkable
class Store(Protocol):
    """The single chokepoint for all historical data access.

    Nothing downstream reads raw files. One chokepoint, one place lookahead can
    enter, one place to audit. Enforced by an import-linter rule in CI
    (see .importlinter) rather than by code-review convention.
    """

    def as_of(self, knowledge_time: date) -> StoreView:
        """Everything that was genuinely knowable on this date, and nothing else.

        Note there is no method to ask for 'the current value' of anything. The
        honest question is the only question this API allows. That is the design
        preventing the mistake, rather than you remembering not to make it.
        """
        ...


@runtime_checkable
class FeatureSet(Protocol):
    """Precomputed building blocks: momentum, realized volatility, ADV, etc.

    Features are PURE functions of point-in-time data. No I/O, no state, no
    reference to the current date. Purity is what makes lookahead detectable by
    reading the code rather than by testing for it, and it is enforced by an AST
    check in CI that rejects datetime.now / date.today / store imports inside
    the features package.
    """

    def get(self, name: str, params: Mapping[str, Any], as_of: date) -> pd.DataFrame:
        """Feature values indexed by AssetId, computed as of `as_of`.

        Cacheable on (name, params, as_of) precisely because the function is pure.
        """
        ...

    def available(self) -> Sequence[str]:
        """Registered feature names."""
        ...


@runtime_checkable
class CostModel(Protocol):
    """What a trade will actually cost. P2 must subtract this before claiming edge.

    Retail US equity round trips run roughly 2-6 bps in large caps and 25-70 bps
    in small caps, against a daily predictable component of 5-10 bps. One round
    trip in large caps consumes about half the daily edge; in small caps it
    consumes several days of it.

    This is why the model must be PER-NAME and liquidity-dependent. A flat
    basis-point assumption understates small-cap costs specifically — and small
    caps are exactly where a naive backtest finds its most exciting fake results.
    """

    def round_trip_bps(self, asset: AssetId, notional: float, as_of: date) -> float:
        """Expected round-trip cost in basis points: half-spread x 2, plus
        slippage and market impact at this notional, plus commission."""
        ...

    def is_tradeable(self, asset: AssetId, notional: float, as_of: date) -> bool:
        """False if this name falls below the liquidity floor at this size.

        Names below the floor are excluded from the portfolio regardless of score.
        """
        ...
