"""P2-02 — FixtureStore: a real Store implementation over synthetic data.

This is what makes P2 unblockable. P2 builds the whole measurement stack against
this in week 1; when P1's real point-in-time store lands in week 3, P2's code does
not change, because both satisfy the same Protocol.

It is NOT a stub. It implements the bitemporal filter properly — the same
knowledge-time logic P1 will implement over Parquet — so that P2's code is
exercised against correct semantics from day one. A fixture that cheats on
knowledge time would let lookahead bugs hide until the real store arrived.

The as-of rule, which is the whole idea:
    1. discard everything with knowledge_date > the date being asked about
    2. among what remains, for each (asset, period), keep the row with the
       LATEST knowledge_date
Restatements fall out for free: the original and the revision are two rows, and
step 2 picks whichever was current at the time.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from asetpay_contracts import AssetId


class FixtureStoreView:
    """A frozen view of the fixture at one knowledge time."""

    def __init__(self, root: FixtureStore, knowledge_time: date) -> None:
        self._root = root
        self._kt = knowledge_time

    @property
    def knowledge_time(self) -> date:
        return self._kt

    def prices(self, assets: Sequence[AssetId], start: date, end: date) -> pd.DataFrame:
        df = self._root._prices
        kt, s, e = self._kt.isoformat(), start.isoformat(), end.isoformat()
        m = (df["knowledge_date"] <= kt) & (df["event_date"] >= s) & (df["event_date"] <= e)
        if assets:
            m &= df["asset_id"].isin(list(assets))
        return df.loc[m].reset_index(drop=True)

    def fundamentals(self, assets: Sequence[AssetId]) -> pd.DataFrame:
        """As REPORTED at knowledge_time — not as later restated."""
        df = self._root._fundamentals
        m = df["knowledge_date"] <= self._kt.isoformat()
        if assets:
            m &= df["asset_id"].isin(list(assets))
        sub = df.loc[m]
        if sub.empty:
            return sub.reset_index(drop=True)
        # Among what was knowable, keep the most recent statement per period.
        sub = sub.sort_values("knowledge_date")
        return (
            sub.groupby(["asset_id", "period_end", "metric"], as_index=False)
            .tail(1)
            .sort_values(["asset_id", "period_end"])
            .reset_index(drop=True)
        )

    def universe(self, universe_id: str = "all") -> list[AssetId]:
        """Names that were live at knowledge_time — INCLUDING ones that have
        since delisted. Excluding them here is exactly survivorship bias."""
        df = self._root._prices
        kt = self._kt.isoformat()
        live = df.loc[df["event_date"] <= kt, ["asset_id", "event_date"]]
        if live.empty:
            return []
        last_seen = live.groupby("asset_id")["event_date"].max()
        # A name is in the universe if it was still trading recently as of kt.
        cutoff = sorted(live["event_date"].unique())[-5:][0]
        return [AssetId(a) for a in last_seen[last_seen >= cutoff].index.tolist()]

    def resolve_ticker(self, ticker: str) -> AssetId | None:
        """Which company did this ticker mean at knowledge_time?

        Present because the ticker->asset mapping is itself bitemporal. A backtest
        that resolves tickers with today's mapping merges two unrelated companies.
        """
        df = self._root._identifiers
        kt = self._kt.isoformat()
        m = (
            (df["id_type"] == "ticker")
            & (df["id_value"] == ticker)
            & (df["valid_from"] <= kt)
            & (df["valid_to"].isna() | (df["valid_to"] > kt))
        )
        hit = df.loc[m]
        return AssetId(hit.iloc[0]["asset_id"]) if len(hit) else None


class FixtureStore:
    """Store over the synthetic fixture. Satisfies the Store Protocol."""

    def __init__(self, data_dir: Path | str) -> None:
        d = Path(data_dir)
        self._prices = pd.read_parquet(d / "prices.parquet")
        self._fundamentals = pd.read_parquet(d / "fundamentals.parquet")
        self._identifiers = pd.read_parquet(d / "identifiers.parquet")
        self._signals = pd.read_parquet(d / "signals.parquet")
        self._delistings = pd.read_parquet(d / "delistings.parquet")
        self.truth = json.loads((d / "truth.json").read_text())

    def as_of(self, knowledge_time: date) -> FixtureStoreView:
        return FixtureStoreView(self, knowledge_time)

    # -- fixture-only conveniences (a real Store has no equivalent) ----------

    def planted_signals(self) -> pd.DataFrame:
        """The signals whose IC we planted. Only a fixture can offer this."""
        return self._signals

    def all_prices(self) -> pd.DataFrame:
        return self._prices

    def delisting_returns(self) -> pd.DataFrame:
        return self._delistings
