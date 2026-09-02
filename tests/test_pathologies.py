"""The four planted pathologies, and proof the machinery catches each.

Each of these is a real bias that has destroyed real backtests. Because the
generator planted them and recorded where, you can assert that your store
notices — rather than hoping it would.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from asetpay_core.store import FixtureStore


def _d(s: str) -> date:
    return date.fromisoformat(s)


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> FixtureStore:
    from asetpay_core.synthetic import GeneratorConfig, write

    out = tmp_path_factory.mktemp("fixture")
    write(out, GeneratorConfig(n_assets=200, n_years=4))
    return FixtureStore(out)


# --------------------------------------------------------------------------
# 1. Restatement — the case that kills naive stores
# --------------------------------------------------------------------------


def test_restatement_returns_the_figure_that_was_current_at_the_time(store) -> None:
    """A company reports Q1, then revises it two quarters later.

    Ask about a date BEFORE the revision and you must get the ORIGINAL number.
    A store that overwrites instead of versioning returns the revision, and every
    backtest date before it silently trades on figures nobody had.
    """
    r = store.truth["restatements"][0]
    asset = r["asset_id"]

    before = store.as_of(_d(r["restated_known"]) - timedelta(days=1)).fundamentals([asset])
    row = before[before["period_end"] == r["period_end"]]
    assert len(row) == 1
    assert row.iloc[0]["value"] == pytest.approx(r["original_value"], rel=1e-9), (
        "the store leaked a future restatement backwards"
    )

    after = store.as_of(_d(r["restated_known"]) + timedelta(days=1)).fundamentals([asset])
    row_after = after[after["period_end"] == r["period_end"]]
    assert row_after.iloc[0]["value"] == pytest.approx(r["restated_value"], rel=1e-9)


def test_a_figure_is_invisible_before_it_was_published(store) -> None:
    """Fiscal periods end before their numbers exist. Asking between period end
    and publication must return nothing for that period."""
    r = store.truth["restatements"][0]
    view = store.as_of(_d(r["period_end"]) + timedelta(days=1))
    f = view.fundamentals([r["asset_id"]])
    assert f[f["period_end"] == r["period_end"]].empty


# --------------------------------------------------------------------------
# 2. Delisting — survivorship bias, the subtlest one
# --------------------------------------------------------------------------


def test_a_delisted_name_is_present_in_the_universe_before_it_died(store) -> None:
    """The bias is not just dropping delisted names from today's universe. It is
    dropping them from PAST universes, where they were tradeable."""
    d = store.truth["delistings"][0]
    before = store.as_of(_d(d["delisting_date"]) - timedelta(days=30)).universe()
    assert d["asset_id"] in before, "a live name is missing from its own era"


def test_every_delisting_carries_its_final_return(store) -> None:
    """A dataset containing delisted names WITHOUT their delisting returns still
    has survivorship bias, because the loss never registers. This is why the
    column is NOT NULL in the schema."""
    dr = store.delisting_returns()
    assert len(dr) == len(store.truth["delistings"])
    assert dr["final_return"].notna().all()
    assert (dr["final_return"] < -0.5).all()


def test_the_delisting_loss_is_visible_in_the_price_series(store) -> None:
    """The final return must actually be in the returns, not merely recorded in a
    side table nobody joins to."""
    d = store.truth["delistings"][0]
    px = store.all_prices()
    row = px[(px["asset_id"] == d["asset_id"]) & (px["event_date"] == d["delisting_date"])]
    assert len(row) == 1
    assert row.iloc[0]["ret_1d"] == pytest.approx(d["final_return"], rel=1e-6)
    after = px[(px["asset_id"] == d["asset_id"]) & (px["event_date"] > d["delisting_date"])]
    assert after.empty, "a delisted name kept trading"


# --------------------------------------------------------------------------
# 3. Ticker reuse — two companies silently merged into one
# --------------------------------------------------------------------------


def test_a_reused_ticker_resolves_to_different_companies_at_different_times(store) -> None:
    """The mapping from ticker to company is itself bitemporal. Resolve with
    today's mapping and your backtest merges two unrelated firms into one
    fictional company with a continuous price history."""
    u = store.truth["ticker_reuses"][0]
    handover = _d(u["handover_date"])

    assert (
        store.as_of(handover - timedelta(days=30)).resolve_ticker(u["ticker"])
        == u["first_asset_id"]
    )
    assert (
        store.as_of(handover + timedelta(days=30)).resolve_ticker(u["ticker"])
        == u["second_asset_id"]
    )


def test_the_two_companies_behind_one_ticker_are_distinct_assets(store) -> None:
    u = store.truth["ticker_reuses"][0]
    assert u["first_asset_id"] != u["second_asset_id"]


# --------------------------------------------------------------------------
# 4. Stale fundamentals — 252 rows, about 4 observations
# --------------------------------------------------------------------------


def test_fundamentals_arrive_quarterly_not_daily(store) -> None:
    """Carrying quarterly figures forward daily gives 252 rows a year containing
    roughly 4 independent observations. Reporting significance on the 252 is how
    fundamentals research manufactures results out of nothing (P2-07)."""
    fund = store.as_of(date(2022, 12, 31)).fundamentals([])
    per_asset_periods = fund.groupby("asset_id")["period_end"].nunique()
    assert per_asset_periods.max() <= 4 * 4 + 1  # 4 years of quarters
    assert store.truth["fundamentals_per_year"] == 4
    assert store.truth["trading_days_per_year"] == 252


# --------------------------------------------------------------------------
# The as-of view must be monotone: knowing more later, never less
# --------------------------------------------------------------------------


def test_later_views_never_know_less(store) -> None:
    early = store.as_of(date(2020, 6, 30)).fundamentals([])
    late = store.as_of(date(2022, 6, 30)).fundamentals([])
    assert len(late) >= len(early)
    early_keys = set(zip(early["asset_id"], early["period_end"], strict=True))
    late_keys = set(zip(late["asset_id"], late["period_end"], strict=True))
    assert early_keys <= late_keys, "a later view forgot something an earlier one knew"
