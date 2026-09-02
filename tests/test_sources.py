"""The price feed is a choice, and the choice must be visible and reversible.

Alpaca was doing two unrelated jobs: the nightly research feed, and the Phase 4
execution broker. Broker eligibility is jurisdictional; the feed has a clock on
it. Coupling them meant a residency question could stop history accruing, which
is the one loss this project cannot recover from.

These tests hold the seam open.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from asetpay_snapshotter import sources
from asetpay_snapshotter.capture import MIN_UNIVERSE_COVERAGE, capture, universe_coverage

# ------------------------------------------------------------------ registry


def test_every_registered_source_satisfies_the_protocol():
    for name, cls in sources.SOURCES.items():
        assert isinstance(cls(), sources.PriceSource), f"{name} does not satisfy PriceSource"


def test_every_source_emits_the_same_columns(monkeypatch):
    """A provider switch must be a change of origin, not a change of shape.
    Otherwise the store's schema depends on which vendor was cheapest that
    quarter, and the point-in-time layout stops being comparable across time."""
    for var in ("POLYGON_API_KEY", "ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    for name, cls in sources.SOURCES.items():
        df = cls().fetch(date(2026, 8, 17), ["AAPL"])
        assert list(df.columns) == sources.COLUMNS, f"{name} returned {list(df.columns)}"


def test_unknown_source_is_a_hard_error_not_a_silent_default():
    """A typo that quietly captures from a different vendor is a discontinuity
    in the history that nobody would think to look for."""
    with pytest.raises(SystemExit) as e:
        sources.get_source("moomoo")
    assert "unknown PRICE_SOURCE" in str(e.value)


def test_price_source_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("PRICE_SOURCE", "alpaca")
    assert isinstance(sources.get_source(), sources.AlpacaSource)
    monkeypatch.setenv("PRICE_SOURCE", "polygon")
    assert isinstance(sources.get_source(), sources.PolygonSource)


def test_the_default_source_is_not_a_brokerage(monkeypatch):
    """The research feed must not inherit a broker's jurisdictional constraints.
    If this default ever becomes a broker again, the coupling is back."""
    monkeypatch.delenv("PRICE_SOURCE", raising=False)
    assert sources.DEFAULT_SOURCE == "polygon"
    assert isinstance(sources.get_source(), sources.PolygonSource)


def test_sources_are_distinguishable_in_the_manifest():
    """Volume from one venue and consolidated volume are different numbers. A
    history assembled from both is fine; a history where you cannot tell which
    is which is not."""
    provs = {cls().provider for cls in sources.SOURCES.values()}
    assert len(provs) == len(sources.SOURCES)


def test_missing_credentials_yield_an_empty_frame_not_an_exception(monkeypatch):
    """So the pipeline can be exercised end to end before any account exists.
    The workflow's non-empty check is what stops this becoming blank releases."""
    for var in ("POLYGON_API_KEY", "ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    for cls in sources.SOURCES.values():
        assert cls().fetch(date(2026, 8, 17), ["AAPL"]).empty


# ------------------------------------------------------------------ coverage


def _frame(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_date": "2026-08-17",
                "symbol": s,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 100,
                "vwap": 1.0,
                "trade_count": 5,
            }
            for s in symbols
        ],
        columns=sources.COLUMNS,
    )


def test_coverage_counts_the_universe_not_the_rows():
    """The failure this exists for: a response with ten thousand rows and none
    of the names you asked for. Row count says healthy; coverage says nothing
    arrived. A wrong date and an expired entitlement both look like this."""
    wanted = ["AAPL", "MSFT", "NVDA", "AMZN"]
    noise = _frame([f"XX{i}" for i in range(10_000)] + ["AAPL"])
    assert universe_coverage(noise, wanted) == 0.25
    assert len(noise) > 10_000


def test_full_coverage_is_one_and_empty_is_zero():
    wanted = ["AAPL", "MSFT"]
    assert universe_coverage(_frame(wanted), wanted) == 1.0
    assert universe_coverage(_frame([]), wanted) == 0.0


def test_a_broad_source_is_not_penalised_for_returning_extra_names():
    """Polygon returns the whole US market by design — breadth captured tonight
    is breadth you have forever. Coverage measures the floor, not the ceiling."""
    wanted = ["AAPL", "MSFT"]
    assert universe_coverage(_frame([*wanted, "TSLA", "F", "GM"]), wanted) == 1.0


def test_capture_records_coverage_in_the_manifest(monkeypatch):
    class Stub:
        provider = "stub"

        def parameters(self):
            return {}

        def fetch(self, knowledge_date, symbols):
            return _frame(["AAPL"])

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        m = capture(Path(d), date(2026, 8, 17), ["AAPL", "MSFT"], Stub())
    assert m.universe_coverage == 0.5
    assert m.provider == "stub"


def test_the_coverage_floor_tolerates_halts_but_not_a_truncated_session():
    """Names halt, get acquired and stop trading, so the floor is not 100%. A
    truncated response looks nothing like a few halts."""
    assert 0.5 < MIN_UNIVERSE_COVERAGE < 1.0


# -------------------------------------------------- two dates, never one


def test_knowledge_date_defaults_to_today_not_the_session():
    """Nobody had Tuesday's close during Tuesday. Stamping those rows
    knowledge_date=Tuesday makes as_of(Tuesday) hand a strategy a number that
    did not exist yet — a few hours of lookahead, applied uniformly, in the
    direction that flatters every backtest."""
    import tempfile
    from datetime import UTC, datetime
    from pathlib import Path

    class Stub:
        provider = "stub"

        def parameters(self):
            return {}

        def fetch(self, session, symbols):
            return _frame(["AAPL"])

    session = date(2026, 8, 17)
    with tempfile.TemporaryDirectory() as d:
        m = capture(Path(d), session, ["AAPL"], Stub())

    assert m.session == "2026-08-17"
    assert m.knowledge_date == datetime.now(tz=UTC).date().isoformat()
    assert m.knowledge_date != m.session


def test_a_knowledge_date_before_its_session_is_refused():
    """The negative control. Claiming to have known Tuesday's close on Monday
    is lookahead by construction, so it is rejected rather than written."""
    import tempfile
    from pathlib import Path

    class Stub:
        provider = "stub"

        def parameters(self):
            return {}

        def fetch(self, session, symbols):
            return _frame(["AAPL"])

    with tempfile.TemporaryDirectory() as d, pytest.raises(SystemExit) as e:
        capture(
            Path(d),
            date(2026, 8, 18),
            ["AAPL"],
            Stub(),
            knowledge_date=date(2026, 8, 17),
        )
    assert "lookahead" in str(e.value)


def test_a_backfill_can_state_when_the_data_became_knowable():
    """A backfill run today must not pretend it happened years ago, but it also
    must not claim today's knowledge for a decade-old session. The date is an
    argument precisely so the honest answer can be written down."""
    import tempfile
    from pathlib import Path

    class Stub:
        provider = "stub"

        def parameters(self):
            return {}

        def fetch(self, session, symbols):
            return _frame(["AAPL"])

    with tempfile.TemporaryDirectory() as d:
        m = capture(
            Path(d),
            date(2020, 3, 16),
            ["AAPL"],
            Stub(),
            knowledge_date=date(2020, 3, 17),
        )
    assert m.session == "2020-03-16"
    assert m.knowledge_date == "2020-03-17"


def test_event_date_comes_from_the_bar_not_the_label():
    """Taking the date from the DATA means a response for the wrong day shows up
    as a mismatch instead of being relabelled to look correct."""
    ms = 1755388800000  # 2025-08-17T00:00:00Z
    assert sources._epoch_ms_to_date(ms) == "2025-08-17"


# ------------------------------------- a number says something is wrong;
# ------------------------------------- a list says what


def test_the_manifest_names_the_missing_symbols():
    """`coverage 0.996` sent a person off to work out which two names it was.
    Recording them means the next person reads the answer instead."""
    import tempfile
    from pathlib import Path

    class Stub:
        provider = "stub"

        def parameters(self):
            return {}

        def fetch(self, session, symbols):
            return _frame(["AAPL", "MSFT"])

    with tempfile.TemporaryDirectory() as d:
        m = capture(Path(d), date(2026, 8, 17), ["AAPL", "MSFT", "BRK.B", "BF.B"], Stub())

    assert m.missing_symbols == ["BF.B", "BRK.B"]
    assert m.universe_coverage == 0.5


def test_a_complete_response_records_no_missing_symbols():
    """The control: a field that is never empty is not carrying information."""
    from asetpay_snapshotter.capture import missing_from

    assert missing_from(_frame(["AAPL", "MSFT"]), ["AAPL", "MSFT"]) == []


def test_the_missing_list_is_capped():
    """A genuinely broken night would otherwise write thousands of symbols into
    every manifest. Twenty is enough to recognise the pattern."""
    import tempfile
    from pathlib import Path

    from asetpay_snapshotter.capture import MAX_MISSING_RECORDED

    class Stub:
        provider = "stub"

        def parameters(self):
            return {}

        def fetch(self, session, symbols):
            return _frame([])

    wanted = [f"SYM{i:04d}" for i in range(500)]
    with tempfile.TemporaryDirectory() as d:
        m = capture(Path(d), date(2026, 8, 17), wanted, Stub())

    assert len(m.missing_symbols) == MAX_MISSING_RECORDED
    assert m.universe_coverage == 0.0
