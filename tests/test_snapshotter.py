"""P1-08 — the manifest must be deterministic, and the capture immutable.

Determinism is not a nicety here. `features_hash` downstream is built on the
same principle, and if a digest wobbles between runs over identical input, it
presents as a data bug and takes a week to find.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from asetpay_snapshotter.capture import (
    canonical_checksum,
    capture,
    previous_session,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_date": ["2026-08-17", "2026-08-17", "2026-08-17"],
            "symbol": ["MSFT", "AAPL", "SPY"],
            "close": [415.2266, 227.101, 559.9],
            "volume": [1_234_567, 8_765_432, 55_555],
        }
    )


def test_checksum_is_stable_across_runs() -> None:
    assert canonical_checksum(_frame()) == canonical_checksum(_frame())


def test_checksum_ignores_row_and_column_order() -> None:
    """Content, not incidental ordering. Otherwise a provider that returns rows
    in a different order looks like changed data."""
    base = _frame()
    shuffled = base.sample(frac=1.0, random_state=7).reset_index(drop=True)
    reordered = shuffled[["volume", "symbol", "close", "event_date"]]
    assert canonical_checksum(base) == canonical_checksum(reordered)


def test_checksum_changes_when_a_value_changes() -> None:
    """The control: a digest that never changes is not a digest."""
    base = _frame()
    edited = base.copy()
    edited.loc[0, "close"] = 415.2267
    assert canonical_checksum(base) != canonical_checksum(edited)


def test_empty_frame_has_a_defined_checksum() -> None:
    assert canonical_checksum(pd.DataFrame()) == canonical_checksum(pd.DataFrame())


def test_capture_writes_a_complete_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)

    m = capture(tmp_path, date(2026, 8, 17), ["AAPL", "MSFT"])
    written = json.loads((tmp_path / "_manifest.json").read_text())

    for field in (
        "snapshot_id",
        "knowledge_date",
        "session",
        "provider",
        "parameters",
        "row_count",
        "checksum",
        "fetched_at",
        "schema_version",
    ):
        assert field in written, f"manifest is missing {field}"

    # The session is what was asked for; the knowledge date is when we asked.
    # They are deliberately different — see the capture module docstring.
    assert written["session"] == "2026-08-17"
    assert written["knowledge_date"] >= written["session"]
    assert written["checksum"].startswith("sha256:")
    assert m.schema_version >= 1
    assert (tmp_path / "prices.parquet").exists()


def test_every_row_carries_the_knowledge_date(tmp_path, monkeypatch) -> None:
    """knowledge_date is the axis the point-in-time store partitions by. It is
    stamped at capture time and never inferred later from a filename."""
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    capture(tmp_path, date(2026, 8, 17), ["AAPL"], knowledge_date=date(2026, 8, 18))
    df = pd.read_parquet(tmp_path / "prices.parquet")
    assert "knowledge_date" in df.columns
    if len(df):
        assert (df["knowledge_date"] == "2026-08-18").all()


def test_previous_session_skips_the_weekend() -> None:
    assert previous_session(date(2026, 8, 17)) == date(2026, 8, 14)  # Mon -> Fri
    assert previous_session(date(2026, 8, 18)) == date(2026, 8, 17)  # Tue -> Mon
    assert previous_session(date(2026, 8, 16)) == date(2026, 8, 14)  # Sun -> Fri


@pytest.mark.parametrize("d", ["2026-08-17", "2026-01-02"])
def test_recapturing_the_same_day_is_byte_identical(tmp_path, monkeypatch, d) -> None:
    """Immutability in practice: capturing the same day twice must produce the
    same content digest, so a re-run can never silently alter history."""
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    a = capture(tmp_path / "a", date.fromisoformat(d), ["AAPL", "MSFT"])
    b = capture(tmp_path / "b", date.fromisoformat(d), ["AAPL", "MSFT"])
    assert a.checksum == b.checksum
