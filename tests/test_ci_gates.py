"""The gates that guard the other gates.

Every check in this file exists because the thing it checks was, at some point,
silently not running. That is the specific failure this repo claims to design
against — "a test that has never failed may be incapable of failing" — and a CI
step that exits 0 because its config never parsed is the same failure wearing a
green tick.
"""

from __future__ import annotations

import configparser
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- P1-13 config


def _importlinter_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / ".importlinter")
    return cfg


def test_importlinter_root_packages_are_real_importable_packages():
    """The original config wrote `root_packages = a, b` on one line.

    import-linter splits multi-values on NEWLINES, so it read that as a package
    literally named `a` and died with `Could not find package 'a'` — after CI
    had already reported the architectural boundaries as enforced. The names
    must resolve, or the P1-13 gate is decoration.
    """
    raw = _importlinter_config()["importlinter"]["root_packages"]
    packages = [line.strip() for line in raw.splitlines() if line.strip()]

    assert len(packages) >= 2, f"expected several root packages, parsed {packages!r}"
    for name in packages:
        assert "," not in name, (
            f"{name!r} still contains a comma — import-linter splits on newlines, "
            "not commas, and will treat this whole string as one package name"
        )
        assert importlib.util.find_spec(name) is not None, f"{name} is not importable"


def test_importlinter_declares_the_two_architectural_contracts():
    """Both named rules must still exist: the dependency-free contract, and the
    single chokepoint for raw file reads."""
    sections = _importlinter_config().sections()
    contracts = [s for s in sections if s.startswith("importlinter:contract:")]
    assert len(contracts) >= 2, f"expected >=2 contracts, found {contracts}"


# ------------------------------------------------------------- the universe


UNIVERSE = ROOT / "universe.txt"
SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,6}$")


def test_universe_file_exists():
    """capture.py has no fallback universe by design. If this file goes missing
    the nightly job fails loudly; without this test it goes missing quietly."""
    assert UNIVERSE.exists(), "universe.txt is missing — the snapshotter will refuse to run"


def test_universe_is_large_enough_to_be_a_universe():
    """A guard against the failure that motivated removing the fallback: a run
    that succeeds, publishes a release, and captures a handful of names."""
    symbols = UNIVERSE.read_text().split()
    assert len(symbols) >= 400, f"universe has only {len(symbols)} symbols"


def test_universe_symbols_are_well_formed():
    bad = [s for s in UNIVERSE.read_text().split() if not SYMBOL.match(s)]
    assert not bad, f"malformed symbols: {bad}"


def test_universe_has_no_duplicates_and_is_sorted():
    """Sorted and deduplicated so the git diff of this file is a clean record of
    when each name entered the universe. build_universe.py writes it this way."""
    symbols = UNIVERSE.read_text().split()
    assert len(symbols) == len(set(symbols)), "duplicate symbols"
    assert symbols == sorted(symbols), "universe.txt is not sorted"


# ------------------------------------------------- no implicit universe, ever


def test_capture_refuses_to_run_without_a_universe(tmp_path, monkeypatch):
    """The negative control for the removed fallback.

    It previously defaulted to ["AAPL", "MSFT", "SPY"], which produced a release
    that passed the workflow's non-empty check and contained three names. A
    missing universe must be a failed job, not a thin one.
    """
    from asetpay_snapshotter import capture as cap

    monkeypatch.setattr(
        "sys.argv",
        [
            "capture",
            "--out",
            str(tmp_path / "out"),
            "--symbols-file",
            str(tmp_path / "does-not-exist.txt"),
        ],
    )
    with pytest.raises(SystemExit) as e:
        cap.main()
    assert "universe file not found" in str(e.value)


def test_capture_refuses_an_empty_universe(tmp_path, monkeypatch):
    empty = tmp_path / "universe.txt"
    empty.write_text("\n  \n")

    from asetpay_snapshotter import capture as cap

    monkeypatch.setattr(
        "sys.argv",
        ["capture", "--out", str(tmp_path / "out"), "--symbols-file", str(empty)],
    )
    with pytest.raises(SystemExit) as e:
        cap.main()
    assert "empty" in str(e.value)


# ------------------------------------------------------ the feed is recorded


def test_manifest_records_the_feed_actually_used(tmp_path, monkeypatch):
    """IEX volume is not consolidated volume. Every liquidity number downstream
    depends on which feed a row came from, so the manifest records it rather
    than hardcoding a literal that drifts from reality on the day it changes.
    """
    import json
    from datetime import date

    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    monkeypatch.setenv("PRICE_SOURCE", "alpaca")
    monkeypatch.setenv("ALPACA_FEED", "sip")

    from asetpay_snapshotter.capture import capture

    capture(tmp_path, date(2026, 8, 17), ["AAPL"])
    written = json.loads((tmp_path / "_manifest.json").read_text())
    assert written["provider"].startswith("alpaca")
    assert written["parameters"]["feed"] == "sip"


# ------------------------------------------- delistings and the coverage floor


DELISTED = ROOT / "universe_delisted.txt"


def test_delisted_names_stay_in_the_universe():
    """Additive only. A delisted name dropped from universe.txt is the first
    move of survivorship bias — the pathology truth.json plants a trap for."""
    from asetpay_snapshotter.capture import read_delistings

    universe = set(UNIVERSE.read_text().split())
    for symbol in read_delistings(DELISTED):
        assert symbol in universe, f"{symbol} was delisted AND removed from the universe"


def test_every_delisting_names_its_successor_or_reason():
    """A delisting with no recorded reason is indistinguishable from a typo
    somebody quietly deleted."""
    for line in DELISTED.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",", 2)
        assert len(parts) == 3, f"no reason recorded: {line!r}"
        assert parts[2].strip(), f"empty reason: {line!r}"


def test_coverage_is_measured_against_names_expected_to_trade():
    """The AvalonBay/Equity Residential merger cost 0.4% of coverage on the
    first live capture. Measured against the full universe, coverage decays
    with every corporate action until the floor trips for reasons that are not
    failures — and the 2am fix is to lower the floor, which disables the check.
    """
    from datetime import date

    from asetpay_snapshotter.capture import expected_trading

    universe = ["AAPL", "AVB", "EQR", "VMRK"]
    delisted = {"AVB": date(2026, 8, 17), "EQR": date(2026, 8, 17)}

    # On its last trading day the name is still expected.
    assert expected_trading(universe, delisted, date(2026, 8, 17)) == universe
    # The day after, it is not.
    assert expected_trading(universe, delisted, date(2026, 8, 18)) == ["AAPL", "VMRK"]
    # Long before, it is.
    assert expected_trading(universe, delisted, date(2020, 1, 2)) == universe


def test_the_delisted_file_parses():
    """It is read by the nightly job. A malformed line must fail here, not at
    03:00 UTC."""
    from asetpay_snapshotter.capture import read_delistings

    assert read_delistings(DELISTED), "expected at least one recorded delisting"


def test_a_malformed_delisting_line_is_refused(tmp_path):
    from asetpay_snapshotter.capture import read_delistings

    bad = tmp_path / "d.txt"
    bad.write_text("AVB\n")
    with pytest.raises(SystemExit):
        read_delistings(bad)


def test_a_missing_delisted_file_is_not_an_error(tmp_path):
    """Nothing has delisted yet is a legitimate state for a new universe."""
    from asetpay_snapshotter.capture import read_delistings

    assert read_delistings(tmp_path / "nope.txt") == {}
