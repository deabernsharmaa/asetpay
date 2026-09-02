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
    monkeypatch.setenv("ALPACA_FEED", "sip")

    from asetpay_snapshotter.capture import capture

    capture(tmp_path, date(2026, 8, 17), ["AAPL"])
    written = json.loads((tmp_path / "_manifest.json").read_text())
    assert written["parameters"]["feed"] == "sip"
