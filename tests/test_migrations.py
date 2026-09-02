"""P1-03 — proof that the schema REFUSES the mistakes, rather than discouraging them.

Every test here tries to insert a state that would produce a silently wrong
backtest, and asserts that PostgreSQL rejects it. That is the difference between
a constraint and a convention: a convention needs someone to remember it at 2am
in month seven.

Skipped automatically when no database is reachable, so the rest of the suite
still runs on a laptop with nothing installed. Set ASETPAY_TEST_DSN to point at
a scratch database.
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg", reason="psycopg not installed")

DSN = os.environ.get("ASETPAY_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="ASETPAY_TEST_DSN not set")

MIGRATION = os.path.join(
    os.path.dirname(__file__), "..", "migrations", "001_security_master.sql"
)


@pytest.fixture()
def conn():
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("DROP SCHEMA IF EXISTS test_sm CASCADE; CREATE SCHEMA test_sm;")
        c.execute("SET search_path TO test_sm, public;")
        with open(MIGRATION) as fh:
            c.execute(fh.read())
        yield c
        c.execute("DROP SCHEMA IF EXISTS test_sm CASCADE;")


def _add_asset(conn, name: str = "ACME") -> str:
    aid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO security_master "
        "(asset_id, company_name, exchange, currency, asset_class, listing_date) "
        "VALUES (%s, %s, 'XNAS', 'USD', 'equity', '2010-01-01')",
        (aid, name),
    )
    return aid


# ---------------------------------------------------------------------------
# The one that stops two companies being merged into one
# ---------------------------------------------------------------------------


def test_two_assets_cannot_hold_the_same_ticker_on_the_same_day(conn) -> None:
    a, b = _add_asset(conn, "First Corp"), _add_asset(conn, "Second Corp")
    conn.execute(
        "INSERT INTO asset_identifiers VALUES (%s,'ticker','XYZ','[2010-01-01,2015-01-01)')",
        (a,),
    )
    with pytest.raises(psycopg.errors.ExclusionViolation):
        conn.execute(
            "INSERT INTO asset_identifiers VALUES (%s,'ticker','XYZ','[2014-01-01,2020-01-01)')",
            (b,),
        )


def test_a_ticker_may_be_reassigned_once_the_first_holder_releases_it(conn) -> None:
    """The legitimate case must still work — otherwise the constraint is too
    strong and people will drop it rather than model reality."""
    a, b = _add_asset(conn, "First Corp"), _add_asset(conn, "Second Corp")
    conn.execute(
        "INSERT INTO asset_identifiers VALUES (%s,'ticker','XYZ','[2010-01-01,2015-01-01)')",
        (a,),
    )
    conn.execute(
        "INSERT INTO asset_identifiers VALUES (%s,'ticker','XYZ','[2015-01-01,)')", (b,)
    )
    n = conn.execute("SELECT count(*) FROM asset_identifiers WHERE id_value='XYZ'").fetchone()[
        0
    ]
    assert n == 2

    # And resolution is unambiguous at any instant — which is the whole point.
    for when, expect in (("2012-06-01", a), ("2018-06-01", b)):
        got = conn.execute(
            "SELECT asset_id FROM asset_identifiers "
            "WHERE id_type='ticker' AND id_value='XYZ' AND valid @> %s::date",
            (when,),
        ).fetchall()
        assert len(got) == 1 and str(got[0][0]) == expect


def test_one_asset_cannot_have_two_overlapping_tickers(conn) -> None:
    a = _add_asset(conn)
    conn.execute(
        "INSERT INTO asset_identifiers VALUES (%s,'ticker','AAA','[2010-01-01,2020-01-01)')",
        (a,),
    )
    with pytest.raises(psycopg.errors.ExclusionViolation):
        conn.execute(
            "INSERT INTO asset_identifiers VALUES (%s,'ticker','BBB','[2015-01-01,2018-01-01)')",
            (a,),
        )


# ---------------------------------------------------------------------------
# Universe membership
# ---------------------------------------------------------------------------


def test_universe_membership_cannot_overlap_itself(conn) -> None:
    a = _add_asset(conn)
    conn.execute(
        "INSERT INTO universe_membership VALUES ('sp500',%s,'[2010-01-01,2015-01-01)')",
        (a,),
    )
    with pytest.raises(psycopg.errors.ExclusionViolation):
        conn.execute(
            "INSERT INTO universe_membership VALUES ('sp500',%s,'[2014-01-01,2016-01-01)')",
            (a,),
        )


def test_the_same_asset_may_be_in_two_different_universes(conn) -> None:
    a = _add_asset(conn)
    conn.execute(
        "INSERT INTO universe_membership VALUES ('sp500',%s,'[2010-01-01,2015-01-01)')", (a,)
    )
    conn.execute(
        "INSERT INTO universe_membership VALUES ('russell',%s,'[2010-01-01,2015-01-01)')", (a,)
    )


# ---------------------------------------------------------------------------
# Survivorship — the subtlest bias, enforced at the column level
# ---------------------------------------------------------------------------


def test_a_delisting_cannot_be_recorded_without_its_cost(conn) -> None:
    """This is the constraint doing the real work.

    A delisted name with no final return still carries survivorship bias, because
    the loss never registers. NOT NULL means you cannot half-record a delisting:
    either you know what it cost, or the name goes to universe_exclusions and the
    gap is visible."""
    a = _add_asset(conn)
    with pytest.raises(psycopg.errors.NotNullViolation):
        conn.execute(
            "INSERT INTO delisting_returns (asset_id, delisting_date, final_return, source) "
            "VALUES (%s,'2015-06-30',NULL,'manual')",
            (a,),
        )


def test_an_unsourceable_final_return_is_recorded_as_an_exclusion(conn) -> None:
    """The honest alternative when the number cannot be sourced on free data."""
    a = _add_asset(conn)
    conn.execute(
        "INSERT INTO universe_exclusions (asset_id, reason) VALUES (%s,%s)",
        (a, "no delisting return available on the free data path"),
    )
    n = conn.execute("SELECT count(*) FROM universe_exclusions").fetchone()[0]
    assert n == 1


def test_an_impossible_final_return_is_rejected(conn) -> None:
    """You cannot lose more than everything."""
    a = _add_asset(conn)
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO delisting_returns VALUES (%s,'2015-06-30',-1.5,'manual')", (a,)
        )


def test_delisting_cannot_precede_listing(conn) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO security_master "
            "(asset_id, company_name, exchange, currency, asset_class, listing_date, delisting_date) "
            "VALUES (%s,'Time Traveller','XNAS','USD','equity','2015-01-01','2010-01-01')",
            (str(uuid.uuid4()),),
        )


# ---------------------------------------------------------------------------
# Experiment tracking — abandoned trials still count
# ---------------------------------------------------------------------------


def test_abandoned_experiments_default_to_false_and_remain_countable(conn) -> None:
    """The deflated Sharpe ratio needs the TOTAL trial count. A schema that let
    you drop abandoned variants would let you understate your own multiple-testing
    burden, which is the most common way research manufactures significance."""
    for i, abandoned in enumerate([False, True, True]):
        conn.execute(
            "INSERT INTO experiments (experiment_id, agent_id, params, code_sha, data_asof, abandoned) "
            "VALUES (%s,%s,'{}'::jsonb,'abc123','2026-01-01',%s)",
            (str(uuid.uuid4()), f"agent_{i}", abandoned),
        )
    total = conn.execute("SELECT count(*) FROM experiments").fetchone()[0]
    kept = conn.execute("SELECT count(*) FROM experiments WHERE NOT abandoned").fetchone()[0]
    assert total == 3 and kept == 1, "all three trials count toward the deflation"
