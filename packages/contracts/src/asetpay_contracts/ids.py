"""Permanent security identifiers.

A ticker is NOT an identifier. Tickers get reused and reassigned, and a backtest
keyed on tickers silently merges two unrelated companies without anybody noticing.
AssetId is issued once per security and never reused.

The ticker -> AssetId mapping is itself bitemporal and lives in the
`asset_identifiers` table (see migrations/001_security_master.sql), because the
mapping changes over time and a backtest that uses today's mapping for a 2019
decision is using information nobody had.
"""

from __future__ import annotations

import uuid
from typing import NewType

AssetId = NewType("AssetId", str)
"""A permanent, opaque security identifier. Never a ticker, never reused."""


def new_asset_id() -> AssetId:
    """Issue a fresh AssetId. Call once per security, at security-master creation."""
    return AssetId(str(uuid.uuid4()))
