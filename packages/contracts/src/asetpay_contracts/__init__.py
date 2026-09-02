"""asetpay_contracts — the coordination surface between the Truth and Judgment rails.

HARD RULE: this package imports ONLY the Python standard library. Not pydantic,
not numpy, not pandas. Every other environment in the workspace depends on it, so
any third-party import here forces the models environment (torch) to agree with
the backtest environment (nautilus) about that library's version. Those
disagreements are how a project becomes unbuildable around month three.

Enforced by tests/test_contracts_zero_deps.py, which runs in CI.
"""

from asetpay_contracts.ids import AssetId, new_asset_id
from asetpay_contracts.protocols import CostModel, FeatureSet, Store, StoreView
from asetpay_contracts.signal import Signal

__all__ = [
    "AssetId",
    "CostModel",
    "FeatureSet",
    "Signal",
    "Store",
    "StoreView",
    "new_asset_id",
]
