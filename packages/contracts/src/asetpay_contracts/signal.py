"""The Signal contract — the only thing a signal agent is allowed to produce.

An agent emits scores. It does not size positions, place orders, or express an
opinion in prose. An agent that wants to place an order is a design error, not a
feature request.

Four rules carry more weight than their apparent triviality:

  asset      never a ticker. See ids.py.
  score      monotone in expected return but NOT calibrated to it. Calibration
             happens downstream via measured IC, which is why an agent can be
             rescaled or reimplemented without disturbing sizing.
  meta       WRITE-ONLY. Nothing downstream may read it. Enforced structurally:
             the combiner's signature does not accept it.
  confidence must validate as conditional IC (high-confidence signals measurably
             outperforming low-confidence ones) or be omitted. A confidence field
             that cannot demonstrate this is noise given weight. Default None.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Any

from asetpay_contracts.ids import AssetId

_EMPTY_META: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Signal:
    """One agent's view of one asset on one decision date."""

    as_of: date
    """The DECISION date — never the fill date. Three clocks must never be
    conflated: knowledge time (latest observable data), decision time (when the
    forecast is computed), fill time (when the position changes)."""

    asset: AssetId

    agent_id: str
    """Includes a version, e.g. 'momentum_v1.2'. An agent whose behaviour changed
    is a different agent for evaluation purposes."""

    score: float
    """Normalized to roughly [-1, 1], cross-sectional. Monotone in expected
    return, not calibrated to it."""

    horizon_days: int
    """When this view expires."""

    confidence: float | None = None
    """Omit unless validated as conditional IC."""

    features_hash: str = ""
    """For reproducibility. blake2b of the sorted feature bytes plus params."""

    meta: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_META)
    """Audit only. NEVER read downstream."""

    def __post_init__(self) -> None:
        if not (-1.0000001 <= self.score <= 1.0000001):
            raise ValueError(
                f"score must be normalized to [-1, 1], got {self.score!r}. "
                "Normalize cross-sectionally before emitting."
            )
        if self.horizon_days <= 0:
            raise ValueError(f"horizon_days must be positive, got {self.horizon_days!r}")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1] or None, got {self.confidence!r}")
        if not self.agent_id:
            raise ValueError("agent_id is required and should include a version")
