"""J-02's hard rule, enforced: asetpay_contracts imports the stdlib and nothing else.

Why this is worth a dedicated test rather than a code-review habit. Every
environment in the workspace depends on `contracts`. The moment it gains a
third-party import, the models environment (torch) is forced to agree with the
backtest environment (nautilus_trader) about that library's version. Those
disagreements are exactly how a multi-environment project becomes unbuildable,
and it happens gradually enough that nobody notices until it is expensive.

The test runs contracts in a SUBPROCESS with a clean interpreter, then inspects
what actually got imported. Checking in-process would not work — by then pandas
is already loaded by the test session itself.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys

_PROBE = r"""
import sys, json
before = set(sys.modules)
import asetpay_contracts  # noqa: F401
new = {m.split(".")[0] for m in set(sys.modules) - before if not m.startswith("_")}
# sys.stdlib_module_names is the authoritative list and maintains itself across
# Python versions — far better than a hand-written allowlist, which goes stale
# silently and then either blocks a legitimate stdlib import or lets a real
# dependency through.
third_party = sorted(new - set(sys.stdlib_module_names) - {"asetpay_contracts"})
print(json.dumps(third_party))
"""


def test_contracts_imports_only_stdlib() -> None:
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    third_party = json.loads(out.stdout.strip().splitlines()[-1])
    assert not third_party, (
        f"asetpay_contracts pulled in third-party modules: {third_party}. "
        "The contract package must stay dependency-free — see its docstring."
    )


def test_pandas_is_not_imported_by_contracts() -> None:
    """Named explicitly because protocols.py references pd.DataFrame in its type
    hints. Those live under `if TYPE_CHECKING:` and must never execute."""
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import asetpay_contracts; "
            "print('pandas' in sys.modules or 'numpy' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


def test_signal_rejects_an_unnormalized_score() -> None:
    """The contract validates at construction, so a mis-scaled agent fails loudly
    at the boundary rather than quietly skewing the combiner."""
    import datetime as dt

    import pytest

    from asetpay_contracts import Signal, new_asset_id

    ok = Signal(
        as_of=dt.date(2026, 1, 5),
        asset=new_asset_id(),
        agent_id="momentum_v1",
        score=0.42,
        horizon_days=20,
    )
    assert ok.confidence is None, "confidence must default to None until validated"

    with pytest.raises(ValueError, match="normalized"):
        Signal(
            as_of=dt.date(2026, 1, 5),
            asset=new_asset_id(),
            agent_id="broken_v1",
            score=7.5,
            horizon_days=20,
        )

    with pytest.raises(ValueError, match="agent_id"):
        Signal(
            as_of=dt.date(2026, 1, 5),
            asset=new_asset_id(),
            agent_id="",
            score=0.1,
            horizon_days=20,
        )


def test_signal_is_frozen() -> None:
    """Immutable by construction: a Signal that can be edited after emission is a
    Signal whose audit trail means nothing."""
    import datetime as dt

    import pytest

    from asetpay_contracts import Signal, new_asset_id

    s = Signal(
        as_of=dt.date(2026, 1, 5),
        asset=new_asset_id(),
        agent_id="momentum_v1",
        score=0.1,
        horizon_days=5,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.score = 0.9  # type: ignore[misc]
