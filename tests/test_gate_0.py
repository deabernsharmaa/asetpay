"""GATE 0 — the exit criterion for week 1, as an executable test.

Two claims, both of which must hold before Phase 1 starts:

  1. The measurement instrument gives the RIGHT ANSWER on a problem where the
     right answer is known. (P2-03)
  2. The instrument is capable of being WRONG — planting zero alpha must produce
     a measurement of zero. An instrument that always reports a number you like
     is not measuring anything. (P1's half of the cross-verification.)

Claim 2 is the one people skip. It is here as a separate test on purpose.
"""

from __future__ import annotations

import numpy as np
import pytest

from asetpay_core.evaluation import (
    effective_sample_size,
    forward_returns,
    information_coefficient,
)
from asetpay_core.synthetic import GeneratorConfig, generate


def _measure(data, col: str):
    sig = data["signals"][["as_of", "asset_id", col]].rename(columns={col: "score"})
    return information_coefficient(sig, data["prices"], agent_id=f"test::{col}")


# --------------------------------------------------------------------------
# Claim 1 — the instrument recovers what was planted
# --------------------------------------------------------------------------


@pytest.mark.parametrize("planted", [0.02, 0.05, 0.08])
def test_harness_recovers_planted_ic(planted: float) -> None:
    """The headline Gate 0 assertion, across several planted values.

    One planted value could pass by luck. Three cannot.
    """
    data = generate(GeneratorConfig(ic_a=planted, ic_b=0.01, n_assets=400, n_years=4))
    r = _measure(data, "planted_signal_a")
    assert abs(r.mean_pearson_ic - planted) < 0.006, (
        f"planted IC {planted:.4f}, harness measured {r.mean_pearson_ic:.4f}. "
        "The instrument is miscalibrated — fix it before trusting any agent."
    )


def test_rank_ic_matches_the_spearman_conversion() -> None:
    """Rank IC sits ~4.5% below Pearson IC for jointly normal data.

    rho_spearman = (6/pi) arcsin(rho_pearson / 2)

    This is a property of the estimator, not a defect. Recorded as a test so
    nobody later 'fixes' the harness to make the two agree.
    """
    data = generate(GeneratorConfig(n_assets=400, n_years=4))
    r = _measure(data, "planted_signal_a")
    expected = data["truth"].spearman_ic_a_expected
    assert abs(r.mean_rank_ic - expected) < 0.005
    assert r.mean_rank_ic < r.mean_pearson_ic


def test_second_signal_and_their_correlation_are_also_recovered() -> None:
    """Both planted signals, and the correlation between them.

    The signal correlation is what drives how much a second agent adds, so the
    fixture has to get it right for the combiner ground truth to mean anything.
    """
    cfg = GeneratorConfig(ic_a=0.05, ic_b=0.03, rho_ab=0.30, n_assets=400, n_years=4)
    data = generate(cfg)
    assert abs(_measure(data, "planted_signal_b").mean_pearson_ic - cfg.ic_b) < 0.006

    sig = data["signals"]
    rho = (
        sig.groupby("as_of")[["planted_signal_a", "planted_signal_b"]]
        .corr()
        .iloc[0::2, 1]
        .mean()
    )
    assert abs(rho - cfg.rho_ab) < 0.02


# --------------------------------------------------------------------------
# Claim 2 — the instrument is capable of reporting nothing
# --------------------------------------------------------------------------


def test_zero_planted_alpha_measures_as_zero() -> None:
    """P1's cross-verification: plant nothing, and the harness must find nothing.

    If this fails, every positive result the harness has ever produced is
    suspect, because the instrument reports signal where none exists.
    """
    data = generate(GeneratorConfig(ic_a=0.0, ic_b=0.0, n_assets=400, n_years=4))
    r = _measure(data, "planted_signal_a")
    assert abs(r.mean_pearson_ic) < 0.006, (
        f"no alpha was planted, but the harness reported IC {r.mean_pearson_ic:.4f}"
    )
    assert abs(r.hac_t) < 3.0, f"spurious significance: HAC t = {r.hac_t:.2f}"


def test_shuffling_the_scores_destroys_the_ic() -> None:
    """A second way of being wrong: break the score-to-asset mapping and the
    measured IC must collapse. Catches an evaluator that is accidentally
    correlating something with itself."""
    data = generate(GeneratorConfig(ic_a=0.06, n_assets=400, n_years=4))
    sig = data["signals"][["as_of", "asset_id", "planted_signal_a"]].rename(
        columns={"planted_signal_a": "score"}
    )
    rng = np.random.default_rng(0)
    shuffled = sig.copy()
    shuffled["score"] = rng.permutation(shuffled["score"].to_numpy())

    assert _measure(data, "planted_signal_a").mean_pearson_ic > 0.04
    assert (
        abs(
            information_coefficient(
                shuffled, data["prices"], agent_id="shuffled"
            ).mean_pearson_ic
        )
        < 0.006
    )


# --------------------------------------------------------------------------
# The Newey-West correction must actually do something
# --------------------------------------------------------------------------


def test_hac_standard_errors_materially_exceed_naive_ones() -> None:
    """P2-06's done-when, as a test.

    With a drifting IC the daily IC series is autocorrelated, naive standard
    errors understate the truth, and the HAC correction must visibly widen them.
    If these ever agree, the correction is not wired up.
    """
    data = generate(
        GeneratorConfig(ic_a=0.05, ic_regime_amplitude=0.9, n_assets=400, n_years=4)
    )
    r = _measure(data, "planted_signal_a")
    assert r.ic_autocorr_lag1 > 0.2, "fixture is not producing a persistent IC series"
    assert r.se_inflation > 1.3, (
        f"HAC se is only {r.se_inflation:.2f}x naive — the correction is not biting"
    )
    assert r.hac_t < r.naive_t


def test_constant_ic_needs_no_hac_correction() -> None:
    """The control for the test above. With amplitude 0 the true IC is constant,
    the IC series is serially independent, and HAC correctly reports almost no
    adjustment. Without this, a passing HAC test could just mean 'HAC always
    inflates', which would be a different bug."""
    data = generate(
        GeneratorConfig(ic_a=0.05, ic_regime_amplitude=0.0, n_assets=400, n_years=4)
    )
    r = _measure(data, "planted_signal_a")
    assert abs(r.ic_autocorr_lag1) < 0.12
    assert r.se_inflation < 1.25


def test_effective_sample_size_collapses_under_persistence() -> None:
    """P2-07: 1000 correlated observations are worth far fewer independent ones."""
    assert effective_sample_size(1000, 0.0) == pytest.approx(1000)
    assert effective_sample_size(1000, 0.9) == pytest.approx(1000 * 0.1 / 1.9)
    assert effective_sample_size(252, 0.98) < 5  # quarterly data carried daily


# --------------------------------------------------------------------------
# Alignment — the bug that would bless every agent you ever test
# --------------------------------------------------------------------------


def test_forward_returns_never_look_backwards() -> None:
    """A score stamped as_of=t must be paired with a return beginning after t.

    Off by one in the wrong direction and the evaluator has a lookahead bug — at
    which point it will report a wonderful IC for every agent, including
    worthless ones.
    """
    data = generate(GeneratorConfig(n_assets=50, n_years=1))
    px = data["prices"]
    fwd = forward_returns(px, horizon_days=1)

    one = px["asset_id"].iloc[0]
    p = px[px["asset_id"] == one].sort_values("event_date").reset_index(drop=True)
    f = fwd[fwd["asset_id"] == one].sort_values("event_date").reset_index(drop=True)

    row = f.iloc[10]
    t_close = p.loc[p["event_date"] == row["event_date"], "close"].iloc[0]
    nxt = p[p["event_date"] > row["event_date"]].iloc[0]["close"]
    assert row["fwd_return"] == pytest.approx(nxt / t_close - 1.0, rel=1e-9)


def test_a_signal_that_peeks_at_the_answer_is_measured_as_near_perfect() -> None:
    """The positive control on the evaluator itself.

    Feed the harness a 'signal' that IS tomorrow's return. It must report an IC
    of essentially 1.0. If it reports anything materially lower, the harness is
    misaligning scores against returns — and a misaligned evaluator will happily
    bless worthless agents, which is far worse than one that is merely noisy.

    Note the scale gap this exposes: a leaked signal reads ~1.0, an honest one
    ~0.05. Anything in your real results closer to the first number than the
    second is leakage, not alpha.
    """
    data = generate(GeneratorConfig(ic_a=0.05, n_assets=400, n_years=3))
    leaked = forward_returns(data["prices"], horizon_days=1).rename(
        columns={"event_date": "as_of", "fwd_return": "score"}
    )
    r = information_coefficient(leaked, data["prices"], agent_id="leaked_v1")
    assert r.mean_pearson_ic > 0.99, (
        f"a signal equal to the answer measured only {r.mean_pearson_ic:.3f} — "
        "the harness is misaligning scores against forward returns"
    )
    honest = _measure(data, "planted_signal_a").mean_pearson_ic
    assert r.mean_pearson_ic > honest * 15
