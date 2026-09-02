#!/usr/bin/env python3
"""GATE 0, second half — run it and read the output.

The week-1 exit criterion is two claims. This script demonstrates the one that
does not need your GitHub account:

    the measurement instrument gives the right answer on a problem where the
    right answer is known.

The other half — snapshots landing three nights unattended — is P1's, and needs
the repo and the Alpaca keys.
"""

from __future__ import annotations

from pathlib import Path

from asetpay_core.evaluation import effective_sample_size, information_coefficient
from asetpay_core.store import FixtureStore
from asetpay_core.synthetic import write

DATA = Path("data/synthetic")
if not (DATA / "truth.json").exists():
    print("generating fixture...")
    write(DATA)

st = FixtureStore(DATA)
t = st.truth
px, sg = st.all_prices(), st.planted_signals()

print("=" * 78)
print("PLANTED  (data/synthetic/truth.json)")
print(
    f"   IC_a {t['planted_ic_a']:.4f}   IC_b {t['planted_ic_b']:.4f}   rho(a,b) {t['planted_rho_ab']:.2f}"
)
print(
    f"   combined IC {t['planted_combined_ic']:.4f}  = sqrt(IC' W^-1 IC), the combiner's target"
)
print(
    f"   pathologies: {len(t['restatements'])} restatements, {len(t['delistings'])} delistings, "
    f"{len(t['ticker_reuses'])} ticker reuses"
)
print("=" * 78)

ok = True
for col, agent, target, rank_target in (
    ("planted_signal_a", "planted_a_v1", t["planted_ic_a"], t["spearman_ic_a_expected"]),
    ("planted_signal_b", "planted_b_v1", t["planted_ic_b"], t["spearman_ic_b_expected"]),
):
    r = information_coefficient(
        sg[["as_of", "asset_id", col]].rename(columns={col: "score"}), px, agent_id=agent
    )
    err = abs(r.mean_pearson_ic - target)
    ok &= err < 0.006
    print()
    print(r)
    print(
        f"    planted {target:.4f} -> measured {r.mean_pearson_ic:.4f}   error {err:.4f}   "
        f"{'PASS' if err < 0.006 else 'FAIL'}"
    )
    print(
        f"    rank IC expected {rank_target:.4f} (Spearman sits ~4.5% below Pearson) "
        f"-> measured {r.mean_rank_ic:.4f}"
    )
    print(
        f"    effective sample size {effective_sample_size(r.n_periods, r.ic_autocorr_lag1):.0f} "
        f"of {r.n_periods} periods"
    )

print()
print("=" * 78)
print(f"GATE 0 (instrument half): {'PASS' if ok else 'FAIL'}")
print("Remaining for Gate 0: three nights of snapshots landing unattended (P1-06).")
print("=" * 78)
raise SystemExit(0 if ok else 1)
