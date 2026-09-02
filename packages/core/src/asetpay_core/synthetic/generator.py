"""J-04 — Synthetic market generator. The primary validation instrument.

You invent a market where you DECIDE IN ADVANCE how predictable it is, then write
the answer to truth.json. This gives you something real market data can never
give you: a dataset where you know the right answer.

Why that matters enough to spend three days on it. Your eval harness measures a
correlation of roughly 0.03 between scores and outcomes. With real data you never
know the true value, so you can never tell whether the harness is measuring
correctly, exaggerating, or blind. Run it against this generator and it must
report the number you planted. If it reports 0.15 the instrument exaggerates; if
it reports 0.00 it is blind. Either way you learn in week 1 rather than month nine.

THE CONSTRUCTION IS THE COMBINATION MATHS, INVERTED
---------------------------------------------------
Two signals are planted with target ICs and a target correlation between them.
Given the signal correlation matrix W and the target IC vector, the loadings are

    c = W^-1 . IC

which is exactly the optimal-combination formula the combiner will later have to
discover. And the achievable combined IC is

    IC_combined = sqrt(IC^T . W^-1 . IC)

so this generator hands you a ground-truth check on the COMBINER (P2-27) as well
as on the harness (P2-03). Two signals at IC 0.02 with zero correlation combine
to 0.028; two at IC 0.04 with correlation 0.9 combine to 0.041. Diversity between
signals beats the quality of any one of them, and here you can prove it.

PLANTED PATHOLOGIES
-------------------
Four traps, each a real bias that has destroyed real backtests, each with its
location recorded in truth.json so you can verify your machinery catches it:

  restatement     a fiscal period reported once, then revised later. A store that
                  overwrites instead of versioning will silently return the
                  revised figure for dates before the revision existed.
  delisting       a company that goes to near-zero and stops trading. A universe
                  that drops it has survivorship bias; one that keeps it WITHOUT
                  the final return also does, because the loss never registers.
  ticker_reuse    one ticker, two unrelated companies, non-overlapping periods. A
                  backtest keyed on tickers merges them into one fictional firm.
  stale_fundamentals  quarterly figures carried forward daily: 252 rows a year
                  containing about 4 independent observations. Most fundamentals
                  research reporting significance is reporting this artefact.

Signals are generated with AR(1) persistence on purpose. Without it the IC series
would be serially independent and P2-06's Newey-West correction would have nothing
to bite on — the naive and HAC standard errors would agree and the test could not
demonstrate anything.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_NAMESPACE = uuid.UUID("6f1d4a2e-3c5b-4e8f-9a7d-2b6c8e0f1a34")

# Trading-day approximation. Real calendars come from the exchange; for a
# validation fixture, weekdays are sufficient and keep the generator pure.
_TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class GeneratorConfig:
    """Every knob, so truth.json can record exactly what produced the data."""

    n_assets: int = 500
    n_years: int = 5
    start: date = date(2019, 1, 1)

    # --- the planted alpha -------------------------------------------------
    ic_a: float = 0.05
    """Target information coefficient of signal A against next-day returns."""
    ic_b: float = 0.03
    """Target IC of signal B."""
    rho_ab: float = 0.20
    """Target correlation BETWEEN the two signals. Drives how much they add."""

    # --- realism -----------------------------------------------------------
    daily_vol: float = 0.018
    """Idiosyncratic daily volatility. Real equities run 1.5-2%."""
    signal_persistence: float = 0.94
    """AR(1) coefficient on the signals. ~0.94 daily gives a signal that decays
    over roughly a month."""

    ic_regime_persistence: float = 0.97
    ic_regime_amplitude: float = 0.9
    """The planted IC DRIFTS over time rather than being constant.

    This is not decoration, it is what makes the fixture able to exercise
    Newey-West at all. Persistent SIGNALS alone do not produce an autocorrelated
    IC series: if the true IC is constant and the return noise is fresh each day,
    the daily IC estimates are serially independent and the HAC correction
    correctly reports no adjustment. Real IC series are autocorrelated because
    the underlying predictive power itself moves — alpha decays, regimes change.

    So the true IC on day t is  IC_target x (1 + amplitude x u_t)  with u_t an
    AR(1) of mean zero and unit variance. The long-run mean is still exactly
    IC_target, but the series now has the persistence that makes naive standard
    errors lie. Set amplitude to 0 for a constant-IC world, and observe that the
    HAC and naive standard errors then agree — which is itself worth seeing once."""
    market_beta_mean: float = 1.0
    market_beta_sd: float = 0.3
    market_vol: float = 0.010

    # --- pathologies -------------------------------------------------------
    n_delistings: int = 12
    delisting_final_return: float = -0.85
    n_restatements: int = 8
    n_ticker_reuses: int = 3
    fundamental_lag_days: int = 45
    """Days between fiscal period end and first publication."""
    restatement_lag_days: int = 180
    """Days between first publication and the revision."""

    seed: int = 20260817

    def combined_ic(self) -> float:
        """sqrt(IC^T W^-1 IC) — the best achievable IC from these two signals."""
        w = np.array([[1.0, self.rho_ab], [self.rho_ab, 1.0]])
        ic = np.array([self.ic_a, self.ic_b])
        return float(np.sqrt(ic @ np.linalg.solve(w, ic)))

    def loadings(self) -> tuple[float, float]:
        """c = W^-1 IC — how much of each signal goes into the true expected return."""
        w = np.array([[1.0, self.rho_ab], [self.rho_ab, 1.0]])
        ic = np.array([self.ic_a, self.ic_b])
        c = np.linalg.solve(w, ic)
        return float(c[0]), float(c[1])


@dataclass
class Truth:
    """Ground truth. Everything a test needs to assert against."""

    config: dict[str, Any]
    planted_ic_a: float
    planted_ic_b: float
    planted_rho_ab: float
    planted_combined_ic: float
    loadings: dict[str, float]
    spearman_ic_a_expected: float
    spearman_ic_b_expected: float
    restatements: list[dict[str, Any]] = field(default_factory=list)
    delistings: list[dict[str, Any]] = field(default_factory=list)
    ticker_reuses: list[dict[str, Any]] = field(default_factory=list)
    fundamentals_per_year: int = 4
    trading_days_per_year: int = _TRADING_DAYS_PER_YEAR


def _asset_id(i: int) -> str:
    """Deterministic AssetId, so regenerating gives identical identifiers."""
    return str(uuid.uuid5(_NAMESPACE, f"asset-{i}"))


def _ticker(i: int) -> str:
    """Deterministic 4-letter ticker. Cosmetic only — never an identifier."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n = i
    out = []
    for _ in range(4):
        out.append(letters[n % 26])
        n //= 26
    return "".join(reversed(out))


def _trading_days(start: date, n_years: int) -> list[date]:
    days: list[date] = []
    d = start
    target = n_years * _TRADING_DAYS_PER_YEAR
    while len(days) < target:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _ar1(rng: np.random.Generator, n_steps: int, n_series: int, phi: float) -> np.ndarray:
    """AR(1) paths with unit unconditional variance.

    x_t = phi x_{t-1} + sqrt(1 - phi^2) e_t  keeps Var(x) = 1 for all t, which
    matters because the planted IC is defined against unit-variance signals.
    """
    out = np.empty((n_steps, n_series))
    out[0] = rng.standard_normal(n_series)
    scale = np.sqrt(1.0 - phi * phi)
    for t in range(1, n_steps):
        out[t] = phi * out[t - 1] + scale * rng.standard_normal(n_series)
    return out


def _cross_sectional_standardize(x: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance across assets on each day.

    The planted IC is a cross-sectional quantity, so the signals must be
    cross-sectionally standardized for it to mean what truth.json says.
    """
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x - mu) / sd


def _pearson_to_spearman(rho: float) -> float:
    """For bivariate normal, rho_spearman = (6/pi) arcsin(rho_pearson / 2).

    Worth having explicitly: the harness measures RANK IC by default, and rank IC
    is about 4.5% below the planted Pearson IC. That gap is a property of the
    estimator, not a bug, and a test that does not account for it will look like
    a failure.
    """
    return float((6.0 / np.pi) * np.arcsin(rho / 2.0))


def generate(cfg: GeneratorConfig | None = None) -> dict[str, Any]:
    """Generate the whole fixture. Returns frames plus the Truth object."""
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)

    days = _trading_days(cfg.start, cfg.n_years)
    n_t, n_a = len(days), cfg.n_assets
    assets = [_asset_id(i) for i in range(n_a)]

    # ---- signals, persistent and cross-sectionally standardized -----------
    z_a = _cross_sectional_standardize(_ar1(rng, n_t, n_a, cfg.signal_persistence))
    w = _cross_sectional_standardize(_ar1(rng, n_t, n_a, cfg.signal_persistence))
    # z_b correlates with z_a at rho, preserving unit variance
    z_b = cfg.rho_ab * z_a + np.sqrt(1.0 - cfg.rho_ab**2) * w
    z_b = _cross_sectional_standardize(z_b)

    # ---- returns: r/sigma = c_a,t z_a + c_b,t z_b + sqrt(1 - v_t) eta -----
    c_a, c_b = cfg.loadings()
    v = cfg.combined_ic() ** 2
    if v >= 1.0:
        raise ValueError(
            f"combined IC {np.sqrt(v):.3f} >= 1; the planted ICs are impossible together"
        )

    # Slow drift in the true IC, normalized so its SAMPLE mean is exactly 1.
    #
    # Why normalize rather than rely on the AR(1) having mean zero: with
    # persistence 0.97 a 1259-day path contains only about n(1-phi)/(1+phi) ~ 19
    # independent draws, so its sample mean has a standard deviation of roughly
    # 0.23. Left alone, the REALIZED IC in any given fixture lands 10-20% away
    # from the target and a tight test becomes impossible.
    #
    # That gap is not an artefact — it is the honest statement that five years
    # does not pin down a persistent IC, and the HAC t-stat is what reports it.
    # But a validation fixture has to plant the REALIZED value, not a value the
    # sample merely has in expectation. So the path is centred and scaled here,
    # and truth.json's planted IC is then exactly what the harness must recover.
    u = _ar1(rng, n_t, 1, cfg.ic_regime_persistence)[:, 0]
    u = (u - u.mean()) / (u.std() if u.std() > 1e-12 else 1.0)
    regime = 1.0 + cfg.ic_regime_amplitude * u
    c_a_t = c_a * regime
    c_b_t = c_b * regime

    v_t = c_a_t**2 + c_b_t**2 + 2.0 * c_a_t * c_b_t * cfg.rho_ab
    if float(v_t.max()) >= 1.0:
        raise ValueError("regime amplitude too large: predictable variance exceeds 1")

    eta = rng.standard_normal((n_t, n_a))
    idio = cfg.daily_vol * (
        c_a_t[:, None] * z_a + c_b_t[:, None] * z_b + np.sqrt(1.0 - v_t)[:, None] * eta
    )

    # The signal at day t predicts the return from t to t+1. Shift the
    # predictable part forward by one day so nothing is contemporaneous.
    idio = np.vstack([rng.standard_normal((1, n_a)) * cfg.daily_vol, idio[:-1]])

    # ---- add a market factor (unpredictable, for realism) -----------------
    betas = rng.normal(cfg.market_beta_mean, cfg.market_beta_sd, n_a)
    market = rng.standard_normal(n_t) * cfg.market_vol
    returns = idio + market[:, None] * betas[None, :]

    # ---- prices ------------------------------------------------------------
    p0 = rng.uniform(10.0, 200.0, n_a)
    prices = p0[None, :] * np.cumprod(1.0 + returns, axis=0)

    # ---- PATHOLOGY: delistings --------------------------------------------
    delisted_idx = rng.choice(n_a, size=cfg.n_delistings, replace=False)
    delist_day = {
        int(i): int(rng.integers(int(n_t * 0.25), int(n_t * 0.9))) for i in delisted_idx
    }
    delistings = []
    for i, t_end in delist_day.items():
        prices[t_end + 1 :, i] = np.nan
        returns[t_end + 1 :, i] = np.nan
        returns[t_end, i] = cfg.delisting_final_return
        prices[t_end, i] = prices[t_end - 1, i] * (1.0 + cfg.delisting_final_return)
        delistings.append(
            {
                "asset_id": assets[i],
                "ticker": _ticker(i),
                "delisting_date": days[t_end].isoformat(),
                "final_return": cfg.delisting_final_return,
            }
        )

    # ---- long frames --------------------------------------------------------
    date_idx = np.repeat([d.isoformat() for d in days], n_a)
    asset_idx = np.tile(assets, n_t)

    px = pd.DataFrame(
        {
            "event_date": date_idx,
            "asset_id": asset_idx,
            "close": prices.ravel(),
            "volume": np.abs(rng.lognormal(13.0, 1.2, n_t * n_a)).round(),
            "ret_1d": returns.ravel(),
        }
    ).dropna(subset=["close"])
    px["knowledge_date"] = px["event_date"]  # prices are known the day they occur

    sig = pd.DataFrame(
        {
            "as_of": date_idx,
            "asset_id": asset_idx,
            "planted_signal_a": z_a.ravel(),
            "planted_signal_b": z_b.ravel(),
        }
    )
    live = set(zip(px["event_date"], px["asset_id"], strict=True))
    sig = sig[[(d, a) in live for d, a in zip(sig["as_of"], sig["asset_id"], strict=True)]]

    # ---- PATHOLOGY: quarterly fundamentals + restatements ------------------
    q_ends, seen = [], set()
    for d in days:
        q = (d.month - 1) // 3
        key = (d.year, q)
        if key not in seen:
            seen.add(key)
            q_ends.append(d)

    restate_targets = {
        (int(a_i), int(q_i))
        for a_i, q_i in zip(
            rng.choice(n_a, cfg.n_restatements, replace=False),
            rng.integers(1, max(2, len(q_ends) - 1), cfg.n_restatements),
            strict=True,
        )
    }

    rows, restatements = [], []
    for q_i, q_end in enumerate(q_ends):
        pub = q_end + timedelta(days=cfg.fundamental_lag_days)
        if pub > days[-1]:
            continue
        vals = rng.lognormal(1.0, 0.5, n_a)
        for i in range(n_a):
            if i in delist_day and days[delist_day[i]] < q_end:
                continue
            rows.append(
                {
                    "asset_id": assets[i],
                    "period_end": q_end.isoformat(),
                    "metric": "eps",
                    "value": float(vals[i]),
                    "knowledge_date": pub.isoformat(),
                }
            )
            if (i, q_i) in restate_targets:
                rev_date = pub + timedelta(days=cfg.restatement_lag_days)
                if rev_date <= days[-1]:
                    revised = float(vals[i] * rng.uniform(0.55, 0.85))
                    rows.append(
                        {
                            "asset_id": assets[i],
                            "period_end": q_end.isoformat(),
                            "metric": "eps",
                            "value": revised,
                            "knowledge_date": rev_date.isoformat(),
                        }
                    )
                    restatements.append(
                        {
                            "asset_id": assets[i],
                            "period_end": q_end.isoformat(),
                            "metric": "eps",
                            "original_value": float(vals[i]),
                            "restated_value": revised,
                            "first_known": pub.isoformat(),
                            "restated_known": rev_date.isoformat(),
                        }
                    )
    fundamentals = pd.DataFrame(rows)

    # ---- PATHOLOGY: ticker reuse -------------------------------------------
    ident_rows, reuses = [], []
    reuse_pairs = rng.choice(n_a, size=(cfg.n_ticker_reuses, 2), replace=False)
    reused = {int(x) for pair in reuse_pairs for x in pair}
    for i in range(n_a):
        if i not in reused:
            ident_rows.append(
                {
                    "asset_id": assets[i],
                    "id_type": "ticker",
                    "id_value": _ticker(i),
                    "valid_from": days[0].isoformat(),
                    "valid_to": None,
                }
            )
    for k, (i, j) in enumerate(reuse_pairs):
        i, j = int(i), int(j)
        cut = days[int(n_t * 0.5)]
        shared = f"RCY{k}"
        ident_rows += [
            {
                "asset_id": assets[i],
                "id_type": "ticker",
                "id_value": shared,
                "valid_from": days[0].isoformat(),
                "valid_to": cut.isoformat(),
            },
            {
                "asset_id": assets[j],
                "id_type": "ticker",
                "id_value": shared,
                "valid_from": cut.isoformat(),
                "valid_to": None,
            },
        ]
        reuses.append(
            {
                "ticker": shared,
                "first_asset_id": assets[i],
                "second_asset_id": assets[j],
                "handover_date": cut.isoformat(),
            }
        )
    identifiers = pd.DataFrame(ident_rows)

    truth = Truth(
        config={
            k: (v.isoformat() if isinstance(v, date) else v) for k, v in asdict(cfg).items()
        },
        planted_ic_a=cfg.ic_a,
        planted_ic_b=cfg.ic_b,
        planted_rho_ab=cfg.rho_ab,
        planted_combined_ic=cfg.combined_ic(),
        loadings={"c_a": c_a, "c_b": c_b},
        spearman_ic_a_expected=_pearson_to_spearman(cfg.ic_a),
        spearman_ic_b_expected=_pearson_to_spearman(cfg.ic_b),
        restatements=restatements,
        delistings=delistings,
        ticker_reuses=reuses,
    )

    return {
        "prices": px,
        "signals": sig,
        "fundamentals": fundamentals,
        "identifiers": identifiers,
        "delistings": pd.DataFrame(delistings),
        "truth": truth,
        "trading_days": [d.isoformat() for d in days],
    }


def write(out_dir: Path, cfg: GeneratorConfig | None = None) -> Path:
    """Generate and write the fixture to disk. Returns the output directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = generate(cfg)
    for name in ("prices", "signals", "fundamentals", "identifiers", "delistings"):
        data[name].to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "truth.json").write_text(json.dumps(asdict(data["truth"]), indent=2))
    return out_dir


if __name__ == "__main__":  # pragma: no cover
    import sys

    target = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
    write(target)
    print(f"wrote synthetic fixture to {target}")
