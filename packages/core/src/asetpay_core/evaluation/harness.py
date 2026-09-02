"""P2-03 / P2-06 — the evaluation harness. The measurement instrument.

This is not quality assurance sitting beside the model. IC is a COEFFICIENT in
the forecast (alpha = IC x sigma x z), estimated from data with large error. If
IC is wrong, alpha is wrong, weights are wrong and sizing is wrong, regardless of
how sophisticated the feature model is. The harness is parameter estimation for
the model, which is why it gets built before the combiner.

WHAT IT COMPUTES
----------------
For each decision date, the cross-sectional correlation between scores and the
NEXT period's returns. That series of daily ICs is then summarised.

Two flavours, both reported, because they answer different questions:
  pearson_ic   linear correlation. Matches the planted value exactly, so this is
               what a ground-truth test should assert against.
  rank_ic      Spearman. Robust to outliers and the one to trust on real data.
               For bivariate normal it sits about 4.5% BELOW the Pearson value —
               rho_s = (6/pi) arcsin(rho_p / 2) — which is a property of the
               estimator, not a bug. A test that ignores this looks like a failure.

THE CORRECTION THAT MATTERS MOST
--------------------------------
IC series are serially correlated: today's momentum score is nearly yesterday's,
so consecutive ICs are not independent draws. Naive standard errors assume they
are, and therefore understate the true uncertainty — sometimes by a large
multiple. This single omission is how a signal with no real edge reports t = 3
and looks publishable.

Newey-West (HAC) corrects for it. Both are reported side by side so the size of
the correction is visible rather than assumed. If they agree, either your signal
has no persistence or your lag length is too short.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class ICResult:
    """Everything needed to judge whether a signal predicts anything."""

    agent_id: str
    n_periods: int
    mean_pearson_ic: float
    mean_rank_ic: float
    sd_ic: float

    naive_se: float
    naive_t: float
    hac_se: float
    hac_t: float
    hac_maxlags: int

    ic_autocorr_lag1: float
    """How persistent the IC series is. The bigger this is, the more the naive
    t-stat lies, and the more the HAC correction should differ from it."""

    @property
    def se_inflation(self) -> float:
        """hac_se / naive_se. The factor by which naive statistics flatter you."""
        return self.hac_se / self.naive_se if self.naive_se > 0 else float("nan")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["se_inflation"] = self.se_inflation
        return d

    def __str__(self) -> str:
        return (
            f"{self.agent_id}: rank IC {self.mean_rank_ic:+.4f} "
            f"(pearson {self.mean_pearson_ic:+.4f}) over {self.n_periods} periods\n"
            f"    naive  t = {self.naive_t:6.2f}  (se {self.naive_se:.5f})\n"
            f"    HAC    t = {self.hac_t:6.2f}  (se {self.hac_se:.5f}, "
            f"{self.se_inflation:.2f}x naive, {self.hac_maxlags} lags)\n"
            f"    IC autocorrelation (lag 1) = {self.ic_autocorr_lag1:+.3f}"
        )


def forward_returns(prices: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """Return from the close on date t to the close `horizon_days` later.

    Alignment is the thing to get right: a score stamped as_of=t must be paired
    with a return that begins AFTER t. Off by one in the wrong direction and you
    have built a lookahead bug into your evaluator, which will then bless every
    agent you test.
    """
    px = prices[["event_date", "asset_id", "close"]].copy()
    px = px.sort_values(["asset_id", "event_date"])
    px["fwd_close"] = px.groupby("asset_id", observed=True)["close"].shift(-horizon_days)
    px["fwd_return"] = px["fwd_close"] / px["close"] - 1.0
    return px[["event_date", "asset_id", "fwd_return"]].dropna()


def _newey_west_lags(n: int) -> int:
    """Standard automatic bandwidth: floor(4 (n/100)^(2/9))."""
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def information_coefficient(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    agent_id: str,
    score_col: str = "score",
    date_col: str = "as_of",
    horizon_days: int = 1,
    min_names: int = 30,
) -> ICResult:
    """Measure whether `scores` predicted the next `horizon_days` of returns."""
    fwd = forward_returns(prices, horizon_days)
    df = scores.rename(columns={date_col: "event_date"}).merge(
        fwd, on=["event_date", "asset_id"], how="inner"
    )
    if df.empty:
        raise ValueError(
            "no overlap between scores and forward returns — check the date "
            "column and that scores are stamped with the DECISION date"
        )

    per_day = []
    for d, g in df.groupby("event_date", observed=True):
        if len(g) < min_names:
            continue
        s, r = g[score_col].to_numpy(), g["fwd_return"].to_numpy()
        if np.std(s) < 1e-12 or np.std(r) < 1e-12:
            continue
        pear = float(np.corrcoef(s, r)[0, 1])
        rank = float(
            np.corrcoef(pd.Series(s).rank().to_numpy(), pd.Series(r).rank().to_numpy())[0, 1]
        )
        per_day.append((d, pear, rank))

    if len(per_day) < 20:
        raise ValueError(f"only {len(per_day)} usable periods; need at least 20")

    ic = pd.DataFrame(per_day, columns=["event_date", "pearson", "rank"]).sort_values(
        "event_date"
    )
    y = ic["pearson"].to_numpy()
    n = len(y)

    naive_se = float(np.std(y, ddof=1) / np.sqrt(n))
    lags = _newey_west_lags(n)
    ols = sm.OLS(y, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    hac_se = float(ols.bse[0])

    ac1 = float(pd.Series(y).autocorr(lag=1)) if n > 2 else float("nan")

    return ICResult(
        agent_id=agent_id,
        n_periods=n,
        mean_pearson_ic=float(y.mean()),
        mean_rank_ic=float(ic["rank"].mean()),
        sd_ic=float(np.std(y, ddof=1)),
        naive_se=naive_se,
        naive_t=float(y.mean() / naive_se) if naive_se > 0 else float("nan"),
        hac_se=hac_se,
        hac_t=float(ols.tvalues[0]),
        hac_maxlags=lags,
        ic_autocorr_lag1=ac1,
    )


def effective_sample_size(n_obs: int, autocorr: float) -> float:
    """How many INDEPENDENT observations n_obs correlated ones are worth.

    n_eff = n (1 - rho) / (1 + rho)

    The reason this matters (P2-07): quarterly fundamentals carried forward daily
    give 252 rows a year containing about 4 real observations. Reporting
    significance on the 252 is the single most common way fundamentals research
    manufactures a result out of nothing.
    """
    if not -1.0 < autocorr < 1.0:
        return float(n_obs)
    return float(n_obs * (1.0 - autocorr) / (1.0 + autocorr))
