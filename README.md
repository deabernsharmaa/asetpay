# asetpay

Point-in-time research and decision system. Phase 0 scaffold.

**P1 owns the edges**, where the system touches reality.
**P2 owns the middle**, where it thinks.

---

## What is already built

Everything that could be completed without your GitHub repo, Alpaca account, or
local Postgres. **44 tests pass** — but 10 of them need a database, and `pytest`
SKIPS those silently when `ASETPAY_TEST_DSN` is unset. On a laptop with no
Postgres you will see `34 passed, 10 skipped`, and the schema guarantees below
will be unverified. `make testdb` is the target that actually runs them.

| Task | What landed | Verified by |
|---|---|---|
| J-01 | uv workspace, four environments | `uv sync` |
| J-02 | `Signal`, `AssetId` — dependency-free | `test_contracts_zero_deps.py` |
| J-03 | The four Protocols | `protocols.py` |
| J-04 | Synthetic generator: planted alpha + 4 pathologies | `test_gate_0.py` |
| J-05 | ruff, mypy, pytest, import-linter, purity check | `.github/workflows/ci.yml` |
| P1-03 | Schema with GiST exclusion constraints | `test_migrations.py` (10 tests, real Postgres) |
| P1-05 | Snapshotter + deterministic manifest | `test_snapshotter.py` |
| P1-06 | Scheduled GitHub Actions capture | `.github/workflows/snapshot.yml` |
| P1-06b | Gap detector, opens an Issue when stale | `.github/workflows/gap-detector.yml` |
| P1-08 | Manifest determinism | `test_snapshotter.py` |
| P1-19 | Feature purity AST check | `scripts/check_feature_purity.py` |
| P2-02 | FixtureStore with correct bitemporal semantics | `test_pathologies.py` |
| P2-03 | Eval harness: IC, Newey-West, effective sample size | `test_gate_0.py` |

### Gate 0, second half — already passing

```
planted_a_v1: rank IC +0.0490 (pearson +0.0515) over 1259 periods
    naive  t =  28.94  (se 0.00178)
    HAC    t =  14.69  (se 0.00351, 1.97x naive, 7 lags)
    IC autocorrelation (lag 1) = +0.446
```

Planted 0.0500, measured 0.0515. **The instrument reports what was planted.**

Note the second line. The naive t-stat says 28.9; Newey-West says 14.7. That
gap — 1.97x — is the whole reason the correction exists. A harness without it
would have reported a t-stat twice as confident as the data supports.

---

## Start here

```bash
uv sync --all-packages          # four environments; .python-version pins 3.12
make fixture                    # generate the synthetic market
make gate0                      # prove the harness recovers planted alpha
make test                       # 34 pass, 10 skip without a database
make testdb                     # the 10 schema tests, against real Postgres
```

Python is pinned to 3.12 by `.python-version`. Without it `uv` resolves to the
newest interpreter satisfying `>=3.12`, which is not the one CI or your
colleague is running — and a research result that reproduces on one machine and
not the other costs a day to diagnose every time.

`make gate0` is the one to run first. It is the week-1 exit criterion.

---

## The four Protocols — read before writing any code

`packages/contracts/src/asetpay_contracts/protocols.py`

P2 writes code against these **descriptions**. P1 builds the real
implementations behind them. That is why P2 is not blocked: `FixtureStore`
already satisfies `Store` over synthetic data, so the whole measurement stack
can be built now, and **when P1's real store lands nothing in P2's code
changes.**

Changing anything in that file needs both of you in the room. It is the only
mandatory meeting in the project.

---

## Why `contracts` has no dependencies

Every environment depends on it. One third-party import there forces the models
environment (torch) to agree with the backtest environment (nautilus_trader)
about that library's version. Those disagreements are how a multi-environment
project becomes unbuildable — gradually, and then expensively.

`tests/test_contracts_zero_deps.py` runs the package in a clean subprocess and
diffs what got imported against `sys.stdlib_module_names`. It fails CI.

---

## The synthetic generator

`packages/core/src/asetpay_core/synthetic/generator.py`

You choose how predictable the market is, generate prices so it is true, and
write the answer to `truth.json`. That gives you something real data never
does: **a dataset where you know the right answer.** If the harness cannot
recover a planted IC of 0.05, the harness is broken — and you learn that in
week 1 rather than month nine.

Two design points worth knowing before you change anything:

**The construction is the combination maths, inverted.** Given target ICs and a
target correlation between the two signals, the loadings are `c = W⁻¹·IC` — the
same formula the combiner will later have to discover — and the achievable
combined IC is `sqrt(ICᵀ·W⁻¹·IC)`. So this fixture is ground truth for the
combiner (P2-27) as well as for the harness.

**The planted IC drifts on purpose.** Persistent signals alone do **not**
produce an autocorrelated IC series: with a constant true IC and fresh noise
each day, the daily ICs are serially independent and Newey-West correctly
reports no adjustment. Real IC series are autocorrelated because the underlying
predictive power itself moves. So the true IC follows a slow AR(1), normalised
so its sample mean is exactly the planted value. Set
`ic_regime_amplitude=0` and watch HAC and naive standard errors converge —
that control is `test_constant_ic_needs_no_hac_correction`.

### The four planted pathologies

Each is a real bias that has destroyed real backtests. Locations recorded in
`truth.json`; each has a test in `tests/test_pathologies.py`.

| Trap | The bug it catches |
|---|---|
| Restatement | A store that overwrites returns the revised figure for dates before the revision existed |
| Delisting | A universe that drops the name, or keeps it without its final return, hides the loss |
| Ticker reuse | Resolving with today's mapping merges two companies into one fictional firm |
| Stale fundamentals | 252 rows a year containing about 4 observations |

---

## The schema makes mistakes impossible

`migrations/001_security_master.sql`

```sql
CONSTRAINT one_owner_per_identifier_per_day
    EXCLUDE USING gist (id_type WITH =, id_value WITH =, valid WITH &&)
```

Two assets cannot claim the same ticker on the same day. The **database refuses
the insert**. Structural, not disciplinary — which matters because P1's bugs are
silent: nothing crashes, the backtest just looks better than it should.

`delisting_returns.final_return` is `NOT NULL` for the same reason. You cannot
half-record a delisting. When free data cannot supply the number, the name goes
to `universe_exclusions` and the gap stays visible.

All ten constraint tests run against a real Postgres 16 in CI.

---

## What you still have to do yourselves

These need accounts or machines I do not have.

1. **Create the GitHub repo** and push this. Private is fine — 2,000 Actions
   minutes/month is ample for a ~3-minute daily job.

   ```bash
   gh repo create asetpay --private --source=. --push
   ```
2. **P1-14b — a price feed key.** The default source is Polygon's free
   "Stocks Basic" tier: put `POLYGON_API_KEY` in `.env` and in repo secrets.

   The feed and the broker are **separate decisions** and this repo no longer
   couples them. See "Where prices come from" below. Alpaca remains a supported
   source (`PRICE_SOURCE=alpaca`) and a candidate Phase 4 broker where residency
   allows, but broker eligibility is jurisdictional and must never be allowed to
   stop history accruing.
3. **Validate the universe.** `universe.txt` is seeded with 502 S&P 500 names
   and decides what gets captured every night. Once the keys exist:

   ```bash
   make universe                                    # report drift, change nothing
   python scripts/build_universe.py --add-all-active  # widen, then commit the diff
   ```

   The script is additive only: it never removes a symbol, because a delisted
   name dropped from the universe is the first move of survivorship bias — the
   pathology `truth.json` plants a trap for. Widening is nearly free (~30
   requests a night), so capture broadly now and narrow at research time.

4. **Enable both workflows**, then run `snapshot` once via `workflow_dispatch`
   to confirm a release appears. **This is the item that cannot wait** — every
   day of delay is a day of history you cannot buy back.
5. **P1-02 — local Postgres**, native install, no Docker:
   `brew install postgresql@16` or `apt install postgresql-16`, then
   `psql -f migrations/001_security_master.sql`.
6. **P2-01 — inventory your existing agents.** What do they output, how often,
   over what coverage? Input to P2-16.
7. **J-03 — sit down together and read the Protocols.** Change them now if they
   are wrong. Cheap today, expensive in month three.

Then Gate 0 is: snapshots landing three nights unattended (yours), and the
harness recovering planted alpha (already passing).

---

## Layout

```
packages/
  contracts/   Signal, AssetId, the four Protocols   — ZERO dependencies
  core/        synthetic, store, evaluation, signals
  snapshotter/ daily capture + the PriceSource implementations
migrations/    Postgres schema with the exclusion constraints
scripts/       feature purity checker, universe builder
tests/         78 tests, incl. Gate 0, the four pathologies, and the CI gates
universe.txt   every name ever in scope — additive, never pruned
universe_delisted.txt  what has stopped trading, and when
uv.lock        committed — CI runs `--frozen` and fails without it
```

Four environments on purpose. `torch` + OpenBB + `nautilus_trader` + `cvxpy` in
one resolver is a fight you lose repeatedly; each loss costs a day.

---

## Where prices come from

`packages/snapshotter/src/asetpay_snapshotter/sources.py`

The snapshotter was the only place in this codebase with a vendor name baked
into it. It is now a `PriceSource` Protocol with implementations behind it,
selected by `PRICE_SOURCE`, and the choice is recorded in every manifest.

`PriceSource` is deliberately **not** in `contracts`. contracts is the
coordination surface between the two rails and changing it needs both people in
the room; P2 never touches the price feed, it reads the Store.

| Source | Requests per night, 502 names | Volume | Free tier |
|---|---|---|---|
| **polygon** (default) | **1** — the whole US market | **consolidated** | grouped daily, end-of-day |
| alpaca | 3 (chunked at 200) | IEX only | no per-symbol quota |

The free-tier request maths is what decided this, and it is worth understanding
before anyone changes it:

- **polygon** returns every US ticker for a date in one call, so the response is
  not filtered to `universe.txt` — it is not cheaper to ask for less, and the
  universe you want in month six is not the one you can name in week one.
  `universe.txt` becomes a coverage **floor** that gets checked rather than a
  list that silently bounds what history exists.
- **tiingo** is not implemented: one request per symbol against a free tier
  metered in unique symbols per month does not fit a daily full-universe
  capture, and you would discover that on night three.
- **moomoo and IBKR** are not implemented, and should not be. Both need a
  logged-in gateway daemon (OpenD, IB Gateway) alive on an ephemeral CI runner,
  and moomoo meters historical candles by *account assets* — 100 per 7 days
  below 10,000 HKD. These are execution venues, not research feeds.

### Why the default is not a brokerage

Alpaca was doing two unrelated jobs: the nightly research feed, and the Phase 4
execution broker. Broker eligibility is jurisdictional. The feed has a clock on
it. Coupling them means a residency question can stop history accruing, and that
is the one loss this project cannot recover from.
`test_the_default_source_is_not_a_brokerage` holds that seam open.

### The volume caveat, now scoped

IEX is one venue carrying a few percent of consolidated US volume: prices are
representative, **volume is not**. Anything computed from the volume column —
ADV, dollar volume, the liquidity floor (P1-25), the per-name cost model
(P1-23) — measures venue participation rather than market liquidity, and
understates costs in exactly the small caps where a naive backtest finds its
most exciting fake results.

On `polygon` this is **not** a live problem: grouped daily is consolidated. It
returns if you switch to `alpaca` on the free tier. Either way the manifest
records the provider, so the seam is visible in the data rather than remembered.
Treat P1-24's bands (2–6 bps large cap, 25–70 bps small cap) as the check that
catches you if this gets forgotten.

### Two dates, never one

`event_date` is when the trading happened. `knowledge_date` is when we could
first have known about it. Collapsing them is the lookahead this project exists
to prevent.

The job runs 03:00 UTC Wednesday and captures Tuesday's session. Nobody
possessed Tuesday's closing price at any point during Tuesday, so stamping those
rows `knowledge_date = Tuesday` would make `as_of(Tuesday)` hand a strategy a
number that did not exist yet — a few hours of lookahead, applied uniformly, in
the direction that flatters every backtest.

So `event_date` comes from the bar's own timestamp and `knowledge_date` is the
date the capture ran. `as_of(Tuesday)` cannot see Tuesday's close, and the
one-day trade lag stops being a convention someone has to remember. Same
reasoning as the GiST exclusion constraints: structural, not disciplinary.

A `knowledge_date` earlier than the session it describes is refused outright.
Backfills pass `--knowledge-date` explicitly, so a historical load states when
the data actually became knowable rather than claiming today's knowledge for a
decade-old session.

Releases are tagged by **session**, and the workflow refuses to publish over an
existing tag. `action-gh-release` will happily update a release in place, which
would rewrite a day of history leaving no trace. A correction is a new snapshot,
never an overwrite.

### What every snapshot now records

`provider`, the source's own `parameters`, `universe_size`, `expected_trading`,
`universe_coverage` and `missing_symbols`.

Coverage is measured against the names **expected to trade on that session**,
not against the whole universe. `universe.txt` is additive — a delisted name is
never removed, because its history stays in scope — so `universe_delisted.txt`
records what has stopped trading and when, and coverage uses the difference.

Without that split, coverage decays a little with every corporate action until
it trips the 90% floor for reasons that are not failures. The first live capture
came back at 0.996 because AvalonBay and Equity Residential had merged into
VMRK two weeks earlier. The fix somebody reaches for at 2am is lowering the
floor, which disables the check entirely. `universe_delisted.txt` is the interim
form of `universe_exclusions` and `delisting_returns` (P1-15, P1-17); when those
tables land, this file is their seed.

`missing_symbols` names what did not arrive. A coverage number tells you
something is wrong; the list tells you what, and it survives in the manifest
rather than having to be re-derived from the parquet. Row count alone cannot distinguish a
full market day from a response carrying ten thousand rows and none of the names
you asked for, which is what a wrong date or a lapsed entitlement produces. A
capture with rows but coverage below 90% is refused rather than published.

`schema_version` is **3**: `vwap` and `trade_count` are now captured. Both are
free from both providers, both are exactly what a per-name cost model wants, and
neither can be bought back later.

---

## Conventions worth keeping

- **A test that has never failed may be incapable of failing.** Every important
  check here has a negative control: `test_zero_planted_alpha_measures_as_zero`,
  `test_constant_ic_needs_no_hac_correction`,
  `test_a_signal_that_peeks_at_the_answer_is_measured_as_near_perfect`.
- **Gates are verified by the person who did not build the thing.**
- **`meta` on a `Signal` is write-only.** The combiner's signature does not
  accept it, so this is structural rather than a rule to remember.
- Anything in Postgres must be rebuildable from snapshots plus migrations.
  Never share a database between laptops.

---

*Not financial advice.*
