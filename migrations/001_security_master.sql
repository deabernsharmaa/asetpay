-- P1-03 — Security master, identifiers, universe, delisting returns.
--
-- The theme of this file: make the mistakes IMPOSSIBLE rather than discouraged.
-- Every constraint below replaces a rule somebody would otherwise have to
-- remember. A backtest bug that the database refuses to let you create is worth
-- more than any amount of code review, because P1's bugs are silent — nothing
-- crashes, the backtest just looks better than it should.
--
-- Requires PostgreSQL 13+ for gist with btree_gist. Tested against 16.

CREATE EXTENSION IF NOT EXISTS btree_gist;

-- ---------------------------------------------------------------------------
-- Securities. asset_id is permanent and never reused.
-- ---------------------------------------------------------------------------
CREATE TABLE security_master (
    asset_id        uuid PRIMARY KEY,
    isin            text UNIQUE,
    company_name    text NOT NULL,
    exchange        text NOT NULL,
    currency        char(3) NOT NULL,
    asset_class     text NOT NULL,
    listing_date    date NOT NULL,
    delisting_date  date,
    created_at      timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT delisting_after_listing
        CHECK (delisting_date IS NULL OR delisting_date >= listing_date)
);

COMMENT ON COLUMN security_master.asset_id IS
  'Permanent. Never a ticker, never reused. Issued once at creation.';

-- ---------------------------------------------------------------------------
-- Identifiers. The ticker -> asset mapping is ITSELF bitemporal.
--
-- This is the table people leave out, and its absence is why backtests silently
-- merge two unrelated companies: a ticker is reassigned, today's mapping is used
-- for a decision made years ago, and one continuous fictional price history is
-- the result. The exclusion constraint makes that state unrepresentable.
-- ---------------------------------------------------------------------------
CREATE TABLE asset_identifiers (
    asset_id   uuid NOT NULL REFERENCES security_master ON DELETE RESTRICT,
    id_type    text NOT NULL CHECK (id_type IN ('ticker','figi','sedol','cusip')),
    id_value   text NOT NULL,
    valid      daterange NOT NULL,

    -- No two assets may claim the same identifier on the same day.
    -- Try it and Postgres refuses the INSERT.
    CONSTRAINT one_owner_per_identifier_per_day
        EXCLUDE USING gist (id_type WITH =, id_value WITH =, valid WITH &&),

    -- The same asset may not hold two overlapping rows for one id_type either.
    CONSTRAINT no_overlapping_history_per_asset
        EXCLUDE USING gist (asset_id WITH =, id_type WITH =, valid WITH &&)
);

CREATE INDEX asset_identifiers_lookup ON asset_identifiers
    USING gist (id_type, id_value, valid);

-- ---------------------------------------------------------------------------
-- Sector / industry, versioned. Reclassifications are real and are an
-- underrated source of bias in sector-neutral backtests, so these are rows with
-- validity ranges rather than columns on security_master.
-- ---------------------------------------------------------------------------
CREATE TABLE asset_classification (
    asset_id  uuid NOT NULL REFERENCES security_master ON DELETE RESTRICT,
    scheme    text NOT NULL,
    sector    text NOT NULL,
    industry  text,
    valid     daterange NOT NULL,

    CONSTRAINT one_classification_per_scheme_per_day
        EXCLUDE USING gist (asset_id WITH =, scheme WITH =, valid WITH &&)
);

-- ---------------------------------------------------------------------------
-- Universe membership as intervals.
-- ---------------------------------------------------------------------------
CREATE TABLE universe_membership (
    universe_id text NOT NULL,
    asset_id    uuid NOT NULL REFERENCES security_master ON DELETE RESTRICT,
    valid       daterange NOT NULL,

    CONSTRAINT no_overlapping_membership
        EXCLUDE USING gist (universe_id WITH =, asset_id WITH =, valid WITH &&)
);

CREATE INDEX universe_membership_lookup ON universe_membership
    USING gist (universe_id, valid);

-- ---------------------------------------------------------------------------
-- Delisting returns.
--
-- final_return is NOT NULL, and that is the entire point of this table.
--
-- A dataset containing delisted names WITHOUT their delisting returns still has
-- survivorship bias, because the loss never registers — it is the subtlest bias
-- in the project and the most expensive to fix late. Making the column
-- non-nullable means you cannot record a delisting while quietly omitting what
-- it cost. On the free data path you often cannot source the number; the correct
-- response is to exclude that name from the backtest universe and record the
-- exclusion, NOT to insert a row with a null.
-- ---------------------------------------------------------------------------
CREATE TABLE delisting_returns (
    asset_id       uuid PRIMARY KEY REFERENCES security_master ON DELETE RESTRICT,
    delisting_date date NOT NULL,
    final_return   numeric NOT NULL,
    source         text NOT NULL,

    CONSTRAINT final_return_is_plausible
        CHECK (final_return >= -1.0 AND final_return <= 10.0)
);

-- Names excluded from the backtest universe because no final return could be
-- sourced. Recording the gap is what keeps it honest — an unquantified upward
-- bias is a usable caveat, silence is not.
CREATE TABLE universe_exclusions (
    asset_id   uuid PRIMARY KEY REFERENCES security_master ON DELETE RESTRICT,
    reason     text NOT NULL,
    noted_on   date NOT NULL DEFAULT CURRENT_DATE
);

-- ---------------------------------------------------------------------------
-- Experiment tracking.
--
-- Not an afterthought and deliberately not MLflow: the deflated Sharpe ratio
-- needs the TOTAL trial count including abandoned variants, which makes this
-- table an input to the significance calculation rather than a convenience.
-- `abandoned` defaults false and is counted anyway — a record that lets you
-- forget your failures is a record that lets you fool yourself.
-- ---------------------------------------------------------------------------
CREATE TABLE experiments (
    experiment_id uuid PRIMARY KEY,
    created_at    timestamptz NOT NULL DEFAULT now(),
    agent_id      text NOT NULL,
    params        jsonb NOT NULL,
    code_sha      text NOT NULL,
    data_asof     date NOT NULL,
    ic_mean       numeric,
    ic_se_naive   numeric,
    ic_se_hac     numeric,
    n_periods     integer,
    abandoned     boolean NOT NULL DEFAULT false,
    notes         text
);

CREATE INDEX experiments_by_agent ON experiments (agent_id, created_at DESC);
