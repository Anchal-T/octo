CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

INSERT INTO schema_version(version)
SELECT 1
WHERE NOT EXISTS (SELECT 1 FROM schema_version);

CREATE TABLE IF NOT EXISTS call_attempt (
    request_id TEXT NOT NULL,
    retry_ordinal INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    step TEXT NOT NULL,
    role TEXT NOT NULL,
    pool TEXT NOT NULL,
    model_pinned TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    sampling_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('reserved', 'completed', 'throttled', 'failed', 'refused')),
    tokens_est INTEGER NOT NULL,
    minute_window TEXT NOT NULL,
    day_window TEXT NOT NULL,
    started_at TEXT NOT NULL,
    decided_at TEXT,
    PRIMARY KEY (request_id, retry_ordinal)
);

CREATE INDEX IF NOT EXISTS call_attempt_request_state
    ON call_attempt(request_id, state);
CREATE INDEX IF NOT EXISTS call_attempt_minute_usage
    ON call_attempt(pool, minute_window, state);
CREATE INDEX IF NOT EXISTS call_attempt_day_usage
    ON call_attempt(pool, day_window, state);

CREATE TABLE IF NOT EXISTS completion (
    request_id TEXT PRIMARY KEY,
    model_pinned TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    sampling_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    model_reported TEXT NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS call_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    step TEXT NOT NULL,
    role TEXT NOT NULL,
    model_pinned TEXT NOT NULL,
    model_reported TEXT,
    outcome TEXT NOT NULL,
    decision TEXT NOT NULL,
    retry_ordinal INTEGER NOT NULL,
    http_status INTEGER,
    retry_after_raw TEXT,
    retry_at TEXT,
    retry_in REAL,
    retry_at_source TEXT,
    scope TEXT,
    wait_s REAL,
    started_at TEXT,
    finished_at TEXT,
    decided_at TEXT NOT NULL,
    error_code TEXT
);
