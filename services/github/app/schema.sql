CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS issue (
    issue_key TEXT PRIMARY KEY,
    number INTEGER NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (state IN
        ('ignored','skipped','pending_start','submitted','finished','failed','cancelled')),
    reason TEXT,
    deferred_reason TEXT,
    fingerprint TEXT NOT NULL,
    github_updated_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    start_attempts INTEGER NOT NULL DEFAULT 0 CHECK (start_attempts >= 0),
    admitted_at TEXT,
    submitted_at TEXT,
    finished_at TEXT,
    result_status TEXT CHECK (result_status IN
        ('committed','quota_exhausted','throttled','approved','rejected','failed')),
    attempt INTEGER NOT NULL DEFAULT 1 CHECK (attempt BETWEEN 1 AND 99),
    label_seen_at TEXT
);
CREATE INDEX IF NOT EXISTS issue_state_created ON issue(state, created_at);
CREATE INDEX IF NOT EXISTS issue_admitted_at ON issue(admitted_at);

CREATE TABLE IF NOT EXISTS admission (
    issue_key TEXT NOT NULL REFERENCES issue(issue_key),
    attempt INTEGER NOT NULL,
    admitted_at TEXT NOT NULL,
    PRIMARY KEY (issue_key, attempt)
);
CREATE INDEX IF NOT EXISTS admission_admitted_at ON admission(admitted_at);

CREATE TABLE IF NOT EXISTS comment (
    issue_key TEXT NOT NULL REFERENCES issue(issue_key),
    attempt INTEGER NOT NULL DEFAULT 1,
    kind TEXT NOT NULL CHECK (kind IN ('ack','queued','skip','failure','result')),
    fingerprint TEXT NOT NULL,
    github_comment_id INTEGER NOT NULL,
    posted_at TEXT NOT NULL,
    PRIMARY KEY (issue_key, attempt, kind, fingerprint)
);

CREATE TABLE IF NOT EXISTS pull_request (
    pr_number INTEGER PRIMARY KEY,
    issue_key TEXT NOT NULL,
    attempt INTEGER NOT NULL CHECK (attempt BETWEEN 1 AND 99),
    branch TEXT NOT NULL,
    base_branch TEXT NOT NULL,
    head_sha TEXT NOT NULL,
    html_url TEXT NOT NULL,
    author_login TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('open','closed','merged')),
    review_requested_sha TEXT,
    decision TEXT NOT NULL DEFAULT 'pending' CHECK (decision IN ('pending','approved','rejected')),
    decided_by TEXT,
    decision_reason TEXT,
    decided_review_id INTEGER,
    decided_head_sha TEXT,
    decided_at TEXT,
    delivered_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS pull_request_state ON pull_request(state, pr_number);
CREATE INDEX IF NOT EXISTS pull_request_issue_attempt ON pull_request(issue_key, attempt);

CREATE TABLE IF NOT EXISTS write_log (
    idempotency_key TEXT PRIMARY KEY,
    endpoint TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lease (
    name TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
