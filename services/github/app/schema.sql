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
    result_status TEXT CHECK (result_status IN ('committed','quota_exhausted','throttled'))
);
CREATE INDEX IF NOT EXISTS issue_state_created ON issue(state, created_at);
CREATE INDEX IF NOT EXISTS issue_admitted_at ON issue(admitted_at);

CREATE TABLE IF NOT EXISTS comment (
    issue_key TEXT NOT NULL REFERENCES issue(issue_key),
    kind TEXT NOT NULL CHECK (kind IN ('ack','queued','skip','failure','result')),
    fingerprint TEXT NOT NULL,
    github_comment_id INTEGER NOT NULL,
    posted_at TEXT NOT NULL,
    PRIMARY KEY (issue_key, kind, fingerprint)
);

CREATE TABLE IF NOT EXISTS lease (
    name TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
