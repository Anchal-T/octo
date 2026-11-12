import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from services.github.app.store import IssueRow, SchemaVersionError, Store

NOW = datetime(2026, 11, 6, 12, tzinfo=UTC)

V1_SCHEMA = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE issue (
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
CREATE TABLE comment (
    issue_key TEXT NOT NULL REFERENCES issue(issue_key),
    kind TEXT NOT NULL CHECK (kind IN ('ack','queued','skip','failure','result')),
    fingerprint TEXT NOT NULL,
    github_comment_id INTEGER NOT NULL,
    posted_at TEXT NOT NULL,
    PRIMARY KEY (issue_key, kind, fingerprint)
);
CREATE TABLE lease (
    name TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
INSERT INTO schema_version(version) VALUES (1);
"""


def _seed_v1(path: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(V1_SCHEMA)
        connection.execute(
            "INSERT INTO issue (issue_key, number, state, fingerprint, "
            "github_updated_at, created_at, admitted_at, submitted_at, finished_at, "
            "result_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "gh-100-7",
                7,
                "finished",
                "fp-1",
                NOW.isoformat(),
                NOW.isoformat(),
                NOW.isoformat(),
                NOW.isoformat(),
                NOW.isoformat(),
                "committed",
            ),
        )
        connection.execute(
            "INSERT INTO comment (issue_key, kind, fingerprint, github_comment_id, "
            "posted_at) VALUES (?, ?, ?, ?, ?)",
            ("gh-100-7", "ack", "fp-1", 11, NOW.isoformat()),
        )
        connection.execute(
            "INSERT INTO comment (issue_key, kind, fingerprint, github_comment_id, "
            "posted_at) VALUES (?, ?, ?, ?, ?)",
            ("gh-100-7", "result", "fp-1", 12, NOW.isoformat()),
        )
        connection.commit()


def test_v1_fixture_migrates_to_v2_with_admission_and_attempted_comments(tmp_path):
    path = str(tmp_path / "github.sqlite")
    _seed_v1(path)

    store = Store(path)
    row = store.get_issue("gh-100-7")

    assert row is not None
    assert row.attempt == 1
    assert row.label_seen_at is None
    assert row.state == "finished"
    assert store.comment_exists("gh-100-7", 1, "ack", "fp-1")
    assert store.comment_exists("gh-100-7", 1, "result", "fp-1")
    assert store.count_admitted_since(NOW - timedelta(hours=1)) == 1
    with store.connection() as connection:
        assert (
            connection.execute("SELECT version FROM schema_version").fetchone()[0] == 3
        )
        admission = connection.execute(
            "SELECT issue_key, attempt, admitted_at FROM admission"
        ).fetchall()
        comments = connection.execute(
            "SELECT issue_key, attempt, kind FROM comment ORDER BY kind"
        ).fetchall()
    assert [
        (row["issue_key"], row["attempt"], row["admitted_at"]) for row in admission
    ] == [("gh-100-7", 1, NOW.isoformat())]
    assert [(row["issue_key"], row["attempt"], row["kind"]) for row in comments] == [
        ("gh-100-7", 1, "ack"),
        ("gh-100-7", 1, "result"),
    ]


def test_migrate_v1_to_v2_commits_before_returning(tmp_path):
    path = str(tmp_path / "github.sqlite")
    _seed_v1(path)
    store = Store.__new__(Store)
    store.path = path
    store.busy_timeout_ms = 5000

    connection = store.connect()
    try:
        assert Store._schema_version(connection) == 1
        store._migrate_v1_to_v2(connection)
    finally:
        connection.close()

    with sqlite3.connect(path) as other:
        version = other.execute("SELECT version FROM schema_version").fetchone()[0]
        columns = {
            row[1] for row in other.execute("PRAGMA table_info(issue)").fetchall()
        }
        admission = other.execute("SELECT COUNT(*) FROM admission").fetchone()[0]
    assert version == 2
    assert "attempt" in columns
    assert "label_seen_at" in columns
    assert admission == 1


def test_migration_failure_leaves_version_1(tmp_path, monkeypatch):
    path = str(tmp_path / "github.sqlite")
    _seed_v1(path)
    monkeypatch.setattr(
        Store,
        "_MIGRATE_V1_TO_V2",
        (
            (
                "ALTER TABLE issue ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1 "
                "CHECK (attempt BETWEEN 1 AND 99)"
            ),
            "THIS IS NOT VALID SQL",
        ),
    )

    with pytest.raises(sqlite3.OperationalError):
        Store(path)

    with sqlite3.connect(path) as connection:
        version = connection.execute("SELECT version FROM schema_version").fetchone()[0]
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(issue)").fetchall()
        }
    assert version == 1
    assert "attempt" not in columns


def test_two_attempts_on_one_day_count_as_two(tmp_path):
    store = Store(str(tmp_path / "github.sqlite"))
    first = IssueRow(
        issue_key="gh-100-7",
        number=7,
        state="finished",
        reason=None,
        deferred_reason=None,
        fingerprint="fp-1",
        github_updated_at=NOW.isoformat(),
        created_at=NOW.isoformat(),
        start_attempts=0,
        admitted_at=NOW.isoformat(),
        submitted_at=NOW.isoformat(),
        finished_at=NOW.isoformat(),
        result_status="committed",
        attempt=1,
    )
    store.upsert_issue(first)
    store.record_admission("gh-100-7", 1, NOW.isoformat())
    store.upsert_issue(
        IssueRow(
            issue_key="gh-100-7",
            number=7,
            state="pending_start",
            reason=None,
            deferred_reason=None,
            fingerprint="fp-2",
            github_updated_at=NOW.isoformat(),
            created_at=NOW.isoformat(),
            start_attempts=0,
            admitted_at=(NOW + timedelta(minutes=1)).isoformat(),
            submitted_at=None,
            finished_at=None,
            result_status=None,
            attempt=2,
        )
    )
    store.record_admission("gh-100-7", 2, (NOW + timedelta(minutes=1)).isoformat())

    assert store.count_admitted_since(NOW.replace(hour=0, minute=0, second=0)) == 2


def test_store_refuses_schema_version_four(tmp_path):
    path = str(tmp_path / "github.sqlite")
    Store(path)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE schema_version SET version = 4")

    with pytest.raises(SchemaVersionError, match="4"):
        Store(path)


def test_v2_migration_failure_rolls_back_and_leaves_version_2(tmp_path, monkeypatch):
    path = str(tmp_path / "github.sqlite")
    _seed_v1(path)
    store = Store.__new__(Store)
    store.path = path
    store.busy_timeout_ms = 5000
    conn = store.connect()
    try:
        store._migrate_v1_to_v2(conn)
    finally:
        conn.close()

    monkeypatch.setattr(
        Store,
        "_MIGRATE_V2_TO_V3",
        ("CREATE TABLE pull_request (pr_number INTEGER PRIMARY KEY)", "INVALID SQL"),
    )
    with pytest.raises(sqlite3.OperationalError):
        Store(path)

    with sqlite3.connect(path) as check:
        version = check.execute("SELECT version FROM schema_version").fetchone()[0]
        tables = {
            row[0]
            for row in check.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert version == 2
    assert "pull_request" not in tables
