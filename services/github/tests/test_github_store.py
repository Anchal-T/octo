import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from services.github.app.store import IssueRow, SchemaVersionError, Store

NOW = datetime(2026, 11, 6, 12, tzinfo=UTC)


def make_row(
    issue_key="gh-100-7", number=7, *, state="pending_start", admitted_at=None
):
    return IssueRow(
        issue_key=issue_key,
        number=number,
        state=state,
        reason=None,
        deferred_reason=None,
        fingerprint="fp-1",
        github_updated_at=NOW.isoformat(),
        created_at=NOW.isoformat(),
        start_attempts=0,
        admitted_at=admitted_at,
        submitted_at=None,
        finished_at=None,
        result_status=None,
    )


def test_store_initializes_schema_version_and_round_trips_issue(tmp_path):
    store = Store(str(tmp_path / "github.sqlite"))
    row = make_row(admitted_at=NOW.isoformat())
    store.upsert_issue(row)
    store.record_admission(row.issue_key, row.attempt, NOW.isoformat())

    assert store.get_meta("repo_id") is None
    assert store.get_issue(row.issue_key) == row
    assert store.list_issues(state="pending_start") == [row]
    assert store.pending_starts(limit=10) == [row]
    assert store.count_admitted_since(NOW - timedelta(hours=12)) == 1
    with store.connection() as connection:
        assert (
            connection.execute("SELECT version FROM schema_version").fetchone()[0] == 2
        )

    store.set_meta("repo_id", "100")
    assert store.get_meta("repo_id") == "100"


def test_store_refuses_an_unknown_schema_version(tmp_path):
    path = str(tmp_path / "github.sqlite")
    Store(path)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE schema_version SET version = 3")

    with pytest.raises(SchemaVersionError, match="3"):
        Store(path)


def test_store_enforces_checks_and_comment_foreign_key(tmp_path):
    store = Store(str(tmp_path / "github.sqlite"))
    with store.connection() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO issue (issue_key, number, state, fingerprint, "
                "github_updated_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("bad", 1, "unknown", "fp", NOW.isoformat(), NOW.isoformat()),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO comment (issue_key, kind, fingerprint, "
                "github_comment_id, posted_at) VALUES (?, ?, ?, ?, ?)",
                ("missing", "ack", "fp", 1, NOW.isoformat()),
            )


def test_store_comment_deduplication_and_submitted_issues(tmp_path):
    store = Store(str(tmp_path / "github.sqlite"))
    row = make_row(state="submitted")
    store.upsert_issue(row)
    store.record_comment(row.issue_key, row.attempt, "ack", row.fingerprint, 42, NOW)
    store.record_comment(row.issue_key, row.attempt, "ack", row.fingerprint, 99, NOW)

    assert store.comment_exists(row.issue_key, row.attempt, "ack", row.fingerprint)
    assert store.submitted_issues() == [row]
    with store.connection() as connection:
        comment = connection.execute("SELECT github_comment_id FROM comment").fetchone()
    assert comment[0] == 42


def test_store_lease_acquire_expiry_takeover_and_release(tmp_path):
    store = Store(str(tmp_path / "github.sqlite"))

    assert store.acquire_lease("first", NOW, 300)
    assert not store.acquire_lease("second", NOW + timedelta(seconds=1), 300)
    assert store.acquire_lease("second", NOW + timedelta(seconds=301), 300)
    store.release_lease("first")
    assert not store.acquire_lease("third", NOW + timedelta(seconds=302), 300)
    store.release_lease("second")
    assert store.acquire_lease("third", NOW + timedelta(seconds=303), 300)


def test_two_store_instances_share_the_same_lease(tmp_path):
    path = str(tmp_path / "github.sqlite")
    first = Store(path)
    second = Store(path)

    assert first.acquire_lease("first", NOW, 300)
    assert not second.acquire_lease("second", NOW, 300)
    second.release_lease("first")
    assert second.acquire_lease("second", NOW, 300)
