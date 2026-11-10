import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

IssueState = Literal[
    "ignored",
    "skipped",
    "pending_start",
    "submitted",
    "finished",
    "failed",
    "cancelled",
]
CommentKind = Literal["ack", "queued", "skip", "failure", "result"]
ResultStatus = Literal["committed", "quota_exhausted", "throttled"]

_MIGRATE_V1_TO_V2 = (
    (
        "ALTER TABLE issue ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1 "
        "CHECK (attempt BETWEEN 1 AND 99)"
    ),
    "ALTER TABLE issue ADD COLUMN label_seen_at TEXT",
    (
        "CREATE TABLE admission ("
        "issue_key TEXT NOT NULL REFERENCES issue(issue_key), "
        "attempt INTEGER NOT NULL, "
        "admitted_at TEXT NOT NULL, "
        "PRIMARY KEY (issue_key, attempt))"
    ),
    "CREATE INDEX admission_admitted_at ON admission(admitted_at)",
    (
        "INSERT INTO admission SELECT issue_key, 1, admitted_at FROM issue "
        "WHERE admitted_at IS NOT NULL"
    ),
    (
        "CREATE TABLE comment_v2 ("
        "issue_key TEXT NOT NULL REFERENCES issue(issue_key), "
        "attempt INTEGER NOT NULL DEFAULT 1, "
        "kind TEXT NOT NULL CHECK (kind IN "
        "('ack','queued','skip','failure','result')), "
        "fingerprint TEXT NOT NULL, "
        "github_comment_id INTEGER NOT NULL, "
        "posted_at TEXT NOT NULL, "
        "PRIMARY KEY (issue_key, attempt, kind, fingerprint))"
    ),
    (
        "INSERT INTO comment_v2 SELECT issue_key, 1, kind, fingerprint, "
        "github_comment_id, posted_at FROM comment"
    ),
    "DROP TABLE comment",
    "ALTER TABLE comment_v2 RENAME TO comment",
    "UPDATE schema_version SET version = 2",
)


class SchemaVersionError(RuntimeError):
    pass


@dataclass(frozen=True)
class IssueRow:
    issue_key: str
    number: int
    state: IssueState
    reason: str | None
    deferred_reason: str | None
    fingerprint: str
    github_updated_at: str
    created_at: str
    start_attempts: int
    admitted_at: str | None
    submitted_at: str | None
    finished_at: str | None
    result_status: ResultStatus | None
    attempt: int = 1
    label_seen_at: str | None = None


class Store:
    _MIGRATE_V1_TO_V2 = _MIGRATE_V1_TO_V2

    def __init__(self, path: str, busy_timeout_ms: int = 5000) -> None:
        self.path = path
        self.busy_timeout_ms = busy_timeout_ms
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        schema = Path(__file__).with_name("schema.sql").read_text()
        with self.connection() as connection:
            version = self._schema_version(connection)
            if version == 1:
                self._migrate_v1_to_v2(connection)
            elif version is not None and version != 2:
                raise SchemaVersionError(
                    f"unsupported GitHub schema version: {[version]}"
                )
            connection.executescript(schema)
            versions = connection.execute(
                "SELECT version FROM schema_version"
            ).fetchall()
            if not versions:
                connection.execute("INSERT INTO schema_version(version) VALUES (2)")
            elif len(versions) != 1 or versions[0]["version"] != 2:
                value = [row["version"] for row in versions]
                raise SchemaVersionError(f"unsupported GitHub schema version: {value}")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000,
            check_same_thread=True,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def get_meta(self, key: str) -> str | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def upsert_issue(self, issue: IssueRow) -> None:
        columns = (
            "issue_key, number, state, reason, deferred_reason, fingerprint, "
            "github_updated_at, created_at, start_attempts, admitted_at, "
            "submitted_at, finished_at, result_status, attempt, label_seen_at"
        )
        placeholders = ", ".join("?" for _ in range(15))
        values = tuple(issue.__dict__.values())
        with self.connection() as connection:
            connection.execute(
                f"INSERT INTO issue ({columns}) VALUES ({placeholders}) "
                "ON CONFLICT(issue_key) DO UPDATE SET "
                "number=excluded.number, state=excluded.state, reason=excluded.reason, "
                "deferred_reason=excluded.deferred_reason, "
                "fingerprint=excluded.fingerprint, "
                "github_updated_at=excluded.github_updated_at, "
                "created_at=excluded.created_at, "
                "start_attempts=excluded.start_attempts, "
                "admitted_at=excluded.admitted_at, "
                "submitted_at=excluded.submitted_at, "
                "finished_at=excluded.finished_at, "
                "result_status=excluded.result_status, "
                "attempt=excluded.attempt, "
                "label_seen_at=excluded.label_seen_at",
                values,
            )

    def get_issue(self, issue_key: str) -> IssueRow | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM issue WHERE issue_key = ?", (issue_key,)
            ).fetchone()
        return None if row is None else self._issue_row(row)

    def list_issues(self, state: IssueState | None = None) -> list[IssueRow]:
        query = "SELECT * FROM issue"
        values: tuple[str, ...] = ()
        if state is not None:
            query += " WHERE state = ?"
            values = (state,)
        query += " ORDER BY created_at, issue_key"
        with self.connection() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._issue_row(row) for row in rows]

    def pending_starts(self, limit: int) -> list[IssueRow]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM issue WHERE state = 'pending_start' "
                "ORDER BY created_at, issue_key LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._issue_row(row) for row in rows]

    def submitted_issues(self) -> list[IssueRow]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM issue WHERE state = 'submitted' "
                "ORDER BY created_at, issue_key"
            ).fetchall()
        return [self._issue_row(row) for row in rows]

    def count_admitted_since(self, day_start: datetime) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM admission WHERE admitted_at >= ?",
                (self._iso(day_start),),
            ).fetchone()
        return int(row["count"])

    def record_admission(
        self, issue_key: str, attempt: int, admitted_at: datetime | str
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO admission "
                "(issue_key, attempt, admitted_at) VALUES (?, ?, ?)",
                (
                    issue_key,
                    attempt,
                    self._iso(admitted_at)
                    if isinstance(admitted_at, datetime)
                    else admitted_at,
                ),
            )

    def record_comment(
        self,
        issue_key: str,
        attempt: int,
        kind: CommentKind,
        fingerprint: str,
        github_comment_id: int,
        posted_at: datetime | str,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO comment "
                "(issue_key, attempt, kind, fingerprint, github_comment_id, posted_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    issue_key,
                    attempt,
                    kind,
                    fingerprint,
                    github_comment_id,
                    self._iso(posted_at)
                    if isinstance(posted_at, datetime)
                    else posted_at,
                ),
            )

    def comment_exists(
        self, issue_key: str, attempt: int, kind: CommentKind, fingerprint: str
    ) -> bool:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM comment WHERE issue_key = ? AND attempt = ? "
                "AND kind = ? AND fingerprint = ?",
                (issue_key, attempt, kind, fingerprint),
            ).fetchone()
        return row is not None

    def acquire_lease(self, holder: str, now: datetime, ttl_s: int) -> bool:
        if ttl_s <= 0:
            raise ValueError("lease TTL must be positive")
        now_text = self._iso(now)
        expires_at = self._iso(now + timedelta(seconds=ttl_s))
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT holder, expires_at FROM lease WHERE name = 'poller'"
            ).fetchone()
            if (
                row is not None
                and row["expires_at"] > now_text
                and row["holder"] != holder
            ):
                return False
            connection.execute(
                "INSERT INTO lease (name, holder, expires_at) VALUES ('poller', ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET holder=excluded.holder, "
                "expires_at=excluded.expires_at",
                (holder, expires_at),
            )
        return True

    def release_lease(self, holder: str) -> None:
        with self.connection() as connection:
            connection.execute(
                "DELETE FROM lease WHERE name = 'poller' AND holder = ?", (holder,)
            )

    def _migrate_v1_to_v2(self, connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in self._MIGRATE_V1_TO_V2:
                connection.execute(statement)
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    @staticmethod
    def _schema_version(connection: sqlite3.Connection) -> int | None:
        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'schema_version'"
        ).fetchone()
        if row is None:
            return None
        versions = connection.execute("SELECT version FROM schema_version").fetchall()
        if not versions:
            return None
        if len(versions) != 1:
            value = [item["version"] for item in versions]
            raise SchemaVersionError(f"unsupported GitHub schema version: {value}")
        return int(versions[0]["version"])

    @staticmethod
    def _issue_row(row: sqlite3.Row) -> IssueRow:
        return IssueRow(**{key: row[key] for key in IssueRow.__dataclass_fields__})

    @staticmethod
    def _iso(value: datetime) -> str:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC).isoformat()
