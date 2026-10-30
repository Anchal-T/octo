import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agentteam_common import Settings
from agentteam_contracts import ModelOutcome

LOG_COLUMNS = (
    "request_id",
    "run_id",
    "step",
    "role",
    "model_pinned",
    "model_reported",
    "outcome",
    "decision",
    "retry_ordinal",
    "http_status",
    "retry_after_raw",
    "retry_at",
    "retry_in",
    "retry_at_source",
    "scope",
    "wait_s",
    "started_at",
    "finished_at",
    "decided_at",
    "error_code",
)


class Store:
    def __init__(self, settings: Settings) -> None:
        self.path = settings.models_sqlite_path
        self.busy_timeout_ms = settings.models_busy_timeout_ms
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        schema = Path(__file__).with_name("schema.sql").read_text()
        with self.connection() as connection:
            connection.executescript(schema)

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
        return connection

    def get_completion(self, request_id: str) -> ModelOutcome | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT response_json FROM completion WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        return ModelOutcome.model_validate_json(row["response_json"])

    def append_log(self, row: dict[str, Any]) -> None:
        placeholders = ", ".join("?" for _ in LOG_COLUMNS)
        columns = ", ".join(LOG_COLUMNS)
        values = [row.get(column) for column in LOG_COLUMNS]
        with self.connection() as connection:
            connection.execute(
                f"INSERT INTO call_log ({columns}) VALUES ({placeholders})", values
            )

    def finish_attempt(
        self,
        *,
        request_id: str,
        retry_ordinal: int,
        state: str,
        log_row: dict[str, Any],
    ) -> None:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE call_attempt SET state = ?, decided_at = ? "
                "WHERE request_id = ? AND retry_ordinal = ?",
                (state, log_row["decided_at"], request_id, retry_ordinal),
            )
            self._insert_log(connection, log_row)

    def complete_attempt(
        self,
        *,
        outcome: ModelOutcome,
        payload_sha256: str,
        sampling_json: str,
        retry_ordinal: int,
        log_row: dict[str, Any],
    ) -> None:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO completion (request_id, model_pinned, payload_sha256, "
                "sampling_json, response_json, model_reported, tokens_in, tokens_out, "
                "started_at, finished_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    outcome.request_id,
                    outcome.model_pinned,
                    payload_sha256,
                    sampling_json,
                    outcome.model_dump_json(),
                    outcome.model_reported or outcome.model_pinned,
                    outcome.tokens_in,
                    outcome.tokens_out,
                    _iso(outcome.started_at),
                    _iso(outcome.finished_at),
                ),
            )
            connection.execute(
                "UPDATE call_attempt SET state = 'completed', decided_at = ? "
                "WHERE request_id = ? AND retry_ordinal = ?",
                (_iso(outcome.decided_at), outcome.request_id, retry_ordinal),
            )
            self._insert_log(connection, log_row)

    @staticmethod
    def _insert_log(connection: sqlite3.Connection, row: dict[str, Any]) -> None:
        placeholders = ", ".join("?" for _ in LOG_COLUMNS)
        columns = ", ".join(LOG_COLUMNS)
        connection.execute(
            f"INSERT INTO call_log ({columns}) VALUES ({placeholders})",
            [row.get(column) for column in LOG_COLUMNS],
        )


def _iso(value: Any) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None
