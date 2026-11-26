import fcntl
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


_RECEIPT_VIEW = """
CREATE VIEW IF NOT EXISTS side_effect_receipt AS
SELECT
    attempt.request_id || ':' || attempt.retry_ordinal AS idempotency_key,
    'groq.chat_completion' AS action,
    COALESCE(attempt.target, attempt.run_id) AS target,
    attempt.payload_sha256 AS request_sha256,
    CASE WHEN attempt.send_phase = 'unknown' THEN 'outcome_unknown'
         WHEN attempt.state = 'reserved' AND attempt.send_phase = 'sent' THEN 'sent'
         ELSE attempt.state END AS outcome,
    COALESCE(attempt.decided_at, attempt.started_at) AS decided_at,
    completion.response_json
FROM call_attempt AS attempt
LEFT JOIN completion
    ON completion.request_id = attempt.request_id AND attempt.state = 'completed';
"""


class Store:
    def __init__(self, settings: Settings) -> None:
        self.path = settings.models_sqlite_path
        ledger_path = Path(self.path).resolve()
        self._provider_lock_path = ledger_path.with_name(
            f"{ledger_path.name}.provider.lock"
        )
        self.busy_timeout_ms = settings.models_busy_timeout_ms
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        schema = Path(__file__).with_name("schema.sql").read_text()
        with self.connection() as connection:
            connection.executescript(schema)
            connection.execute("BEGIN IMMEDIATE")
            columns = {
                column["name"]
                for column in connection.execute("PRAGMA table_info(call_attempt)")
            }
            if "target" not in columns:
                connection.execute("ALTER TABLE call_attempt ADD COLUMN target TEXT")
            if "send_phase" not in columns:
                connection.execute("ALTER TABLE call_attempt ADD COLUMN send_phase TEXT")
                connection.execute("DROP VIEW IF EXISTS side_effect_receipt")
            if "tokens_accounted" not in columns:
                connection.execute("ALTER TABLE call_attempt ADD COLUMN tokens_accounted INTEGER")
            if "failure_json" not in columns:
                connection.execute("ALTER TABLE call_attempt ADD COLUMN failure_json TEXT")
            if "budget_mode" not in columns:
                connection.execute(
                    "ALTER TABLE call_attempt ADD COLUMN budget_mode TEXT NOT NULL DEFAULT 'p1_estimate'"
                )
            connection.execute(_RECEIPT_VIEW)

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

    @contextmanager
    def try_provider_slot(self) -> Iterator[bool]:
        # Closing the descriptor releases the slot, including on process exit.
        # Workers must share both the ledger and this lock file on the same host.
        with self._provider_lock_path.open("ab") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
            else:
                yield True

    @staticmethod
    def recorded_outcome(
        connection: sqlite3.Connection, request_id: str, *, retry_refused: bool
    ) -> ModelOutcome | None:
        completion = connection.execute(
            "SELECT response_json FROM completion WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if completion is not None:
            return ModelOutcome.model_validate_json(completion["response_json"])

        failure = connection.execute(
            "SELECT retry_ordinal, model_pinned, run_id, step, started_at, decided_at, "
            "failure_json FROM call_attempt WHERE request_id = ? "
            "AND (state = 'failed' OR (state = 'refused' AND NOT ?)) "
            "AND (failure_json IS NOT NULL OR COALESCE(send_phase, 'sent') = 'sent') "
            "ORDER BY retry_ordinal DESC LIMIT 1",
            (request_id, retry_refused),
        ).fetchone()
        if failure is None:
            return None
        if failure["failure_json"] is not None:
            return ModelOutcome.model_validate_json(failure["failure_json"])

        # Older ledgers have the terminal status and log, but no serialized outcome.
        log = connection.execute(
            "SELECT error_code, http_status, model_reported, finished_at FROM call_log "
            "WHERE request_id = ? AND retry_ordinal = ? AND decision = 'failed' "
            "ORDER BY seq DESC LIMIT 1",
            (request_id, failure["retry_ordinal"]),
        ).fetchone()
        return ModelOutcome.model_validate(
            {
                **(dict(log) if log is not None else {}),
                "outcome": "failed",
                "request_id": request_id,
                "model_pinned": failure["model_pinned"],
                "run_id": failure["run_id"],
                "step": failure["step"],
                "retry_count": failure["retry_ordinal"] - 1,
                "started_at": failure["started_at"],
                "decided_at": failure["decided_at"] or failure["started_at"],
                "reason": "previous provider call was recorded as failed",
            }
        )

    def append_log(self, row: dict[str, Any]) -> None:
        placeholders = ", ".join("?" for _ in LOG_COLUMNS)
        columns = ", ".join(LOG_COLUMNS)
        values = [row.get(column) for column in LOG_COLUMNS]
        with self.connection() as connection:
            connection.execute(
                f"INSERT INTO call_log ({columns}) VALUES ({placeholders})", values
            )

    def mark_sent(self, *, request_id: str, retry_ordinal: int) -> bool:
        with self.connection() as connection:
            result = connection.execute(
                "UPDATE call_attempt SET send_phase = 'sent' "
                "WHERE request_id = ? AND retry_ordinal = ? "
                "AND state = 'reserved' AND send_phase = 'pending'",
                (request_id, retry_ordinal),
            )
            return result.rowcount == 1

    def finish_attempt(
        self,
        *,
        outcome: ModelOutcome,
        retry_ordinal: int,
        state: str,
        log_row: dict[str, Any],
    ) -> None:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE call_attempt SET state = ?, decided_at = ?, "
                "send_phase = CASE WHEN ? THEN 'unknown' ELSE 'sent' END, failure_json = ? "
                "WHERE request_id = ? AND retry_ordinal = ?",
                (
                    state,
                    log_row["decided_at"],
                    outcome.error_code == "provider_outcome_unknown",
                    outcome.model_dump_json() if outcome.outcome == "failed" else None,
                    outcome.request_id,
                    retry_ordinal,
                ),
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
        tokens_accounted: int | None,
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
                "UPDATE call_attempt SET state = 'completed', send_phase = 'sent', "
                "decided_at = ?, tokens_accounted = ? WHERE request_id = ? AND retry_ordinal = ?",
                (_iso(outcome.decided_at), tokens_accounted, outcome.request_id, retry_ordinal),
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
