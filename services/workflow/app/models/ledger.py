import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from agentteam_common import Settings
from agentteam_contracts import ModelCallRequest, ModelOutcome

from .pools import Pin, Pool
from .store import Store

_COUNTED_STATES_SQL = "('reserved', 'completed', 'throttled', 'failed')"


@dataclass(frozen=True)
class Reservation:
    status: Literal[
        "reserved",
        "cached",
        "in_flight",
        "pool_busy",
        "quota_exhausted",
        "outcome_unknown",
    ]
    retry_ordinal: int = 0
    completion: ModelOutcome | None = None
    retry_at: datetime | None = None
    scope: str | None = None
    retry_in: float | None = None


class Ledger:
    def __init__(self, settings: Settings, store: Store | None = None) -> None:
        self.settings = settings
        self.store = store or Store(settings)
        self.pool_timeout_s = max(
            settings.triager_timeout_s,
            settings.reproducer_timeout_s,
            settings.fixer_timeout_s,
            settings.tester_timeout_s,
        )

    def reserve(
        self,
        *,
        request_id: str,
        request: ModelCallRequest,
        pin: Pin,
        pool: Pool,
        payload_sha256: str,
        sampling_json: str,
        tokens_est: int,
        now: datetime,
    ) -> Reservation:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("quota reservations require a timezone-aware datetime")
        now = now.astimezone(UTC)
        zone = ZoneInfo(pool.reset_tz)
        local_now = now.astimezone(zone)
        minute_start = local_now.replace(second=0, microsecond=0)
        day_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        minute_key = _iso(minute_start.astimezone(UTC))
        day_key = _iso(day_start.astimezone(UTC))

        with self.store.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            completion = connection.execute(
                "SELECT response_json FROM completion WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if completion is not None:
                return Reservation(
                    status="cached",
                    completion=ModelOutcome.model_validate_json(
                        completion["response_json"]
                    ),
                )

            pool_wait = self._recover_pool(connection, pool.pool, now)
            previous = connection.execute(
                "SELECT retry_ordinal, send_phase FROM call_attempt "
                "WHERE request_id = ? AND (state = 'reserved' OR send_phase = 'unknown') "
                "ORDER BY retry_ordinal DESC LIMIT 1",
                (request_id,),
            ).fetchone()
            if previous is not None:
                ordinal = int(previous["retry_ordinal"])
                if previous["send_phase"] == "unknown":
                    return Reservation(status="outcome_unknown", retry_ordinal=ordinal)
                return Reservation(status="in_flight", retry_ordinal=ordinal, retry_in=5.0)

            ordinal_row = connection.execute(
                "SELECT COALESCE(MAX(retry_ordinal), 0) + 1 AS next_ordinal "
                "FROM call_attempt WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            retry_ordinal = int(ordinal_row["next_ordinal"])
            budget_mode = self.settings.models_budget_mode
            other_mode = connection.execute(
                "SELECT 1 FROM call_attempt "
                "WHERE pool = ? AND day_window = ? AND budget_mode != ? "
                f"AND state IN {_COUNTED_STATES_SQL} LIMIT 1",
                (pool.pool, day_key, budget_mode),
            ).fetchone()
            if other_mode is not None:
                # A UTC-day (or configured reset day) must use one accounting policy.
                return Reservation(
                    status="quota_exhausted",
                    retry_ordinal=retry_ordinal,
                    retry_at=(day_start + timedelta(days=1)).astimezone(UTC),
                    scope="day_tokens",
                )
            minute = self._usage(connection, pool.pool, "minute_window", minute_key)
            day = self._usage(connection, pool.pool, "day_window", day_key)
            limits = (
                (
                    "minute_requests",
                    minute[0],
                    1,
                    pool.rpm,
                    minute_start + timedelta(minutes=1),
                ),
                (
                    "minute_tokens",
                    minute[1],
                    tokens_est,
                    pool.tpm,
                    minute_start + timedelta(minutes=1),
                ),
                ("day_requests", day[0], 1, pool.rpd, day_start + timedelta(days=1)),
                (
                    "day_tokens",
                    day[1],
                    tokens_est,
                    pool.tpd,
                    day_start + timedelta(days=1),
                ),
            )
            for scope, used, addition, limit, reset_local in limits:
                if used + addition > limit:
                    return Reservation(
                        status="quota_exhausted",
                        retry_ordinal=retry_ordinal,
                        retry_at=reset_local.astimezone(UTC),
                        scope=scope,
                    )

            if pool_wait:
                return Reservation(
                    status="pool_busy", retry_ordinal=retry_ordinal, retry_in=pool_wait
                )

            connection.execute(
                "INSERT INTO call_attempt (request_id, retry_ordinal, run_id, step, "
                "role, pool, model_pinned, payload_sha256, sampling_json, state, "
                "tokens_est, budget_mode, minute_window, day_window, started_at, target, send_phase) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?, ?, ?, ?, ?, 'pending')",
                (
                    request_id,
                    retry_ordinal,
                    request.run_id,
                    request.step,
                    request.role,
                    pool.pool,
                    pin.model,
                    payload_sha256,
                    sampling_json,
                    tokens_est,
                    budget_mode,
                    minute_key,
                    day_key,
                    _iso(now),
                    request.target,
                ),
            )
            return Reservation(status="reserved", retry_ordinal=retry_ordinal)

    def _recover_pool(
        self, connection: sqlite3.Connection, pool: str, now: datetime
    ) -> float:
        # The caller holds the host lock, so a live sender keeps its slot even
        # if the wall-clock lease would have elapsed.
        active = connection.execute(
            "SELECT request_id, retry_ordinal, started_at, send_phase "
            "FROM call_attempt WHERE pool = ? AND state = 'reserved'",
            (pool,),
        ).fetchall()
        wait_s = 0.0
        for attempt in active:
            expires_at = datetime.fromisoformat(attempt["started_at"]) + timedelta(
                seconds=self.pool_timeout_s + 60
            )
            if now < expires_at:
                wait_s = max(wait_s, 5.0, (expires_at - now).total_seconds())
                continue

            # Legacy reservations without a marker may also have been sent.
            never_sent = attempt["send_phase"] == "pending"
            connection.execute(
                "UPDATE call_attempt SET state = ?, send_phase = ?, decided_at = ? "
                "WHERE request_id = ? AND retry_ordinal = ?",
                (
                    "refused" if never_sent else "failed",
                    "pending" if never_sent else "unknown",
                    _iso(now),
                    attempt["request_id"],
                    attempt["retry_ordinal"],
                ),
            )
        return wait_s

    @staticmethod
    def _usage(
        connection: sqlite3.Connection, pool: str, window_column: str, window_key: str
    ) -> tuple[int, int]:
        if window_column not in {"minute_window", "day_window"}:
            raise ValueError("unsupported quota window")
        row = connection.execute(
            "SELECT COUNT(*) AS requests, COALESCE(SUM(COALESCE(tokens_accounted, tokens_est)), 0) AS tokens "
            f"FROM call_attempt WHERE pool = ? AND {window_column} = ? "
            f"AND state IN {_COUNTED_STATES_SQL}",
            (pool, window_key),
        ).fetchone()
        return int(row["requests"]), int(row["tokens"])


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
