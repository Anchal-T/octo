import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from agentteam_common import Settings
from agentteam_contracts import ModelCallRequest, ModelOutcome

from .pools import Pin, Pool
from .store import Store


@dataclass(frozen=True)
class Reservation:
    status: Literal["reserved", "cached", "in_flight", "quota_exhausted"]
    retry_ordinal: int = 0
    completion: ModelOutcome | None = None
    retry_at: datetime | None = None
    scope: str | None = None
    retry_in: float | None = None


class Ledger:
    def __init__(self, settings: Settings, store: Store | None = None) -> None:
        self.settings = settings
        self.store = store or Store(settings)

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

            reserved = connection.execute(
                "SELECT retry_ordinal FROM call_attempt "
                "WHERE request_id = ? AND state = 'reserved' "
                "ORDER BY retry_ordinal DESC LIMIT 1",
                (request_id,),
            ).fetchone()
            if reserved is not None:
                return Reservation(
                    status="in_flight",
                    retry_ordinal=int(reserved["retry_ordinal"]),
                    retry_in=5.0,
                )

            ordinal_row = connection.execute(
                "SELECT COALESCE(MAX(retry_ordinal), 0) + 1 AS next_ordinal "
                "FROM call_attempt WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            retry_ordinal = int(ordinal_row["next_ordinal"])
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

            connection.execute(
                "INSERT INTO call_attempt (request_id, retry_ordinal, run_id, step, "
                "role, pool, model_pinned, payload_sha256, sampling_json, state, "
                "tokens_est, minute_window, day_window, started_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?, ?, ?)",
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
                    minute_key,
                    day_key,
                    _iso(now),
                ),
            )
            return Reservation(status="reserved", retry_ordinal=retry_ordinal)

    @staticmethod
    def _usage(
        connection: sqlite3.Connection, pool: str, window_column: str, window_key: str
    ) -> tuple[int, int]:
        if window_column not in {"minute_window", "day_window"}:
            raise ValueError("unsupported quota window")
        row = connection.execute(
            f"SELECT COUNT(*) AS requests, COALESCE(SUM(tokens_est), 0) AS tokens "
            f"FROM call_attempt WHERE pool = ? AND {window_column} = ? "
            "AND state IN ('reserved', 'completed', 'throttled', 'failed')",
            (pool, window_key),
        ).fetchone()
        return int(row["requests"]), int(row["tokens"])


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
