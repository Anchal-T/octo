import json
import logging
import math
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from hashlib import sha256
from typing import Any

import httpx
from agentteam_common import Settings, stable_id
from agentteam_contracts import ModelCallRequest, ModelOutcome
from pydantic import ValidationError

from .ledger import Ledger, Reservation
from .pools import Pin, Pool, pin_for, pool_for
from .store import Store

logger = logging.getLogger(__name__)


def retry_fits(
    *, remaining_s: float, retry_after_s: float, expected_generation_s: float
) -> bool:
    return retry_after_s + expected_generation_s <= remaining_s


class ModelClient:
    def __init__(
        self,
        settings: Settings,
        ledger: Ledger,
        store: Store,
        http: httpx.Client,
        now: Callable[[], datetime],
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        settings.validate_model_config()
        self.settings = settings
        self.ledger = ledger
        self.store = store
        self.http = http
        self.now = now
        self.monotonic = monotonic
        self.sleep = sleep

    def call(self, request: ModelCallRequest) -> ModelOutcome:
        pin = pin_for(request.role, self.settings)
        pool = pool_for(pin, self.settings)
        sampling_json, sampling, payload_sha256, prompt_tokens = _payload_identity(
            request, pin
        )
        request_id = stable_id(
            "mc",
            f"{request.run_id}|{request.step}|{request.role}|{pin.model}|{payload_sha256}",
        )
        overall_started = self._now()
        call_started_mono = self.monotonic()
        tokens_est = prompt_tokens + sampling["max_tokens"]

        if (
            request.sampling.max_tokens is not None
            and request.sampling.max_tokens > pin.max_tokens
        ):
            return self._uncalled_outcome(
                request,
                request_id,
                pin,
                "failed",
                "max_tokens_exceeded",
                "requested max_tokens exceeds the configured role limit",
                overall_started,
            )
        if request.source == "user_feedback" and request.taint >= 2:
            return self._uncalled_outcome(
                request,
                request_id,
                pin,
                "failed",
                "taint_refused",
                "provider is not allowed to receive this taint level",
                overall_started,
            )
        if prompt_tokens > pin.prompt_token_cap or tokens_est > pool.tpm:
            return self._uncalled_outcome(
                request,
                request_id,
                pin,
                "failed",
                "prompt_too_long",
                "estimated prompt exceeds the configured token cap",
                overall_started,
            )

        retry_count = 0
        in_call_retries = 0
        while True:
            started = self._now()
            try:
                reservation = self.ledger.reserve(
                    request_id=request_id,
                    request=request,
                    pin=pin,
                    pool=pool,
                    payload_sha256=payload_sha256,
                    sampling_json=sampling_json,
                    tokens_est=tokens_est,
                    now=started,
                )
            except (sqlite3.Error, OSError, ValidationError) as error:
                return self._ledger_error(
                    request, request_id, pin, error, overall_started
                )

            if reservation.status == "cached":
                return self._cached(reservation.completion, reservation.retry_ordinal)
            if reservation.status == "in_flight":
                return self._in_flight(request, request_id, pin, reservation, started)
            if reservation.status == "quota_exhausted":
                return self._quota_exhausted(
                    request,
                    request_id,
                    pin,
                    reservation,
                    started,
                )

            ordinal = reservation.retry_ordinal
            retry_count = max(retry_count, ordinal - 1)
            try:
                response = self._send(request, pin, sampling, pool, call_started_mono)
            except httpx.HTTPError:
                finished = self._now()
                outcome = self._result(
                    "failed",
                    request,
                    request_id,
                    pin,
                    retry_count=retry_count,
                    started=overall_started,
                    finished=finished,
                    http_status=None,
                    error_code="provider_unavailable",
                    reason="provider request failed",
                )
                self._finish(
                    outcome,
                    ordinal,
                    "reserved",
                    decision="failed",
                    started=started,
                    finished=finished,
                )
                return outcome

            status = response.status_code
            finished = self._now()
            body = _json_object(response)
            quota = _provider_quota(body, response, self._now())
            if quota is not None:
                outcome = self._result(
                    "quota_exhausted",
                    request,
                    request_id,
                    pin,
                    retry_count=retry_count,
                    started=overall_started,
                    finished=finished,
                    retry_at=quota[0],
                    scope=quota[1],
                    http_status=status,
                    error_code="quota_exhausted",
                )
                self._finish(
                    outcome,
                    ordinal,
                    "throttled",
                    decision="quota_exhausted",
                    started=started,
                    finished=finished,
                    retry_at=quota[0],
                    retry_at_source=quota[2],
                    scope=quota[1],
                )
                return outcome

            if status in (429, 503):
                retry_after_raw = response.headers.get("retry-after")
                retry_after = _retry_after_seconds(retry_after_raw, finished)
                if retry_after is None:
                    retry_after = _body_retry_after(body, finished)
                if retry_after is None:
                    retry_after = 5.0
                elapsed = max(0.0, self.monotonic() - call_started_mono)
                remaining = max(0.0, pin.timeout_s - elapsed)
                can_retry = in_call_retries == 0 and retry_fits(
                    remaining_s=remaining,
                    retry_after_s=retry_after,
                    expected_generation_s=pin.expected_generation_s,
                )
                if can_retry:
                    throttled = self._result(
                        "throttled",
                        request,
                        request_id,
                        pin,
                        retry_count=retry_count + 1,
                        started=overall_started,
                        finished=finished,
                        retry_in=retry_after,
                        http_status=status,
                    )
                    self._finish(
                        throttled,
                        ordinal,
                        "throttled",
                        decision="retry_in_call",
                        started=started,
                        finished=finished,
                        retry_after_raw=retry_after_raw,
                        retry_in=retry_after,
                        wait_s=retry_after,
                    )
                    self.sleep(retry_after)
                    in_call_retries += 1
                    continue

                outcome = self._result(
                    "throttled",
                    request,
                    request_id,
                    pin,
                    retry_count=retry_count,
                    started=overall_started,
                    finished=finished,
                    retry_in=retry_after,
                    http_status=status,
                )
                self._finish(
                    outcome,
                    ordinal,
                    "throttled",
                    decision="throttled",
                    started=started,
                    finished=finished,
                    retry_after_raw=retry_after_raw,
                    retry_in=retry_after,
                )
                return outcome

            if status < 200 or status >= 300:
                outcome = self._result(
                    "failed",
                    request,
                    request_id,
                    pin,
                    retry_count=retry_count,
                    started=overall_started,
                    finished=finished,
                    http_status=status,
                    error_code=_provider_error_code(status),
                    reason="provider rejected the model call",
                )
                state = (
                    "refused"
                    if 400 <= status < 500 and status not in (408, 429)
                    else "failed"
                )
                self._finish(
                    outcome,
                    ordinal,
                    state,
                    decision="failed",
                    started=started,
                    finished=finished,
                    http_status=status,
                    error_code=outcome.error_code,
                )
                return outcome

            completed = _completion_fields(body)
            if completed is None:
                outcome = self._result(
                    "failed",
                    request,
                    request_id,
                    pin,
                    retry_count=retry_count,
                    started=overall_started,
                    finished=finished,
                    http_status=status,
                    error_code="invalid_provider_response",
                    reason="provider response did not contain a completion",
                )
                self._finish(
                    outcome,
                    ordinal,
                    "failed",
                    decision="failed",
                    started=started,
                    finished=finished,
                    http_status=status,
                    error_code=outcome.error_code,
                )
                return outcome

            model_reported, text, tool_calls, tokens_in, tokens_out = completed
            if model_reported != pin.model:
                outcome = self._result(
                    "failed",
                    request,
                    request_id,
                    pin,
                    retry_count=retry_count,
                    started=overall_started,
                    finished=finished,
                    model_reported=model_reported,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    http_status=status,
                    error_code="model_mismatch",
                    reason="provider served a model other than the configured pin",
                )
                self._finish(
                    outcome,
                    ordinal,
                    "failed",
                    decision="failed",
                    started=started,
                    finished=finished,
                    http_status=status,
                    error_code="model_mismatch",
                    model_reported=model_reported,
                )
                return outcome

            outcome = self._result(
                "completed",
                request,
                request_id,
                pin,
                retry_count=retry_count,
                started=overall_started,
                finished=finished,
                model_reported=model_reported,
                text=text,
                tool_calls=tool_calls,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                http_status=status,
            )
            try:
                self.store.complete_attempt(
                    outcome=outcome,
                    payload_sha256=payload_sha256,
                    sampling_json=sampling_json,
                    retry_ordinal=ordinal,
                    log_row=self._log_row(
                        outcome,
                        decision="completed",
                        retry_ordinal=ordinal,
                        started=started,
                        finished=finished,
                        http_status=status,
                    ),
                )
            except (sqlite3.Error, OSError, ValidationError) as error:
                return self._ledger_error(
                    request, request_id, pin, error, overall_started, ordinal
                )
            self._log(outcome, decision="completed", retry_ordinal=ordinal)
            return outcome

    def _send(
        self,
        request: ModelCallRequest,
        pin: Pin,
        sampling: dict[str, Any],
        pool: Pool,
        call_started_mono: float,
    ) -> httpx.Response:
        remaining = max(0.1, pin.timeout_s - (self.monotonic() - call_started_mono))
        body = {
            "model": pin.model,
            "messages": request.messages,
            **sampling,
        }
        return self.http.post(
            f"{pool.base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.settings.groq_api_key}"},
            timeout=remaining,
        )

    def _uncalled_outcome(
        self,
        request: ModelCallRequest,
        request_id: str,
        pin: Pin,
        outcome_name: str,
        error_code: str,
        reason: str,
        decided_at: datetime,
    ) -> ModelOutcome:
        outcome = self._result(
            outcome_name,
            request,
            request_id,
            pin,
            started=None,
            finished=None,
            decided_at=decided_at,
            error_code=error_code,
            reason=reason,
        )
        self._record_without_attempt(outcome, decision="failed")
        return outcome

    def _cached(
        self, completion: ModelOutcome | None, retry_ordinal: int
    ) -> ModelOutcome:
        if completion is None:
            raise RuntimeError("cached reservation is missing its completion")
        self._record_without_attempt(
            completion,
            decision="stored",
            retry_ordinal=max(retry_ordinal, completion.retry_count + 1),
        )
        return completion

    def _in_flight(
        self,
        request: ModelCallRequest,
        request_id: str,
        pin: Pin,
        reservation: Reservation,
        decided_at: datetime,
    ) -> ModelOutcome:
        outcome = self._result(
            "throttled",
            request,
            request_id,
            pin,
            retry_in=reservation.retry_in or 5.0,
            retry_count=max(0, reservation.retry_ordinal - 1),
            decided_at=decided_at,
        )
        self._record_without_attempt(
            outcome,
            decision="in_flight",
            retry_ordinal=reservation.retry_ordinal,
        )
        return outcome

    def _quota_exhausted(
        self,
        request: ModelCallRequest,
        request_id: str,
        pin: Pin,
        reservation: Reservation,
        decided_at: datetime,
    ) -> ModelOutcome:
        outcome = self._result(
            "quota_exhausted",
            request,
            request_id,
            pin,
            retry_at=reservation.retry_at,
            scope=reservation.scope,
            retry_count=max(0, reservation.retry_ordinal - 1),
            decided_at=decided_at,
            error_code="quota_exhausted",
        )
        self._record_without_attempt(
            outcome,
            decision="quota_exhausted",
            retry_at_source="config",
            retry_at=reservation.retry_at,
            scope=reservation.scope,
        )
        return outcome

    def _ledger_error(
        self,
        request: ModelCallRequest,
        request_id: str,
        pin: Pin,
        error: Exception,
        started: datetime,
        retry_ordinal: int = 0,
    ) -> ModelOutcome:
        decided_at = self._now()
        outcome = self._result(
            "ledger_unavailable",
            request,
            request_id,
            pin,
            started=started,
            finished=decided_at,
            decided_at=decided_at,
            error_code="ledger_unavailable",
            reason="model ledger or completion store is unavailable",
        )
        _emit_log(
            logging.WARNING,
            outcome,
            decision="ledger_unavailable",
            retry_ordinal=retry_ordinal,
            error_type=type(error).__name__,
        )
        return outcome

    def _finish(
        self,
        outcome: ModelOutcome,
        retry_ordinal: int,
        state: str,
        *,
        decision: str,
        started: datetime,
        finished: datetime,
        **extra: Any,
    ) -> None:
        row = self._log_row(
            outcome,
            decision=decision,
            retry_ordinal=retry_ordinal,
            started=started,
            finished=finished,
            **extra,
        )
        try:
            self.store.finish_attempt(
                request_id=outcome.request_id,
                retry_ordinal=retry_ordinal,
                state=state,
                log_row=row,
            )
        except (sqlite3.Error, OSError) as error:
            logger.warning(
                "model_call_store_failure",
                extra={
                    "model_pinned": outcome.model_pinned,
                    "error_type": type(error).__name__,
                },
            )
        self._log(outcome, decision=decision, retry_ordinal=retry_ordinal)

    def _record_without_attempt(
        self, outcome: ModelOutcome, *, decision: str, **extra: Any
    ) -> None:
        row = self._log_row(outcome, decision=decision, **extra)
        try:
            self.store.append_log(row)
        except (sqlite3.Error, OSError) as error:
            logger.warning(
                "model_call_store_failure",
                extra={
                    "model_pinned": outcome.model_pinned,
                    "error_type": type(error).__name__,
                },
            )
        self._log(outcome, decision=decision, retry_ordinal=row["retry_ordinal"])

    def _log_row(
        self,
        outcome: ModelOutcome,
        *,
        decision: str,
        retry_ordinal: int | None = None,
        started: datetime | None = None,
        finished: datetime | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "request_id": outcome.request_id,
            "run_id": outcome.run_id,
            "step": outcome.step,
            "role": "triager",
            "model_pinned": outcome.model_pinned,
            "model_reported": extra.get("model_reported", outcome.model_reported),
            "outcome": outcome.outcome,
            "decision": decision,
            "retry_ordinal": retry_ordinal or outcome.retry_count + 1,
            "http_status": extra.get("http_status", outcome.http_status),
            "retry_after_raw": extra.get("retry_after_raw"),
            "retry_at": _iso(extra.get("retry_at", outcome.retry_at)),
            "retry_in": extra.get("retry_in", outcome.retry_in),
            "retry_at_source": extra.get("retry_at_source"),
            "scope": extra.get("scope", outcome.scope),
            "wait_s": extra.get("wait_s"),
            "started_at": _iso(started or outcome.started_at),
            "finished_at": _iso(finished or outcome.finished_at),
            "decided_at": _iso(outcome.decided_at),
            "error_code": extra.get("error_code", outcome.error_code),
        }

    def _log(self, outcome: ModelOutcome, *, decision: str, retry_ordinal: int) -> None:
        _emit_log(logging.INFO, outcome, decision=decision, retry_ordinal=retry_ordinal)

    def _result(
        self,
        outcome_name: str,
        request: ModelCallRequest,
        request_id: str,
        pin: Pin,
        *,
        started: datetime | None = None,
        finished: datetime | None = None,
        decided_at: datetime | None = None,
        **fields: Any,
    ) -> ModelOutcome:
        return ModelOutcome(
            outcome=outcome_name,
            request_id=request_id,
            decided_at=decided_at or self._now(),
            model_pinned=pin.model,
            run_id=request.run_id,
            step=request.step,
            started_at=started,
            finished_at=finished,
            **fields,
        )

    def _now(self) -> datetime:
        value = self.now()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ModelClient.now must return a timezone-aware datetime")
        return value.astimezone(UTC)


def _payload_identity(
    request: ModelCallRequest, pin: Pin
) -> tuple[str, dict[str, Any], str, int]:
    sampling = request.sampling.model_dump(mode="json", exclude_none=True)
    sampling["max_tokens"] = sampling.get("max_tokens") or pin.max_tokens
    messages_json = json.dumps(request.messages, sort_keys=True, separators=(",", ":"))
    prompt_tokens = math.ceil(len(messages_json.encode("utf-8")) / 4)
    payload_json = json.dumps(
        {"messages": request.messages, "sampling": sampling},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    payload_sha256 = sha256(payload_json.encode("utf-8")).hexdigest()
    sampling_json = json.dumps(sampling, sort_keys=True, separators=(",", ":"))
    return sampling_json, sampling, payload_sha256, prompt_tokens


def _retry_after_seconds(value: str | None, now: datetime) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None or retry_at.utcoffset() is None:
            return None
        return max(0.0, (retry_at - now).total_seconds())


def _body_retry_after(body: dict[str, Any], now: datetime) -> float | None:
    value = body.get("retry_in")
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    retry_at = _parse_datetime(body.get("retry_at"))
    if retry_at is not None:
        return max(0.0, (retry_at - now).total_seconds())
    return None


def _provider_quota(
    body: dict[str, Any], response: httpx.Response, now: datetime
) -> tuple[datetime, str, str] | None:
    error = body.get("error")
    if isinstance(error, dict):
        code = error.get("code")
    else:
        code = error
    if code != "quota_exhausted":
        return None
    scope = body.get("scope")
    if scope not in {
        "minute_requests",
        "minute_tokens",
        "day_requests",
        "day_tokens",
        "week_requests",
    }:
        return None
    retry_at = _parse_datetime(body.get("retry_at"))
    if retry_at is not None:
        return retry_at, scope, "body"
    retry_after = _retry_after_seconds(response.headers.get("retry-after"), now)
    if retry_after is not None:
        return now + timedelta(seconds=retry_after), scope, "header"
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _completion_fields(
    body: dict[str, Any],
) -> tuple[str, str | None, list[dict[str, Any]], int, int] | None:
    model = body.get("model")
    choices = body.get("choices")
    if not isinstance(model, str) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    usage = body.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        tool_calls = []
    return (
        model,
        message.get("content") if isinstance(message.get("content"), str) else None,
        tool_calls,
        max(0, int(usage.get("prompt_tokens") or 0)),
        max(0, int(usage.get("completion_tokens") or 0)),
    )


def _provider_error_code(status: int) -> str:
    return f"provider_http_{status}"


def _log_extra(
    outcome: ModelOutcome, *, decision: str, retry_ordinal: int
) -> dict[str, Any]:
    return {
        "request_id": outcome.request_id,
        "run_id": outcome.run_id,
        "step": outcome.step,
        "model_pinned": outcome.model_pinned,
        "model_reported": outcome.model_reported,
        "outcome": outcome.outcome,
        "decision": decision,
        "retry_ordinal": retry_ordinal,
        "started_at": _iso(outcome.started_at),
        "finished_at": _iso(outcome.finished_at),
        "decided_at": _iso(outcome.decided_at),
        "retry_count": outcome.retry_count,
        "retry_at": _iso(outcome.retry_at),
        "retry_in": outcome.retry_in,
        "http_status": outcome.http_status,
        "scope": outcome.scope,
        "error_code": outcome.error_code,
    }


def _emit_log(
    level: int,
    outcome: ModelOutcome,
    *,
    decision: str,
    retry_ordinal: int,
    **additional: Any,
) -> None:
    fields = {
        **_log_extra(outcome, decision=decision, retry_ordinal=retry_ordinal),
        **additional,
    }
    logger.log(
        level,
        "model_call %s",
        json.dumps(fields, sort_keys=True, separators=(",", ":")),
        extra=fields,
    )


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z") if value else None
