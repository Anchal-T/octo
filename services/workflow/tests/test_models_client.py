import json
import logging
from dataclasses import replace
from datetime import UTC, datetime

import httpx
import pytest
from agentteam_common import Settings
from agentteam_contracts import ModelCallRequest, ModelOutcome, Sampling
from pydantic import ValidationError

from services.workflow.app.models.client import ModelClient, retry_fits
from services.workflow.app.models.ledger import Ledger
from services.workflow.app.models.store import Store

MODEL = "openai/gpt-oss-20b"
FIXED_NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)


def model_request(run_id="run-1", content="Example issue"):
    return ModelCallRequest(
        run_id=run_id,
        step="triage",
        role="triager",
        messages=[{"role": "user", "content": content}],
        sampling=Sampling(),
        taint=2,
        source="github_issue",
    )


def make_client(
    tmp_path,
    provider,
    *,
    now=lambda: FIXED_NOW,
    monotonic=lambda: 0,
    sleep=lambda _: None,
    **overrides,
):
    settings = replace(
        Settings.from_env(),
        models_sqlite_path=str(tmp_path / "models.sqlite"),
        **overrides,
    )
    store = Store(settings)
    return (
        ModelClient(
            settings=settings,
            ledger=Ledger(settings, store),
            store=store,
            http=httpx.Client(transport=httpx.MockTransport(provider)),
            now=now,
            monotonic=monotonic,
            sleep=sleep,
        ),
        settings,
    )


def test_retry_fits_only_when_wait_and_generation_fit_remaining_timeout():
    assert retry_fits(remaining_s=60, retry_after_s=30, expected_generation_s=15)
    assert not retry_fits(remaining_s=40, retry_after_s=30, expected_generation_s=15)


def test_model_request_is_triager_only_and_bounds_taint():
    request = ModelCallRequest(
        run_id="run-1",
        step="triage",
        role="triager",
        messages=[{"role": "user", "content": "Example issue"}],
        sampling=Sampling(),
        taint=2,
        source="github_issue",
    )

    assert request.role == "triager"
    with pytest.raises(ValidationError):
        ModelCallRequest.model_validate({**request.model_dump(), "taint": 4})


def test_model_settings_read_key_from_environment_without_repr_leak(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-only-value")

    settings = Settings.from_env()

    assert settings.groq_api_key == "test-only-value"
    assert "test-only-value" not in repr(settings)


def test_model_settings_reject_non_allowlisted_host(monkeypatch):
    monkeypatch.setenv("GROQ_BASE_URL", "https://api.groq.com.attacker.invalid/v1")

    with pytest.raises(ValueError, match="MODELS_FREE_HOSTS"):
        Settings.from_env()


def test_model_outcome_is_a_single_flat_pydantic_model():
    outcome = ModelOutcome(
        outcome="completed",
        request_id="mc-example",
        decided_at=datetime(2026, 10, 30, tzinfo=UTC),
        model_pinned="openai/gpt-oss-20b",
        run_id="run-1",
        step="triage",
    )

    assert outcome.outcome == "completed"


def test_duplicate_model_call_returns_cached_completion_and_logs_fields(
    tmp_path, caplog, monkeypatch
):
    calls = 0

    def provider(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "model": "openai/gpt-oss-20b",
                "choices": [
                    {"message": {"content": "Example triage", "tool_calls": []}}
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    monkeypatch.setenv("GROQ_API_KEY", "test-only-value")
    client, _ = make_client(tmp_path, provider)
    request = model_request(content="Private issue body")

    with caplog.at_level(logging.INFO, logger="services.workflow.app.models.client"):
        first = client.call(request)
        replay = client.call(request)

    assert first.outcome == "completed"
    assert replay.model_dump() == first.model_dump()
    assert replay.text == "Example triage"
    assert calls == 1
    call_records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("model_call ")
    ]
    assert any(
        record.model_pinned == MODEL
        and record.decided_at
        and record.started_at
        and record.finished_at
        and '"model_pinned":"openai/gpt-oss-20b"' in record.getMessage()
        and '"decided_at":"2026-10-30T12:00:00Z"' in record.getMessage()
        for record in call_records
    )
    assert all(
        "Private issue body" not in record.getMessage() for record in caplog.records
    )
    assert "test-only-value" not in caplog.text


def test_short_429_retries_once_on_the_same_pinned_model(tmp_path):
    requests = []
    sleeps = []

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "5"},
                json={"error": {"code": "rate_limit_exceeded"}},
            )
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [{"message": {"content": "completed"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )

    client, _ = make_client(tmp_path, provider, sleep=sleeps.append)
    result = client.call(model_request())

    assert result.outcome == "completed"
    assert result.retry_count == 1
    assert [request.url.path for request in requests] == [
        "/openai/v1/chat/completions",
        "/openai/v1/chat/completions",
    ]
    assert [json.loads(request.content)["model"] for request in requests] == [
        MODEL,
        MODEL,
    ]
    assert sleeps == [5.0]


def test_503_uses_same_pinned_model_for_one_fitting_retry(tmp_path):
    calls = 0

    def provider(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"Retry-After": "2"})
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [{"message": {"content": "completed"}}],
            },
        )

    client, _ = make_client(tmp_path, provider)

    assert client.call(model_request()).outcome == "completed"
    assert calls == 2


def test_retry_after_throttle_uses_same_identity_and_counts_retry(tmp_path):
    calls = 0

    def provider(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "90"})
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [{"message": {"content": "resumed"}}],
            },
        )

    client, _ = make_client(tmp_path, provider)
    request = model_request()

    assert client.call(request).outcome == "throttled"
    resumed = client.call(request)

    assert resumed.outcome == "completed"
    assert resumed.retry_count == 1
    assert calls == 2


def test_provider_quota_result_carries_retry_at_and_scope(tmp_path):
    def provider(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": "quota_exhausted",
                "model": MODEL,
                "retry_at": "2026-10-31T00:00:00Z",
                "scope": "day_tokens",
            },
        )

    client, _ = make_client(tmp_path, provider)
    result = client.call(model_request())

    assert result.outcome == "quota_exhausted"
    assert result.retry_at == datetime(2026, 10, 31, tzinfo=UTC)
    assert result.scope == "day_tokens"


def test_429_that_does_not_fit_returns_throttled_without_retry(tmp_path):
    calls = 0

    def provider(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            429,
            headers={"Retry-After": "30"},
            json={"error": {"code": "rate_limit_exceeded"}},
        )

    monotonic_values = iter((0.0, 0.0, 20.0))
    client, _ = make_client(
        tmp_path,
        provider,
        monotonic=lambda: next(monotonic_values),
    )
    result = client.call(model_request())

    assert result.outcome == "throttled"
    assert result.retry_in == 30
    assert calls == 1


def test_sampling_above_role_max_is_rejected_without_provider_call(tmp_path):
    calls = 0

    def provider(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    client, _ = make_client(tmp_path, provider)
    request = model_request().model_copy(update={"sampling": Sampling(max_tokens=2048)})
    result = client.call(request)

    assert result.outcome == "failed"
    assert result.error_code == "max_tokens_exceeded"
    assert calls == 0


def test_taint_and_prompt_caps_refuse_before_provider_call(tmp_path):
    calls = 0

    def provider(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    client, _ = make_client(tmp_path, provider, triager_prompt_token_cap=1)
    prompt_too_long = client.call(model_request())
    tainted_feedback = client.call(
        ModelCallRequest(
            run_id="run-feedback",
            step="triage",
            role="triager",
            messages=[{"role": "user", "content": "Feedback"}],
            sampling=Sampling(),
            taint=2,
            source="user_feedback",
        )
    )

    assert prompt_too_long.error_code == "prompt_too_long"
    assert tainted_feedback.error_code == "taint_refused"
    assert calls == 0


def test_reported_model_mismatch_is_not_cached(tmp_path):
    calls = 0

    def provider(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "model": "another/model",
                "choices": [{"message": {"content": "must not be used"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )

    client, _ = make_client(tmp_path, provider)
    first = client.call(model_request())
    second = client.call(model_request())

    assert first.outcome == second.outcome == "failed"
    assert first.error_code == "model_mismatch"
    assert calls == 2


def test_token_window_quota_returns_minute_scope_before_provider_call(tmp_path):
    calls = 0

    def provider(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [{"message": {"content": "completed"}}],
            },
        )

    client, _ = make_client(
        tmp_path,
        provider,
        triager_prompt_token_cap=100,
        groq_tpm=1200,
    )
    assert client.call(model_request(run_id="run-first")).outcome == "completed"

    result = client.call(model_request(run_id="run-second"))

    assert result.outcome == "quota_exhausted"
    assert result.scope == "minute_tokens"
    assert calls == 1


def test_daily_quota_exhaustion_is_not_cached_and_retry_after_reset_calls_provider(
    tmp_path,
):
    calls = 0
    current = [FIXED_NOW]

    def provider(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [{"message": {"content": "completed"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )

    client, _ = make_client(
        tmp_path,
        provider,
        now=lambda: current[0],
        groq_rpd=1,
    )
    assert client.call(model_request(run_id="run-first")).outcome == "completed"

    stopped = client.call(model_request(run_id="run-second"))
    assert stopped.outcome == "quota_exhausted"
    assert stopped.scope == "day_requests"
    assert stopped.retry_at == datetime(2026, 10, 31, tzinfo=UTC)
    assert calls == 1

    current[0] = datetime(2026, 10, 31, 0, 1, tzinfo=UTC)
    resumed = client.call(model_request(run_id="run-second"))
    assert resumed.outcome == "completed"
    assert calls == 2
