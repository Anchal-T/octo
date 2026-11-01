import asyncio
import json
import shutil
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from agentteam_common import Settings, stable_id
from agentteam_contracts import IssueRunRequest
from restate import app as restate_app
from restate import create_test_harness
from restate.client_types import HttpError as RestateHttpError

from services.workflow.app import main as workflow
from services.workflow.app.models import Ledger, ModelClient, Store

MODEL = "openai/gpt-oss-20b"
FIXED_NOW = datetime(2026, 10, 31, 12, tzinfo=UTC)


def _docker_unavailable_reason() -> str | None:
    if shutil.which("docker") is None:
        return (
            "Docker CLI is unavailable; this Restate Testcontainers test needs Docker."
        )
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "Docker daemon is unavailable; start Docker to run the Restate harness."
    if result.returncode != 0:
        return "Docker daemon is unavailable; start Docker to run the Restate harness."
    return None


DOCKER_UNAVAILABLE = _docker_unavailable_reason()
pytestmark = pytest.mark.skipif(
    DOCKER_UNAVAILABLE is not None,
    reason=DOCKER_UNAVAILABLE or "Docker is available",
)


def _provider_response(model: str = MODEL) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [{"message": {"content": "Example triage"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        },
    )


def test_restate_replay_and_triager_retry_paths(monkeypatch, tmp_path):
    provider_calls: dict[str, int] = {}
    requested_models: list[str] = []
    memory_calls: list[tuple[str, dict]] = []
    model_client_calls = []

    def provider(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requested_models.append(payload["model"])
        messages = json.dumps(payload["messages"])
        case = next(
            name
            for name in (
                "replay-case",
                "throttle-success-case",
                "second-throttle-case",
                "quota-retry-case",
                "quota-over-cap-case",
                "mismatch-case",
            )
            if name in messages
        )
        provider_calls[case] = provider_calls.get(case, 0) + 1
        count = provider_calls[case]

        throttle_limit = 1 if case == "throttle-success-case" else 2
        if (
            case in ("throttle-success-case", "second-throttle-case")
            and count <= throttle_limit
        ):
            return httpx.Response(429, headers={"Retry-After": "0.01"})
        if case == "quota-retry-case" and count == 1:
            return httpx.Response(
                429,
                json={
                    "error": "quota_exhausted",
                    "retry_at": (FIXED_NOW + timedelta(seconds=0.01)).isoformat(),
                    "scope": "day_tokens",
                },
            )
        if case == "quota-over-cap-case":
            return httpx.Response(
                429,
                json={
                    "error": "quota_exhausted",
                    "retry_at": (FIXED_NOW + timedelta(seconds=7201)).isoformat(),
                    "scope": "day_tokens",
                },
            )
        if case == "mismatch-case":
            return _provider_response("another/vendor-model")
        return _provider_response()

    monkeypatch.setenv("GROQ_API_KEY", "test-only-value")
    settings = replace(
        Settings.from_env(),
        models_sqlite_path=str(tmp_path / "models.sqlite"),
        triager_max_tokens=32,
        triager_expected_generation_s=60,
    )
    store = Store(settings)
    client = ModelClient(
        settings=settings,
        ledger=Ledger(settings, store),
        store=store,
        http=httpx.Client(transport=httpx.MockTransport(provider)),
        now=lambda: FIXED_NOW,
    )
    original_call = client.call

    def count_model_call(request):
        outcome = original_call(request)
        model_client_calls.append((request.run_id, request, outcome))
        return outcome

    client.call = count_model_call
    monkeypatch.setattr(workflow, "settings", settings)
    monkeypatch.setattr(workflow, "_model_client", client)

    async def fake_post_memory(path: str, payload: dict) -> dict[str, str]:
        memory_calls.append((path, payload))
        return {"status": "ok"}

    monkeypatch.setattr(workflow, "post_memory", fake_post_memory)

    async def invoke(env, case: str):
        request = IssueRunRequest(
            issue_id=f"issue-{case}",
            title=f"Example {case}",
            body=f"Example body {case}",
        )
        return await env.client.workflow_call(
            workflow.run_issue_v1,
            key=request.issue_id,
            arg=request,
        )

    def outcomes_for(case: str):
        run_id = stable_id("run", f"issue-{case}:v1:triage:1")
        return [
            outcome
            for actual_id, _, outcome in model_client_calls
            if actual_id == run_id
        ]

    def assert_retry_attempts(case: str):
        outcomes = outcomes_for(case)
        assert len(outcomes) == 2
        assert outcomes[0].request_id == outcomes[1].request_id
        assert [outcome.retry_count for outcome in outcomes] == [0, 1]
        with store.connection() as connection:
            rows = connection.execute(
                "SELECT retry_ordinal FROM call_attempt WHERE request_id = ? "
                "ORDER BY retry_ordinal",
                (outcomes[0].request_id,),
            ).fetchall()
        assert [row["retry_ordinal"] for row in rows] == [1, 2]

    async def exercise():
        async with create_test_harness(
            restate_app(services=[workflow.issue_run_v1]),
            always_replay=True,
        ) as env:
            replay_result = await invoke(env, "replay-case")
            replay_run_id = stable_id("run", "issue-replay-case:v1:triage:1")
            assert replay_result.status == "committed"
            assert replay_result.issue_id == "issue-replay-case"
            assert provider_calls["replay-case"] == 1
            assert len(outcomes_for("replay-case")) == 1
            assert replay_result.run_id == replay_run_id

            throttle_result = await invoke(env, "throttle-success-case")
            assert throttle_result.status == "committed"
            assert provider_calls["throttle-success-case"] == 2
            assert_retry_attempts("throttle-success-case")

            second_throttle = await invoke(env, "second-throttle-case")
            assert second_throttle.status == "throttled"
            assert second_throttle.retry_in == 0.01
            assert second_throttle.model == MODEL
            assert second_throttle.provider == "groq"
            assert second_throttle.decided_at is not None
            assert second_throttle.started_at is not None
            assert second_throttle.finished_at is not None
            assert provider_calls["second-throttle-case"] == 2
            assert_retry_attempts("second-throttle-case")

            quota_result = await invoke(env, "quota-retry-case")
            assert quota_result.status == "committed"
            assert provider_calls["quota-retry-case"] == 2
            assert_retry_attempts("quota-retry-case")

            quota_over_cap = await invoke(env, "quota-over-cap-case")
            assert quota_over_cap.status == "quota_exhausted"
            assert quota_over_cap.retry_at == FIXED_NOW + timedelta(seconds=7201)
            assert quota_over_cap.model == MODEL
            assert quota_over_cap.scope == "day_tokens"
            assert quota_over_cap.provider == "groq"
            assert quota_over_cap.decided_at == FIXED_NOW
            assert quota_over_cap.started_at == FIXED_NOW
            assert quota_over_cap.finished_at == FIXED_NOW
            assert provider_calls["quota-over-cap-case"] == 1

            with pytest.raises(RestateHttpError) as mismatch_error:
                await invoke(env, "mismatch-case")
            assert mismatch_error.value.status_code == 500
            assert provider_calls["mismatch-case"] == 1
            mismatch_outcome = outcomes_for("mismatch-case")
            assert len(mismatch_outcome) == 1
            assert mismatch_outcome[0].error_code == "model_mismatch"

    asyncio.run(exercise())

    assert requested_models
    assert set(requested_models) == {MODEL}
    assert len([call for call in memory_calls if call[0].endswith("/commit")]) == 3
