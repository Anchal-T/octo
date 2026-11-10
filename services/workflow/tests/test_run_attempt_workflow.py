import asyncio
import json
import shutil
import subprocess
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
import restate
from agentteam_contracts import (
    IssueRunRequest,
    IssueRunResult,
    ModelOutcome,
    TriageResult,
)

from services.workflow.app import main as workflow
from services.workflow.app.plan import (
    build_issue_plan_v1,
    build_triage_proposal_v1,
    map_triage_commit_v1,
    map_triage_return_v1,
)

HttpxAsyncClient = httpx.AsyncClient
BASIS = datetime(2026, 10, 31, 12, tzinfo=UTC)


def anyio_backend():
    return "asyncio"


class FakeRestateClient:
    def __init__(self):
        self.sends = []
        self.calls = []

    async def workflow_send(self, handler, *, key, arg):
        self.sends.append((handler, key, arg))
        return SimpleNamespace(invocation_id="inv_test")

    async def workflow_call(self, handler, *, key, arg):
        self.calls.append((handler, key, arg))
        return IssueRunResult(
            issue_id=arg.issue_id,
            round_id="round-1",
            event_id="event-1",
            run_id="run-1",
            proposal_id="proposal-1",
            status="committed",
            attempt=arg.attempt,
        )


@asynccontextmanager
async def restate_client(client):
    yield client


async def request(app, method, path, *, json_body=None, params=None):
    async with HttpxAsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.request(method, path, json=json_body, params=params)


def _completed_outcome() -> ModelOutcome:
    return ModelOutcome(
        outcome="completed",
        request_id="mc-example",
        decided_at=BASIS,
        model_pinned="openai/gpt-oss-20b",
        model_reported="openai/gpt-oss-20b",
        run_id="run-example",
        step="triage",
        text="{}",
        started_at=BASIS,
        finished_at=BASIS,
    )


def test_attempt_one_plan_keys_match_literal_snapshot():
    plan = build_issue_plan_v1(
        IssueRunRequest(
            issue_id="gh-1-2",
            repo="andialbrecht/sqlparse",
            title="Example",
            read_basis=BASIS,
        )
    )

    assert plan.attempt == 1
    assert plan.round.idempotency_key == "gh-1-2:v1:round:1"
    assert plan.event.idempotency_key == "gh-1-2:v1:issue"
    assert plan.run_key == "gh-1-2:v1:triage:1"
    assert plan.proposal_key == "gh-1-2:v1:triage-proposal:1"
    assert (
        plan.round_id
        == "round-4d8947ee099798f14bf9014479f8f2fe32dd673c6fc6265a34959ad1c7f2e6ef"
    )
    assert (
        plan.event_id
        == "ev-e9a406a9bb67181bb5ffa6f717348a4d94c2df1bd7087e2469bbd6c1e6a59d5f"
    )
    assert (
        plan.run_id
        == "run-aa79d519596caa9aa904c343a745dfaca7343b750095d6c16e1f98fcd075063e"
    )
    assert (
        plan.proposal_id
        == "prop-94516d0e829077a6cb75294a13054e67202d42e65225400b74006e34e91a89eb"
    )


def test_attempt_two_plan_keys_use_r2_segment_and_proposal_carries_attempt():
    plan = build_issue_plan_v1(
        IssueRunRequest(
            issue_id="gh-1-2",
            repo="andialbrecht/sqlparse",
            title="Example",
            read_basis=BASIS,
            attempt=2,
        )
    )
    outcome = _completed_outcome()
    triage = TriageResult(kind="bug", priority="low", summary="Example", area="parser")
    proposal = build_triage_proposal_v1(plan, outcome, triage)
    committed = map_triage_commit_v1(plan, outcome, triage, provider="groq")
    returned = map_triage_return_v1(
        plan,
        ModelOutcome(
            outcome="throttled",
            request_id="mc-example",
            decided_at=BASIS,
            model_pinned="openai/gpt-oss-20b",
            model_reported="openai/gpt-oss-20b",
            run_id="run-example",
            step="triage",
            retry_in=5.0,
        ),
        provider="groq",
    )

    assert plan.attempt == 2
    assert plan.round.idempotency_key == "gh-1-2:v1:r2:round:1"
    assert plan.event.idempotency_key == "gh-1-2:v1:r2:issue"
    assert plan.run_key == "gh-1-2:v1:r2:triage:1"
    assert plan.proposal_key == "gh-1-2:v1:r2:triage-proposal:1"
    assert proposal.payload["attempt"] == 2
    assert committed.attempt == 2
    assert returned.attempt == 2


@pytest.mark.anyio
async def test_start_and_run_routes_key_by_attempt(monkeypatch):
    fake = FakeRestateClient()
    monkeypatch.setattr(
        workflow.restate,
        "create_client",
        lambda _ingress: restate_client(fake),
    )
    payload = {
        "issue_id": "gh-1-2",
        "repo": "andialbrecht/sqlparse",
        "title": "Example",
        "attempt": 2,
    }

    started = await request(
        workflow.app, "POST", "/v1/issues/gh-1-2/start", json_body=payload
    )
    ran = await request(
        workflow.app, "POST", "/v1/issues/gh-1-2/run", json_body=payload
    )

    assert started.status_code == 202
    assert started.json()["attempt"] == 2
    assert ran.status_code == 200
    assert ran.json()["attempt"] == 2
    assert fake.sends[0][1] == "gh-1-2-r2"
    assert fake.calls[0][1] == "gh-1-2-r2"


@pytest.mark.anyio
async def test_status_route_reads_attempt_query(monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(470, json={"message": "not ready"})

    def mock_client(*, base_url, timeout):
        return HttpxAsyncClient(
            base_url=base_url,
            timeout=timeout,
            transport=httpx.MockTransport(handle),
        )

    monkeypatch.setattr(workflow.httpx, "AsyncClient", mock_client)
    async with HttpxAsyncClient(
        transport=httpx.ASGITransport(app=workflow.app), base_url="http://test"
    ) as client:
        ok = await client.get("/v1/issues/gh-1-2/run", params={"attempt": 2})
        bad = await client.get("/v1/issues/gh-1-2/run", params={"attempt": 0})

    assert ok.status_code == 202
    assert ok.json()["attempt"] == 2
    assert json.loads(requests[0].content)["workflowKey"] == "gh-1-2-r2"
    assert bad.status_code == 422


def docker_unavailable_reason():
    if shutil.which("docker") is None:
        return "Docker CLI is unavailable; the Restate harness test needs Docker."
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


@pytest.mark.anyio
@pytest.mark.skipif(
    docker_unavailable_reason() is not None,
    reason=docker_unavailable_reason() or "Docker is available",
)
async def test_harness_attempt_two_commits_after_attempt_one(monkeypatch):
    runs = []
    test_workflow = restate.Workflow("IssueRunV1")

    @test_workflow.main()
    async def result_for_attempt(
        ctx: restate.WorkflowContext, request: IssueRunRequest
    ) -> IssueRunResult:
        await ctx.sleep(timedelta(seconds=0.2), name="test-delay")
        runs.append((request.issue_id, request.attempt))
        return IssueRunResult(
            issue_id=request.issue_id,
            round_id="round-1",
            event_id="event-1",
            run_id="run-1",
            proposal_id="proposal-1",
            status="committed",
            attempt=request.attempt,
        )

    monkeypatch.setattr(workflow, "run_issue_v1", result_for_attempt)
    async with restate.create_test_harness(
        restate.app(services=[test_workflow])
    ) as env:
        monkeypatch.setattr(
            workflow,
            "settings",
            replace(workflow.settings, restate_ingress=env.ingress_url),
        )
        async with HttpxAsyncClient(
            transport=httpx.ASGITransport(app=workflow.app), base_url="http://test"
        ) as client:
            first = await client.post(
                "/v1/issues/gh-1-2/start",
                json={
                    "issue_id": "gh-1-2",
                    "repo": "andialbrecht/sqlparse",
                    "title": "Example",
                },
            )
            assert first.status_code == 202

            for _ in range(30):
                finished = await client.get("/v1/issues/gh-1-2/run")
                if finished.status_code == 200:
                    break
                await asyncio.sleep(0.1)
            assert finished.status_code == 200
            assert finished.json()["attempt"] == 1

            second = await client.post(
                "/v1/issues/gh-1-2/start",
                json={
                    "issue_id": "gh-1-2",
                    "repo": "andialbrecht/sqlparse",
                    "title": "Example",
                    "attempt": 2,
                },
            )
            assert second.status_code == 202
            assert second.json()["attempt"] == 2

            for _ in range(30):
                finished_two = await client.get(
                    "/v1/issues/gh-1-2/run", params={"attempt": 2}
                )
                if finished_two.status_code == 200:
                    break
                await asyncio.sleep(0.1)
            assert finished_two.status_code == 200
            assert finished_two.json()["attempt"] == 2
            assert runs == [("gh-1-2", 1), ("gh-1-2", 2)]
