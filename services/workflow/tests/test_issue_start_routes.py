import asyncio
import json
import shutil
import subprocess
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest
import restate
from agentteam_contracts import (
    IssueRunRequest,
    IssueRunResult,
    RunStartAccepted,
    RunStatus,
)
from restate.client_types import HttpError as RestateHttpError

from services.workflow.app import main as workflow

HttpxAsyncClient = httpx.AsyncClient


def anyio_backend():
    return "asyncio"


class FakeRestateClient:
    def __init__(self):
        self.sends = []
        self.send_error = None

    async def workflow_send(self, handler, *, key, arg):
        self.sends.append((handler, key, arg))
        if self.send_error is not None:
            raise self.send_error
        return SimpleNamespace(invocation_id="inv_test")


@asynccontextmanager
async def restate_client(client):
    yield client


async def request(app, method, path, *, json_body=None):
    async with HttpxAsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.request(method, path, json=json_body)


@pytest.mark.anyio
async def test_start_route_uses_workflow_key_and_returns_accepted(monkeypatch):
    fake = FakeRestateClient()
    monkeypatch.setattr(
        workflow.restate,
        "create_client",
        lambda _ingress: restate_client(fake),
    )
    payload = {
        "issue_id": "gh-100-7",
        "repo": "andialbrecht/sqlparse",
        "title": "Example",
    }

    first = await request(
        workflow.app, "POST", "/v1/issues/gh-100-7/start", json_body=payload
    )
    second = await request(
        workflow.app, "POST", "/v1/issues/gh-100-7/start", json_body=payload
    )

    assert first.status_code == second.status_code == 202
    assert (
        first.json()
        == RunStartAccepted(issue_id="gh-100-7", invocation_id="inv_test").model_dump()
    )
    assert second.json() == first.json()
    assert [call[0] for call in fake.sends] == [workflow.run_issue_v1] * 2
    assert [call[1] for call in fake.sends] == ["gh-100-7"] * 2


@pytest.mark.anyio
async def test_start_route_rejects_a_path_body_mismatch():
    response = await request(
        workflow.app,
        "POST",
        "/v1/issues/gh-100-7/start",
        json_body={
            "issue_id": "gh-100-8",
            "repo": "andialbrecht/sqlparse",
            "title": "Example",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "issue_id does not match the path"


@pytest.mark.anyio
async def test_start_route_preserves_restate_errors(monkeypatch):
    fake = FakeRestateClient()
    fake.send_error = RestateHttpError(409, "Conflict", "already accepted")
    monkeypatch.setattr(
        workflow.restate,
        "create_client",
        lambda _ingress: restate_client(fake),
    )

    response = await request(
        workflow.app,
        "POST",
        "/v1/issues/gh-100-7/start",
        json_body={
            "issue_id": "gh-100-7",
            "repo": "andialbrecht/sqlparse",
            "title": "Example",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "already accepted"


@pytest.mark.anyio
async def test_start_route_maps_transport_errors_to_503(monkeypatch):
    fake = FakeRestateClient()
    fake.send_error = httpx.ConnectError("private transport detail")
    monkeypatch.setattr(
        workflow.restate,
        "create_client",
        lambda _ingress: restate_client(fake),
    )

    response = await request(
        workflow.app,
        "POST",
        "/v1/issues/gh-100-7/start",
        json_body={
            "issue_id": "gh-100-7",
            "repo": "andialbrecht/sqlparse",
            "title": "Example",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Restate is unavailable"
    assert "private transport detail" not in response.text


@pytest.mark.anyio
async def test_run_route_maps_restate_output_states(monkeypatch):
    result = IssueRunResult(
        issue_id="gh-100-7",
        round_id="round-1",
        event_id="event-1",
        run_id="run-1",
        proposal_id="proposal-1",
        status="committed",
    )
    outputs = {
        "not-found": httpx.Response(404, json={"message": "not found"}),
        "running": httpx.Response(470, json={"message": "not ready"}),
        "finished": httpx.Response(200, json=result.model_dump(mode="json")),
    }
    requests = []

    def handle(request):
        requests.append(request)
        issue_key = json.loads(request.content)["workflowKey"]
        return outputs[issue_key]

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
        not_found = await client.get("/v1/issues/not-found/run")
        running = await client.get("/v1/issues/running/run")
        finished = await client.get("/v1/issues/finished/run")

    assert not_found.status_code == 404
    assert running.status_code == 202
    assert running.json() == RunStatus(issue_id="running").model_dump()
    assert finished.status_code == 200
    assert finished.json() == result.model_dump(mode="json")
    assert all(request.url.path == "/restate/output" for request in requests)
    assert all(
        json.loads(request.content)["workflowName"] == "IssueRunV1"
        for request in requests
    )


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
async def test_restate_harness_pins_start_and_output_statuses(monkeypatch):
    runs = []
    test_workflow = restate.Workflow("IssueRunV1")

    @test_workflow.main()
    async def delayed_result(
        ctx: restate.WorkflowContext, request: IssueRunRequest
    ) -> IssueRunResult:
        await ctx.sleep(timedelta(seconds=0.5), name="test-delay")
        runs.append(request.issue_id)
        return IssueRunResult(
            issue_id=request.issue_id,
            round_id="round-1",
            event_id="event-1",
            run_id="run-1",
            proposal_id="proposal-1",
            status="committed",
        )

    monkeypatch.setattr(workflow, "run_issue_v1", delayed_result)
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
            unknown = await client.get("/v1/issues/unknown/run")
            assert unknown.status_code == 404

            payload = {
                "issue_id": "gh-100-7",
                "repo": "andialbrecht/sqlparse",
                "title": "Example",
            }
            started = await client.post("/v1/issues/gh-100-7/start", json=payload)
            repeated = await client.post("/v1/issues/gh-100-7/start", json=payload)
            assert started.status_code == repeated.status_code == 202
            assert started.json() == repeated.json()

            running = await client.get("/v1/issues/gh-100-7/run")
            assert running.status_code == 202
            assert running.json()["state"] == "running"

            for _ in range(20):
                finished = await client.get("/v1/issues/gh-100-7/run")
                if finished.status_code == 200:
                    break
                await asyncio.sleep(0.1)
            assert finished.status_code == 200
            assert finished.json()["issue_id"] == "gh-100-7"
            assert runs == ["gh-100-7"]
