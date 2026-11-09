import httpx
import pytest
from agentteam_contracts import IssueRunRequest

from services.github.app.workflow_client import WorkflowClient
from services.github.tests.github_fakes import FakeWorkflow


@pytest.mark.anyio
async def test_start_sends_issue_request_and_accepts_202():
    fake = FakeWorkflow()
    path = "/v1/issues/gh-100-7/start"
    fake.set_response("POST", path, 202, {"issue_id": "gh-100-7"})
    request = IssueRunRequest(
        issue_id="gh-100-7", repo="bot/fork", title="Example", body="Body"
    )
    http = httpx.AsyncClient(transport=fake.transport)
    client = WorkflowClient(base_url="http://workflow:8002", http=http)

    async with http:
        result = await client.start(request)

    assert result == "submitted"
    assert fake.calls == [
        (
            "POST",
            path,
            {
                "issue_id": "gh-100-7",
                "repo": "bot/fork",
                "title": "Example",
                "body": "Body",
                "read_basis": None,
            },
        )
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "expected"),
    [(200, "submitted"), (422, "rejected"), (503, "unavailable"), (500, "unavailable")],
)
async def test_start_maps_repeat_rejection_and_server_errors(status, expected):
    fake = FakeWorkflow()
    fake.set_response("POST", "/v1/issues/gh-100-7/start", status)
    http = httpx.AsyncClient(transport=fake.transport)
    client = WorkflowClient(base_url="http://workflow:8002", http=http)

    async with http:
        result = await client.start(
            IssueRunRequest(issue_id="gh-100-7", repo="bot/fork", title="Example")
        )

    assert result == expected


@pytest.mark.anyio
async def test_start_maps_timeout_to_unavailable():
    def timeout(request):
        raise httpx.ReadTimeout("simulated timeout", request=request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(timeout))
    client = WorkflowClient(base_url="http://workflow:8002", http=http)

    async with http:
        result = await client.start(
            IssueRunRequest(issue_id="gh-100-7", repo="bot/fork", title="Example")
        )

    assert result == "unavailable"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status_code", "payload", "expected"),
    [
        (404, {"detail": "not found"}, "unknown"),
        (202, {"issue_id": "gh-100-7", "state": "running"}, "running"),
        (
            200,
            {
                "issue_id": "gh-100-7",
                "round_id": "round-1",
                "event_id": "event-1",
                "run_id": "run-1",
                "proposal_id": "proposal-1",
                "status": "committed",
            },
            "result",
        ),
    ],
)
async def test_status_maps_unknown_running_and_finished(status_code, payload, expected):
    fake = FakeWorkflow()
    fake.set_response("GET", "/v1/issues/gh-100-7/run", status_code, payload)
    http = httpx.AsyncClient(transport=fake.transport)
    client = WorkflowClient(base_url="http://workflow:8002", http=http)

    async with http:
        state = await client.status("gh-100-7")

    if expected == "result":
        assert state.status == "committed"
    else:
        assert state == expected


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status_code", "status_payload", "expected"),
    [
        (202, {"issue_id": "gh-100-7", "state": "running"}, "submitted"),
        (404, {"detail": "not found"}, "conflict"),
        (503, {"detail": "unavailable"}, "unavailable"),
    ],
)
async def test_start_resolves_conflict_using_run_status(
    status_code, status_payload, expected
):
    fake = FakeWorkflow()
    path = "/v1/issues/gh-100-7/start"
    fake.set_response("POST", path, 409, {"detail": "conflict"})
    fake.set_response("GET", "/v1/issues/gh-100-7/run", status_code, status_payload)
    http = httpx.AsyncClient(transport=fake.transport)
    client = WorkflowClient(base_url="http://workflow:8002", http=http)

    async with http:
        result = await client.start(
            IssueRunRequest(issue_id="gh-100-7", repo="bot/fork", title="Example")
        )

    assert result == expected
    assert [(method, path) for method, path, _ in fake.calls] == [
        ("POST", path),
        ("GET", "/v1/issues/gh-100-7/run"),
    ]
