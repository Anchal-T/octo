import httpx
import pytest
from agentteam_contracts import IssueRunRequest, IssueRunResult

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
                "attempt": 1,
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
    assert fake.queries[-1] == {"attempt": ["1"]}


@pytest.mark.anyio
async def test_status_sends_attempt_query_param():
    fake = FakeWorkflow()
    fake.set_response(
        "GET",
        "/v1/issues/gh-100-7/run",
        202,
        {"issue_id": "gh-100-7", "state": "running"},
    )
    http = httpx.AsyncClient(transport=fake.transport)
    client = WorkflowClient(base_url="http://workflow:8002", http=http)

    async with http:
        state = await client.status("gh-100-7", attempt=2)

    assert state == "running"
    assert fake.queries == [{"attempt": ["2"]}]


@pytest.mark.anyio
async def test_status_reads_triage_written_before_area_as_unknown():
    stored = {
        "issue_id": "gh-100-7",
        "round_id": "round-1",
        "event_id": "event-1",
        "run_id": "run-1",
        "proposal_id": "proposal-1",
        "status": "committed",
        "model": "openai/gpt-oss-20b",
        "provider": "groq",
        "retry_at": None,
        "retry_in": None,
        "scope": None,
        "decided_at": "2026-10-31T12:00:00Z",
        "started_at": "2026-10-31T12:00:00Z",
        "finished_at": "2026-10-31T12:00:01Z",
        "triage": {
            "kind": "bug",
            "priority": "medium",
            "summary": "Example summary.",
        },
        "triage_error": None,
    }
    fake = FakeWorkflow()
    fake.set_response("GET", "/v1/issues/gh-100-7/run", 200, stored)
    http = httpx.AsyncClient(transport=fake.transport)
    client = WorkflowClient(base_url="http://workflow:8002", http=http)

    async with http:
        state = await client.status("gh-100-7")

    assert isinstance(state, IssueRunResult)
    assert state.area == "unknown"
    assert state.triage is not None
    assert state.triage.area == "unknown"
    assert state.triage.summary == "Example summary."


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
            IssueRunRequest(
                issue_id="gh-100-7", repo="bot/fork", title="Example", attempt=2
            )
        )

    assert result == expected
    assert [(method, path) for method, path, _ in fake.calls] == [
        ("POST", path),
        ("GET", "/v1/issues/gh-100-7/run"),
    ]
    assert fake.queries[-1] == {"attempt": ["2"]}
