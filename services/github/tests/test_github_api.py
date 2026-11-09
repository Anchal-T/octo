import asyncio
from datetime import UTC, datetime

import httpx
import pytest

from services.github.app.main import build_runtime, create_app
from services.github.app.settings import GithubSettings
from services.github.app.store import IssueRow
from services.github.tests.github_fakes import FakeGithub, FakeWorkflow

NOW = datetime(2026, 11, 6, 12, tzinfo=UTC)
TOKEN = "test-token"


def make_runtime(tmp_path):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_workflow = FakeWorkflow()

    def dispatch(request):
        if request.url.host == "api.github.com":
            return fake_github.handle(request)
        return fake_workflow.handle(request)

    async def sleep(_seconds):
        return None

    settings = GithubSettings(
        token=TOKEN,
        repo=fake_github.repo,
        workflow_url="http://workflow:8002",
        sqlite_path=str(tmp_path / "github.sqlite"),
    )
    runtime = build_runtime(
        settings,
        transport=httpx.MockTransport(dispatch),
        clock=lambda: NOW,
        sleep=sleep,
    )
    return runtime, fake_github, fake_workflow


def issue_row(key, number, state, admitted_at=None):
    return IssueRow(
        issue_key=key,
        number=number,
        state=state,
        reason=None,
        deferred_reason=None,
        fingerprint=f"fingerprint-{number}",
        github_updated_at=NOW.isoformat(),
        created_at=NOW.isoformat(),
        start_attempts=0,
        admitted_at=admitted_at,
        submitted_at=None,
        finished_at=None,
        result_status=None,
    )


@pytest.mark.anyio
async def test_health_status_and_issue_routes_expose_only_intake_fields(tmp_path):
    runtime, _, _ = make_runtime(tmp_path)
    app = create_app(runtime)
    admitted = issue_row("gh-100-7", 7, "submitted", NOW.isoformat())
    runtime.store.upsert_issue(admitted)
    runtime.store.record_admission(
        admitted.issue_key, admitted.attempt, NOW.isoformat()
    )
    runtime.store.upsert_issue(issue_row("gh-100-8", 8, "skipped"))

    try:
        await runtime.poller.prepare()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            health = await client.get("/healthz")
            status = await client.get("/v1/status")
            issues = await client.get("/v1/issues?state=submitted")
    finally:
        await runtime.github_http.aclose()
        await runtime.workflow_http.aclose()

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert status.status_code == 200
    assert status.json()["repo"] == "agent-bot/fork"
    assert status.json()["self_login"] == "agent-bot"
    assert status.json()["auth_state"] == "ok"
    assert status.json()["starts_today"] == 1
    assert TOKEN not in status.text
    assert issues.status_code == 200
    assert issues.json() == [
        {
            "issue_key": "gh-100-7",
            "number": 7,
            "state": "submitted",
            "reason": None,
            "start_attempts": 0,
            "result_status": None,
        }
    ]
    assert TOKEN not in issues.text


@pytest.mark.anyio
async def test_poll_once_reports_cycle_in_progress_as_http_409(tmp_path):
    runtime, _, _ = make_runtime(tmp_path)
    app = create_app(runtime)
    entered = asyncio.Event()
    release = asyncio.Event()
    list_issues = runtime.github.list_issues

    async def block_discovery(*, since):
        entered.set()
        await release.wait()
        return await list_issues(since=since)

    try:
        await runtime.poller.prepare()
        runtime.github.list_issues = block_discovery
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            cycle = asyncio.create_task(runtime.poller.run_cycle())
            await entered.wait()
            busy = await client.post("/v1/poll/once")
            release.set()
            completed = await cycle
            available = await client.post("/v1/poll/once")
    finally:
        release.set()
        await runtime.github_http.aclose()
        await runtime.workflow_http.aclose()

    assert busy.status_code == 409
    assert busy.json()["detail"] == "cycle_in_progress"
    assert completed.aborted_reason is None
    assert available.status_code == 200
    assert available.json()["aborted_reason"] is None
