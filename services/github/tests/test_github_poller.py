import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from services.github.app.client import GithubClient
from services.github.app.poller import Poller
from services.github.app.rules import fingerprint, marker
from services.github.app.schemas import GithubIssue
from services.github.app.settings import GithubSettings
from services.github.app.store import IssueRow, Store
from services.github.app.workflow_client import WorkflowClient
from services.github.tests.github_fakes import FakeGithub, FakeWorkflow

NOW = datetime(2026, 11, 6, 12, tzinfo=UTC)
ISSUE_KEY = "gh-100-7"
ISSUE_PATH = "/repos/agent-bot/fork/issues/7"


def make_row(state="submitted"):
    return IssueRow(
        issue_key=ISSUE_KEY,
        number=7,
        state=state,
        reason=None,
        deferred_reason=None,
        fingerprint="stored-fingerprint",
        github_updated_at=(NOW - timedelta(hours=1)).isoformat(),
        created_at=(NOW - timedelta(days=2)).isoformat(),
        start_attempts=0,
        admitted_at=NOW.isoformat(),
        submitted_at=(NOW - timedelta(minutes=1)).isoformat(),
        finished_at=None,
        result_status=None,
    )


def make_poller(
    tmp_path,
    fake_github,
    fake_workflow,
    *,
    max_starts_per_cycle=1,
    max_starts_per_day=10,
    holder="test-poller",
    clock_start=NOW,
    process_own_issues=False,
):
    now = [clock_start]
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    settings = GithubSettings(
        token="test-token",
        repo=fake_github.repo,
        workflow_url="http://workflow:8002",
        sqlite_path=str(tmp_path / "github.sqlite"),
        max_starts_per_cycle=max_starts_per_cycle,
        max_starts_per_day=max_starts_per_day,
        process_own_issues=process_own_issues,
    )
    store = Store(settings.sqlite_path)
    github_http = httpx.AsyncClient(transport=fake_github.transport)
    workflow_http = httpx.AsyncClient(transport=fake_workflow.transport)
    github = GithubClient(
        token=settings.token,
        repo=settings.repo,
        http=github_http,
        clock=lambda: now[0],
        sleep=sleep,
    )
    workflow = WorkflowClient(base_url=settings.workflow_url, http=workflow_http)
    poller = Poller(
        settings=settings,
        github=github,
        workflow=workflow,
        store=store,
        clock=lambda: now[0],
        sleep=sleep,
        holder=holder,
    )
    return (
        poller,
        store,
        github_http,
        workflow_http,
        fake_github,
        fake_workflow,
        now,
        sleeps,
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("state", "labels"),
    [("closed", ("agent-team",)), ("open", ("bug",))],
)
async def test_confirmed_closed_or_unlabelled_cancels_only_intake_state(
    tmp_path, state, labels
):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_github.add_issue(
        7,
        state=state,
        labels=labels,
        created_at=NOW - timedelta(days=2),
        updated_at=NOW - timedelta(hours=1),
    )
    fake_workflow = FakeWorkflow()
    (
        poller,
        store,
        github_http,
        workflow_http,
        fake_github,
        fake_workflow,
        _,
        _,
    ) = make_poller(tmp_path, fake_github, fake_workflow)
    store.upsert_issue(make_row())

    try:
        await poller.prepare()
        await poller.run_cycle()
    finally:
        await github_http.aclose()
        await workflow_http.aclose()

    row = store.get_issue(ISSUE_KEY)
    assert row is not None and row.state == "cancelled"
    assert fake_github.calls.count(("GET", ISSUE_PATH, {})) == 1
    assert fake_workflow.calls == []


@pytest.mark.anyio
async def test_unchanged_eligible_issue_stays_active_after_revalidation(tmp_path):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_github.add_issue(
        7,
        created_at=NOW - timedelta(days=2),
        updated_at=NOW - timedelta(hours=1),
    )
    fake_workflow = FakeWorkflow()
    fake_workflow.set_response(
        "GET",
        f"/v1/issues/{ISSUE_KEY}/run",
        202,
        {"issue_id": ISSUE_KEY, "state": "running"},
    )
    (
        poller,
        store,
        github_http,
        workflow_http,
        fake_github,
        fake_workflow,
        _,
        _,
    ) = make_poller(tmp_path, fake_github, fake_workflow)
    original = make_row()
    store.upsert_issue(original)

    try:
        await poller.prepare()
        await poller.run_cycle()
    finally:
        await github_http.aclose()
        await workflow_http.aclose()

    assert store.get_issue(ISSUE_KEY) == original
    assert fake_github.calls.count(("GET", ISSUE_PATH, {})) == 1
    assert [path for _, path, _ in fake_workflow.calls] == [
        f"/v1/issues/{ISSUE_KEY}/run"
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("failure_status", [404, 401, 403, 503])
async def test_lookup_failures_never_cancel_or_mutate_stored_state(
    tmp_path, failure_status
):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_github.add_issue(
        7,
        created_at=NOW - timedelta(days=2),
        updated_at=NOW - timedelta(hours=1),
    )
    retries = 4 if failure_status == 503 else 1
    for _ in range(retries):
        fake_github.fail_path_next(ISSUE_PATH, failure_status)
    fake_workflow = FakeWorkflow()
    fake_workflow.set_response(
        "GET",
        f"/v1/issues/{ISSUE_KEY}/run",
        202,
        {"issue_id": ISSUE_KEY, "state": "running"},
    )
    (
        poller,
        store,
        github_http,
        workflow_http,
        fake_github,
        fake_workflow,
        _,
        sleeps,
    ) = make_poller(tmp_path, fake_github, fake_workflow)
    original = make_row()
    store.upsert_issue(original)

    try:
        await poller.prepare()
        summary = await poller.run_cycle()
    finally:
        await github_http.aclose()
        await workflow_http.aclose()

    assert store.get_issue(ISSUE_KEY) == original
    assert not any("cancel" in path for _, path, _ in fake_workflow.calls)
    if failure_status == 404:
        assert summary.aborted_reason is None
    else:
        assert summary.aborted_reason is not None
    if failure_status == 503:
        assert sleeps == [1, 2, 4]


@pytest.mark.anyio
async def test_repeated_cycles_post_one_start_and_one_ack(tmp_path):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_github.add_issue(7, created_at=NOW, updated_at=NOW)
    fake_workflow = FakeWorkflow()
    fake_workflow.set_response(
        "POST", f"/v1/issues/{ISSUE_KEY}/start", 202, {"issue_id": ISSUE_KEY}
    )
    poller, store, github_http, workflow_http, *_ = make_poller(
        tmp_path, fake_github, fake_workflow
    )

    try:
        await poller.prepare()
        first = await poller.run_cycle()
        fake_github.issues[7]["updated_at"] = (NOW + timedelta(seconds=1)).isoformat()
        second = await poller.run_cycle()
    finally:
        await github_http.aclose()
        await workflow_http.aclose()

    assert first.started == 1
    assert second.started == 0
    assert len(fake_workflow.accepted_starts) == 1
    assert (
        sum(
            method == "POST" and path.endswith("/comments")
            for method, path, _ in fake_github.calls
        )
        == 1
    )
    assert store.get_issue(ISSUE_KEY).state == "submitted"


@pytest.mark.anyio
async def test_retry_after_crash_between_start_and_state_write_is_idempotent(tmp_path):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_github.add_issue(7, created_at=NOW, updated_at=NOW)
    fake_workflow = FakeWorkflow()
    fake_workflow.set_response(
        "POST", f"/v1/issues/{ISSUE_KEY}/start", 202, {"issue_id": ISSUE_KEY}
    )
    poller, store, github_http, workflow_http, *_ = make_poller(
        tmp_path, fake_github, fake_workflow
    )
    upsert = store.upsert_issue
    should_crash = [True]

    def crash_after_workflow_start(row):
        if row.state == "submitted" and should_crash[0]:
            should_crash[0] = False
            raise RuntimeError("simulated crash")
        upsert(row)

    store.upsert_issue = crash_after_workflow_start
    try:
        await poller.prepare()
        with pytest.raises(RuntimeError, match="simulated crash"):
            await poller.run_cycle()
        store.upsert_issue = upsert
        await poller.run_cycle()
    finally:
        store.upsert_issue = upsert
        await github_http.aclose()
        await workflow_http.aclose()

    start_calls = [
        call
        for call in fake_workflow.calls
        if call[0] == "POST" and call[1].endswith("/start")
    ]
    assert len(start_calls) == 2
    assert len(fake_workflow.accepted_starts) == 1
    assert (
        sum(
            method == "POST" and path.endswith("/comments")
            for method, path, _ in fake_github.calls
        )
        == 1
    )
    assert store.get_issue(ISSUE_KEY).state == "submitted"


@pytest.mark.anyio
async def test_retry_after_crash_between_comment_post_and_record_avoids_duplicate(
    tmp_path,
):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_github.add_issue(7, created_at=NOW, updated_at=NOW)
    fake_workflow = FakeWorkflow()
    fake_workflow.set_response(
        "POST", f"/v1/issues/{ISSUE_KEY}/start", 202, {"issue_id": ISSUE_KEY}
    )
    poller, store, github_http, workflow_http, *_ = make_poller(
        tmp_path, fake_github, fake_workflow
    )
    record_comment = store.record_comment
    should_crash = [True]

    def crash_after_comment_post(*args, **kwargs):
        if should_crash[0]:
            should_crash[0] = False
            raise RuntimeError("simulated crash")
        record_comment(*args, **kwargs)

    store.record_comment = crash_after_comment_post
    try:
        await poller.prepare()
        with pytest.raises(RuntimeError, match="simulated crash"):
            await poller.run_cycle()
        store.record_comment = record_comment
        await poller.run_cycle()
    finally:
        store.record_comment = record_comment
        await github_http.aclose()
        await workflow_http.aclose()

    assert (
        sum(
            method == "POST" and path.endswith("/comments")
            for method, path, _ in fake_github.calls
        )
        == 1
    )
    assert store.comment_exists(
        ISSUE_KEY, "ack", store.get_issue(ISSUE_KEY).fingerprint
    )


@pytest.mark.anyio
async def test_start_caps_are_per_cycle_and_utc_day(tmp_path):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    for number, age_minutes in ((1, 4), (2, 3), (3, 2)):
        fake_github.add_issue(
            number,
            created_at=NOW - timedelta(minutes=age_minutes),
            updated_at=NOW,
        )
    fake_workflow = FakeWorkflow()
    for number in (1, 2, 3):
        key = f"gh-100-{number}"
        fake_workflow.set_response(
            "POST", f"/v1/issues/{key}/start", 202, {"issue_id": key}
        )
        fake_workflow.set_response(
            "GET",
            f"/v1/issues/{key}/run",
            202,
            {"issue_id": key, "state": "running"},
        )
    poller, store, github_http, workflow_http, _, fake_workflow, now, _ = make_poller(
        tmp_path,
        fake_github,
        fake_workflow,
        max_starts_per_cycle=1,
        max_starts_per_day=2,
        clock_start=NOW - timedelta(minutes=5),
    )

    try:
        await poller.prepare()
        first = await poller.run_cycle()
        second = await poller.run_cycle()
        third = await poller.run_cycle()
        now[0] = NOW + timedelta(days=1)
        next_day = await poller.run_cycle()
    finally:
        await github_http.aclose()
        await workflow_http.aclose()

    starts = [
        path
        for method, path, _ in fake_workflow.calls
        if method == "POST" and path.endswith("/start")
    ]
    assert starts == [
        "/v1/issues/gh-100-1/start",
        "/v1/issues/gh-100-2/start",
        "/v1/issues/gh-100-3/start",
    ]
    assert [first.started, second.started, third.started, next_day.started] == [
        1,
        1,
        0,
        1,
    ]
    assert [first.deferred, second.deferred, third.deferred] == [2, 1, 1]
    states = {number: store.get_issue(f"gh-100-{number}").state for number in (1, 2, 3)}
    assert states == {1: "submitted", 2: "submitted", 3: "submitted"}


@pytest.mark.anyio
async def test_five_unavailable_starts_fail_once_with_a_failure_comment(tmp_path):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_github.add_issue(7, created_at=NOW, updated_at=NOW)
    fake_workflow = FakeWorkflow()
    fake_workflow.set_response(
        "POST",
        f"/v1/issues/{ISSUE_KEY}/start",
        503,
        {"detail": "unavailable"},
    )
    poller, store, github_http, workflow_http, *_ = make_poller(
        tmp_path, fake_github, fake_workflow
    )

    try:
        await poller.prepare()
        summaries = [await poller.run_cycle() for _ in range(5)]
    finally:
        await github_http.aclose()
        await workflow_http.aclose()

    row = store.get_issue(ISSUE_KEY)
    assert row.state == "failed"
    assert row.reason == "workflow_unavailable"
    assert row.start_attempts == 5
    assert summaries[-1].failed == 1
    assert (
        sum(
            method == "POST" and path.endswith("/start")
            for method, path, _ in fake_workflow.calls
        )
        == 5
    )
    assert (
        sum(
            method == "POST" and path.endswith("/comments")
            for method, path, _ in fake_github.calls
        )
        == 1
    )


@pytest.mark.anyio
async def test_workflow_conflict_with_unknown_status_fails_intake(tmp_path):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_github.add_issue(7, created_at=NOW, updated_at=NOW)
    fake_workflow = FakeWorkflow()
    fake_workflow.set_response("POST", f"/v1/issues/{ISSUE_KEY}/start", 409)
    fake_workflow.set_response(
        "GET", f"/v1/issues/{ISSUE_KEY}/run", 404, {"detail": "not found"}
    )
    poller, store, github_http, workflow_http, *_ = make_poller(
        tmp_path, fake_github, fake_workflow
    )

    try:
        await poller.prepare()
        await poller.run_cycle()
    finally:
        await github_http.aclose()
        await workflow_http.aclose()

    row = store.get_issue(ISSUE_KEY)
    assert row.state == "failed"
    assert row.reason == "workflow_conflict"
    assert [method for method, _, _ in fake_workflow.calls] == ["POST", "GET"]
    assert any(
        "conflicting run" in comment["body"] for comment in fake_github.comments[7]
    )


@pytest.mark.anyio
@pytest.mark.parametrize("has_opt_in_event", [False, True])
async def test_first_run_baseline_requires_a_post_baseline_label_event(
    tmp_path, has_opt_in_event
):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_github.add_issue(
        7,
        created_at=NOW - timedelta(days=2),
        updated_at=NOW + timedelta(seconds=5),
    )
    if has_opt_in_event:
        fake_github.add_event(7, created_at=NOW + timedelta(seconds=1))
    fake_workflow = FakeWorkflow()
    fake_workflow.set_response(
        "POST", f"/v1/issues/{ISSUE_KEY}/start", 202, {"issue_id": ISSUE_KEY}
    )
    poller, store, github_http, workflow_http, *_ = make_poller(
        tmp_path, fake_github, fake_workflow
    )

    try:
        await poller.prepare()
        await poller.run_cycle()
    finally:
        await github_http.aclose()
        await workflow_http.aclose()

    row = store.get_issue(ISSUE_KEY)
    assert row is not None
    assert row.state == ("submitted" if has_opt_in_event else "ignored")
    assert bool(fake_workflow.accepted_starts) is has_opt_in_event


@pytest.mark.anyio
async def test_sqlite_lease_and_in_process_lock_reject_parallel_cycles(tmp_path):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_workflow = FakeWorkflow()
    poller, _, github_http, workflow_http, *_ = make_poller(
        tmp_path, fake_github, fake_workflow
    )
    second_store = Store(poller.settings.sqlite_path)
    other = Poller(
        settings=poller.settings,
        github=poller.github,
        workflow=poller.workflow,
        store=second_store,
        clock=poller.clock,
        sleep=poller.sleep,
        holder="other-poller",
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    list_issues = poller.github.list_issues

    async def block_discovery(*, since):
        entered.set()
        await release.wait()
        return await list_issues(since=since)

    try:
        await poller.prepare()
        await other.prepare()
        poller.github.list_issues = block_discovery
        running = asyncio.create_task(poller.run_cycle())
        await entered.wait()
        same_poller = await poller.run_cycle()
        other_poller = await other.run_cycle()
        release.set()
        completed = await running
    finally:
        release.set()
        await github_http.aclose()
        await workflow_http.aclose()

    assert same_poller.aborted_reason == "cycle_in_progress"
    assert other_poller.aborted_reason == "cycle_in_progress"
    assert completed.aborted_reason is None


@pytest.mark.anyio
async def test_marker_from_another_login_does_not_suppress_ack_comment(tmp_path):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    issue_data = fake_github.add_issue(7, created_at=NOW, updated_at=NOW)
    issue = GithubIssue.from_api(issue_data)
    fp = fingerprint(issue)
    fake_github.add_comment(
        7,
        body=marker("ack", ISSUE_KEY, fp),
        author="another-user",
    )
    fake_workflow = FakeWorkflow()
    fake_workflow.set_response(
        "GET",
        f"/v1/issues/{ISSUE_KEY}/run",
        202,
        {"issue_id": ISSUE_KEY, "state": "running"},
    )
    poller, store, github_http, workflow_http, *_ = make_poller(
        tmp_path, fake_github, fake_workflow
    )
    store.upsert_issue(
        replace(
            make_row(),
            fingerprint=fp,
            github_updated_at=NOW.isoformat(),
            created_at=NOW.isoformat(),
        )
    )

    try:
        await poller.prepare()
        await poller.run_cycle()
    finally:
        await github_http.aclose()
        await workflow_http.aclose()

    comments = fake_github.comments[7]
    assert len(comments) == 2
    assert {comment["user"]["login"] for comment in comments} == {
        "another-user",
        "agent-bot",
    }
    assert (
        sum(
            method == "POST" and path.endswith("/comments")
            for method, path, _ in fake_github.calls
        )
        == 1
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "retry", "expected"),
    [
        (
            "quota_exhausted",
            {"retry_at": "2026-11-07T00:00:00Z"},
            "2026-11-07T00:00:00Z",
        ),
        ("throttled", {"retry_in": 30}, "in 30 seconds"),
    ],
)
async def test_result_comments_include_model_retry_guidance(
    tmp_path, status, retry, expected
):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_github.add_issue(7, created_at=NOW, updated_at=NOW)
    fake_workflow = FakeWorkflow()
    result = {
        "issue_id": ISSUE_KEY,
        "round_id": "round-1",
        "event_id": "event-1",
        "run_id": "run-1",
        "proposal_id": "proposal-1",
        "status": status,
        **retry,
    }
    fake_workflow.set_response("GET", f"/v1/issues/{ISSUE_KEY}/run", 200, result)
    poller, store, github_http, workflow_http, *_ = make_poller(
        tmp_path, fake_github, fake_workflow
    )
    store.upsert_issue(
        replace(
            make_row(),
            fingerprint=fingerprint(GithubIssue.from_api(fake_github.issues[7])),
            github_updated_at=NOW.isoformat(),
            created_at=NOW.isoformat(),
        )
    )

    try:
        await poller.prepare()
        await poller.run_cycle()
    finally:
        await github_http.aclose()
        await workflow_http.aclose()

    row = store.get_issue(ISSUE_KEY)
    assert row.state == "finished"
    assert row.result_status == status
    result_comment = next(
        comment["body"]
        for comment in fake_github.comments[7]
        if "could not complete triage" in comment["body"]
    )
    assert expected in result_comment


@pytest.mark.anyio
async def test_cancelled_issue_is_reconsidered_when_seen_open_and_labelled(tmp_path):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_github.add_issue(7, created_at=NOW, updated_at=NOW)
    fake_workflow = FakeWorkflow()
    fake_workflow.set_response(
        "POST", f"/v1/issues/{ISSUE_KEY}/start", 202, {"issue_id": ISSUE_KEY}
    )
    poller, store, github_http, workflow_http, *_ = make_poller(
        tmp_path, fake_github, fake_workflow
    )
    store.upsert_issue(
        replace(
            make_row(state="cancelled"),
            fingerprint=fingerprint(GithubIssue.from_api(fake_github.issues[7])),
            github_updated_at=NOW.isoformat(),
            created_at=NOW.isoformat(),
        )
    )

    try:
        await poller.prepare()
        summary = await poller.run_cycle()
    finally:
        await github_http.aclose()
        await workflow_http.aclose()

    assert summary.started == 1
    assert store.get_issue(ISSUE_KEY).state == "submitted"
    assert any(path.endswith("/start") for _, path, _ in fake_workflow.calls)


@pytest.mark.anyio
async def test_submitted_run_without_status_after_ten_minutes_fails(tmp_path):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_github.add_issue(7, created_at=NOW, updated_at=NOW)
    fake_workflow = FakeWorkflow()
    poller, store, github_http, workflow_http, *_ = make_poller(
        tmp_path, fake_github, fake_workflow
    )
    store.upsert_issue(
        replace(
            make_row(),
            fingerprint=fingerprint(GithubIssue.from_api(fake_github.issues[7])),
            github_updated_at=NOW.isoformat(),
            created_at=NOW.isoformat(),
            submitted_at=(NOW - timedelta(minutes=11)).isoformat(),
        )
    )

    try:
        await poller.prepare()
        await poller.run_cycle()
    finally:
        await github_http.aclose()
        await workflow_http.aclose()

    row = store.get_issue(ISSUE_KEY)
    assert row.state == "failed"
    assert row.reason == "workflow_unavailable"


@pytest.mark.anyio
async def test_editing_a_skipped_issue_starts_it_with_one_new_version_comment(
    tmp_path,
):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_github.add_issue(7, title="", created_at=NOW, updated_at=NOW)
    fake_workflow = FakeWorkflow()
    fake_workflow.set_response(
        "POST", f"/v1/issues/{ISSUE_KEY}/start", 202, {"issue_id": ISSUE_KEY}
    )
    poller, store, github_http, workflow_http, *_ = make_poller(
        tmp_path, fake_github, fake_workflow
    )

    try:
        await poller.prepare()
        first = await poller.run_cycle()
        fake_github.issues[7]["title"] = "Now eligible"
        fake_github.issues[7]["updated_at"] = (NOW + timedelta(minutes=1)).isoformat()
        second = await poller.run_cycle()
    finally:
        await github_http.aclose()
        await workflow_http.aclose()

    assert first.skipped == 1
    assert second.started == 1
    assert store.get_issue(ISSUE_KEY).state == "submitted"
    start = next(
        payload
        for method, path, payload in fake_workflow.calls
        if method == "POST" and path.endswith("/start")
    )
    assert start["repo"] == "agent-bot/fork"
    assert start["title"] == "Now eligible"
    assert len(fake_github.comments[7]) == 2
    assert "skipped this issue" in fake_github.comments[7][0]["body"]
    assert "started a triage run" in fake_github.comments[7][1]["body"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("process_own_issues", "expected_state", "expected_starts"),
    [(False, "skipped", 0), (True, "submitted", 1)],
)
async def test_issue_opened_by_the_token_owner_is_processed_only_when_enabled(
    tmp_path, process_own_issues, expected_state, expected_starts
):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_github.add_issue(7, author="agent-bot", created_at=NOW, updated_at=NOW)
    fake_workflow = FakeWorkflow()
    fake_workflow.set_response(
        "POST", f"/v1/issues/{ISSUE_KEY}/start", 202, {"issue_id": ISSUE_KEY}
    )
    poller, store, github_http, workflow_http, *_ = make_poller(
        tmp_path,
        fake_github,
        fake_workflow,
        process_own_issues=process_own_issues,
    )

    try:
        await poller.prepare()
        summary = await poller.run_cycle()
    finally:
        await github_http.aclose()
        await workflow_http.aclose()

    assert summary.started == expected_starts
    assert len(fake_workflow.accepted_starts) == expected_starts
    assert store.get_issue(ISSUE_KEY).state == expected_state


def _committed_result(**extra):
    return {
        "issue_id": ISSUE_KEY,
        "round_id": "round-1",
        "event_id": "event-1",
        "run_id": "run-1",
        "proposal_id": "proposal-1",
        "status": "committed",
        "area": "unknown",
        "model": "openai/gpt-oss-20b",
        "provider": "groq",
        **extra,
    }


async def _run_finished_issue(tmp_path, fake_workflow, *, row_state="submitted"):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_github.add_issue(7, created_at=NOW, updated_at=NOW)
    poller, store, github_http, workflow_http, *_ = make_poller(
        tmp_path, fake_github, fake_workflow
    )
    row = replace(
        make_row(),
        fingerprint=fingerprint(GithubIssue.from_api(fake_github.issues[7])),
        github_updated_at=NOW.isoformat(),
        created_at=NOW.isoformat(),
    )
    if row_state == "finished":
        row = replace(
            row,
            state="finished",
            finished_at=NOW.isoformat(),
            result_status="committed",
        )
    store.upsert_issue(row)
    try:
        await poller.prepare()
        await poller.run_cycle()
        await poller.run_cycle()
    finally:
        await github_http.aclose()
        await workflow_http.aclose()
    results = [
        comment["body"]
        for comment in fake_github.comments.get(7, [])
        if "kind=result" in comment["body"]
    ]
    return store, results


@pytest.mark.anyio
async def test_committed_result_comment_shows_triage_once(tmp_path):
    fake_workflow = FakeWorkflow()
    fake_workflow.set_response(
        "GET",
        f"/v1/issues/{ISSUE_KEY}/run",
        200,
        _committed_result(
            area="fields",
            triage={
                "kind": "bug",
                "priority": "high",
                "summary": "Example summary.",
                "area": "fields",
            },
        ),
    )

    store, results = await _run_finished_issue(tmp_path, fake_workflow)

    assert store.get_issue(ISSUE_KEY).result_status == "committed"
    assert len(results) == 1
    assert "- Type: bug\n- Priority: high\n- Summary: Example summary." in results[0]
    assert "- Area: fields\n" in results[0]
    assert "Model: `openai/gpt-oss-20b` (groq)." in results[0]


@pytest.mark.anyio
async def test_later_cycle_result_comment_refetches_triage(tmp_path):
    fake_workflow = FakeWorkflow()
    fake_workflow.set_response(
        "GET",
        f"/v1/issues/{ISSUE_KEY}/run",
        200,
        _committed_result(triage_error="unparseable"),
    )

    _, results = await _run_finished_issue(
        tmp_path, fake_workflow, row_state="finished"
    )

    assert len(results) == 1
    assert "reply could not be read" in results[0]


@pytest.mark.anyio
async def test_later_cycle_result_comment_without_stored_output(tmp_path):
    _, results = await _run_finished_issue(
        tmp_path, FakeWorkflow(), row_state="finished"
    )

    assert len(results) == 1
    assert "triage details are no longer available" in results[0]


@pytest.mark.anyio
async def test_later_cycle_result_comment_waits_while_workflow_is_down(tmp_path):
    fake_workflow = FakeWorkflow()
    fake_workflow.set_response("GET", f"/v1/issues/{ISSUE_KEY}/run", 503, {})

    store, results = await _run_finished_issue(
        tmp_path, fake_workflow, row_state="finished"
    )

    assert results == []
    row = store.get_issue(ISSUE_KEY)
    assert not store.comment_exists(ISSUE_KEY, "result", row.fingerprint)
