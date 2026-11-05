import logging
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from services.github.app.client import GithubClient, GithubFailure
from services.github.app.schemas import GithubIssue
from services.github.tests.github_fakes import FakeGithub

NOW = datetime(2026, 11, 6, 12, tzinfo=UTC)


def make_client(fake, sleeps=None, clock=lambda: NOW):
    async def sleep(seconds):
        if sleeps is not None:
            sleeps.append(seconds)

    http = httpx.AsyncClient(transport=fake.transport)
    client = GithubClient(
        token="test-token",
        repo=fake.repo,
        http=http,
        clock=clock,
        sleep=sleep,
    )
    return client, http


@pytest.mark.anyio
async def test_client_startup_and_issue_related_endpoints():
    fake = FakeGithub()
    fake.add_issue(7)
    fake.add_event(7)
    fake.add_comment(7, body="existing", author="agent-bot", comment_id=10)
    client, http = make_client(fake)
    async with http:
        assert await client.get_self() == "agent-bot"
        repo = await client.get_repo()
        await client.ensure_label("agent-team")
        events = await client.list_label_events(7)
        comments = await client.list_comments(7)
        comment_id = await client.post_comment(7, "new comment")

    assert repo.id == 100
    assert repo.full_name == "bot/fork"
    assert repo.fork and repo.has_issues
    assert "agent-team" in fake.labels
    assert [event.label_name for event in events] == ["agent-team"]
    assert [comment.id for comment in comments] == [10]
    assert comment_id == 11


@pytest.mark.anyio
async def test_issue_listing_uses_since_sort_and_full_page_pagination():
    fake = FakeGithub()
    for number in range(1, 102):
        fake.add_issue(
            number,
            created_at=NOW + timedelta(seconds=number),
            updated_at=NOW + timedelta(seconds=number),
        )
    fake.add_issue(200, labels=("bug",), updated_at=NOW + timedelta(seconds=1))
    fake.add_issue(201, state="closed", updated_at=NOW + timedelta(seconds=1))
    client, http = make_client(fake)
    async with http:
        result = await client.list_issues(since=NOW)

    assert result.pages == 2
    assert not result.truncated
    assert len(result.issues) == 101
    assert result.issues[0].number == 1
    assert result.max_updated_at == NOW + timedelta(seconds=101)
    issue_calls = [call for call in fake.calls if call[1].endswith("/issues")]
    assert [call[2]["page"][0] for call in issue_calls] == ["1", "2"]
    assert all(call[2]["since"] == [NOW.isoformat()] for call in issue_calls)


@pytest.mark.anyio
async def test_issue_listing_marks_tenth_full_page_truncated():
    fake = FakeGithub()
    for number in range(1, 1001):
        fake.add_issue(
            number,
            created_at=NOW + timedelta(seconds=number),
            updated_at=NOW + timedelta(seconds=number),
        )
    client, http = make_client(fake)
    async with http:
        result = await client.list_issues(since=None)

    assert result.pages == 10
    assert result.truncated
    assert len(result.issues) == 1000


@pytest.mark.anyio
async def test_client_maps_null_body_and_keeps_pull_request_items():
    fake = FakeGithub()
    fake.add_issue(7, body=None, pull_request=True)
    client, http = make_client(fake)
    async with http:
        result = await client.list_issues(since=None)

    assert len(result.issues) == 1
    assert result.issues[0].body == ""
    assert result.issues[0].is_pull_request
    assert isinstance(result.issues[0], GithubIssue)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "headers", "outcome", "error_code"),
    [
        (401, {}, "unauthorized", "github_unauthorized"),
        (403, {}, "forbidden", "github_forbidden"),
        (404, {}, "not_found", "repo_not_found"),
        (410, {}, "gone", "issues_disabled"),
    ],
)
async def test_client_maps_github_auth_permission_and_listing_errors(
    status, headers, outcome, error_code
):
    fake = FakeGithub()
    fake.fail_next(status, headers)
    client, http = make_client(fake)
    async with http:
        with pytest.raises(GithubFailure) as caught:
            await client.list_issues(since=None)

    assert caught.value.outcome.outcome == outcome
    assert caught.value.outcome.error_code == error_code
    assert caught.value.outcome.http_status == status


@pytest.mark.anyio
async def test_client_maps_primary_rate_limit_to_retry_time():
    reset = int((NOW + timedelta(seconds=90)).timestamp())
    fake = FakeGithub()
    fake.fail_next(403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(reset)})
    client, http = make_client(fake)
    async with http:
        with pytest.raises(GithubFailure) as caught:
            await client.list_issues(since=None)

    outcome = caught.value.outcome
    assert outcome.outcome == "rate_limited"
    assert outcome.scope == "primary"
    assert outcome.retry_at == datetime.fromtimestamp(reset, UTC)
    assert (
        outcome.retry_in == (datetime.fromtimestamp(reset, UTC) - NOW).total_seconds()
    )


@pytest.mark.anyio
@pytest.mark.parametrize("status", [403, 429])
async def test_client_sleeps_retry_after_and_retries_secondary_limit(status):
    sleeps = []
    fake = FakeGithub()
    fake.fail_next(status, {"retry-after": "30"})
    client, http = make_client(fake, sleeps)
    async with http:
        assert await client.get_self() == "agent-bot"

    assert sleeps == [30]
    assert len(fake.calls) == 2


@pytest.mark.anyio
async def test_client_does_not_sleep_past_secondary_limit_cap():
    sleeps = []
    fake = FakeGithub()
    fake.fail_next(429, {"retry-after": "121"})
    client, http = make_client(fake, sleeps)
    async with http:
        with pytest.raises(GithubFailure) as caught:
            await client.get_self()

    assert sleeps == []
    assert caught.value.outcome.outcome == "rate_limited"
    assert caught.value.outcome.scope == "secondary"
    assert caught.value.outcome.retry_in == 121


@pytest.mark.anyio
async def test_client_retries_secondary_limit_only_once():
    sleeps = []
    fake = FakeGithub()
    fake.fail_next(429, {"retry-after": "30"})
    fake.fail_next(429, {"retry-after": "30"})
    client, http = make_client(fake, sleeps)
    async with http:
        with pytest.raises(GithubFailure) as caught:
            await client.get_self()

    assert sleeps == [30]
    assert caught.value.outcome.scope == "secondary"


@pytest.mark.anyio
async def test_client_maps_issue_not_found_and_comment_rejected():
    fake = FakeGithub()
    client, http = make_client(fake)
    async with http:
        with pytest.raises(GithubFailure) as missing:
            await client.list_label_events(404)
        fake.add_issue(7)
        fake.fail_next(422)
        with pytest.raises(GithubFailure) as rejected:
            await client.post_comment(7, "bad comment")

    assert missing.value.outcome.error_code == "issue_not_found"
    assert rejected.value.outcome.outcome == "invalid"
    assert rejected.value.outcome.error_code == "comment_rejected"


@pytest.mark.anyio
async def test_client_retries_server_errors_with_exponential_backoff():
    sleeps = []
    fake = FakeGithub()
    fake.fail_next(500)
    fake.fail_next(503)
    client, http = make_client(fake, sleeps)
    async with http:
        assert await client.get_self() == "agent-bot"

    assert sleeps == [1, 2]
    assert len(fake.calls) == 3


@pytest.mark.anyio
async def test_client_exhausts_three_retries_after_server_errors():
    sleeps = []
    fake = FakeGithub()
    for status in (500, 500, 500, 500):
        fake.fail_next(status)
    client, http = make_client(fake, sleeps)
    async with http:
        with pytest.raises(GithubFailure) as caught:
            await client.get_self()

    assert sleeps == [1, 2, 4]
    assert caught.value.outcome.outcome == "unavailable"
    assert caught.value.outcome.error_code == "github_unavailable"


@pytest.mark.anyio
async def test_client_retries_network_errors_and_redacts_log(caplog):
    sleeps = []
    calls = 0

    def fail_once(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("test transport detail", request=request)
        return httpx.Response(200, json={"login": "agent-bot"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(fail_once))

    async def sleep(seconds):
        sleeps.append(seconds)

    client = GithubClient(
        token="test-token",
        repo="bot/fork",
        http=http,
        clock=lambda: NOW,
        sleep=sleep,
    )
    with caplog.at_level(logging.INFO):
        async with http:
            assert await client.get_self() == "agent-bot"

    assert sleeps == [1]
    assert "test-token" not in caplog.text
    client_logs = [
        record
        for record in caplog.records
        if record.name == "services.github.app.client"
    ]
    assert client_logs
    assert all("method=" in record.message for record in client_logs)
    assert all("status=" in record.message for record in client_logs)
