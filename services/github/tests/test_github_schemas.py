from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from services.github.app.schemas import (
    CycleSummary,
    GithubComment,
    GithubIssue,
    GithubOutcome,
    IssueView,
    LabelEvent,
    PollResult,
    RepoInfo,
)

NOW = datetime(2026, 11, 6, 12, tzinfo=UTC)


def test_github_issue_from_api_maps_all_fields_and_null_body():
    issue = GithubIssue.from_api(
        {
            "number": 7,
            "node_id": "I_kwDOexample",
            "title": "Example issue",
            "body": None,
            "state": "open",
            "labels": [{"name": "agent-team"}, {"name": "bug"}],
            "user": {"login": "author"},
            "locked": False,
            "pull_request": {"url": "https://example.invalid/pr"},
            "created_at": NOW.isoformat(),
            "updated_at": NOW.isoformat(),
        }
    )

    assert issue.model_dump() == {
        "number": 7,
        "node_id": "I_kwDOexample",
        "title": "Example issue",
        "body": "",
        "state": "open",
        "labels": ["agent-team", "bug"],
        "author_login": "author",
        "locked": False,
        "is_pull_request": True,
        "created_at": NOW,
        "updated_at": NOW,
    }


def test_github_outcome_validates_retry_and_error_fields():
    limited = GithubOutcome(
        outcome="rate_limited",
        decided_at=NOW,
        retry_in=30,
        scope="secondary",
    )
    failed = GithubOutcome(
        outcome="unauthorized",
        decided_at=NOW,
        http_status=401,
        error_code="github_unauthorized",
        reason="GitHub rejected the token",
    )
    ok = GithubOutcome(outcome="ok", decided_at=NOW, http_status=200)

    assert limited.retry_in == 30
    assert failed.error_code == "github_unauthorized"
    assert ok.outcome == "ok"
    with pytest.raises(ValidationError):
        GithubOutcome(outcome="rate_limited", decided_at=NOW)
    with pytest.raises(ValidationError):
        GithubOutcome(outcome="forbidden", decided_at=NOW)
    with pytest.raises(ValidationError):
        GithubOutcome(outcome="ok", decided_at=NOW, retry_in=30, scope="secondary")


def test_remaining_github_contracts_have_declared_fields():
    repo = RepoInfo(
        id=100,
        full_name="bot/fork",
        fork=True,
        owner_login="bot",
        has_issues=True,
    )
    event = LabelEvent(event="labeled", label_name="agent-team", created_at=NOW)
    comment = GithubComment(id=12, body="ack", author_login="agent-bot", created_at=NOW)
    issue = GithubIssue(
        number=7,
        node_id="I_kwDOexample",
        title="Example issue",
        body="body",
        state="open",
        labels=["agent-team"],
        author_login="author",
        locked=False,
        is_pull_request=False,
        created_at=NOW,
        updated_at=NOW,
    )
    poll = PollResult(issues=[issue], pages=1, truncated=False, max_updated_at=NOW)
    view = IssueView(
        issue_key="gh-100-7", number=7, state="pending_start", start_attempts=0
    )
    cycle = CycleSummary(ran_at=NOW)

    assert repo.owner_login == "bot"
    assert event.label_name == "agent-team"
    assert comment.id == 12
    assert poll.issues == [issue]
    assert view.state == "pending_start"
    assert cycle.comments_posted == 0
