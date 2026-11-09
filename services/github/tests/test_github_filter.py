from datetime import UTC, datetime, timedelta

from agentteam_contracts import IssueRunRequest

from services.github.app.rules import (
    admit,
    decide,
    fingerprint,
    issue_key,
    marker,
    parse_marker,
    render_comment,
    to_run_request,
)
from services.github.app.schemas import GithubIssue
from services.github.app.store import IssueRow

NOW = datetime(2026, 11, 6, 12, tzinfo=UTC)
BASELINE = NOW - timedelta(days=1)


def make_issue(**changes):
    fields = {
        "number": 7,
        "node_id": "I_7",
        "title": "Example issue",
        "body": "Example body",
        "state": "open",
        "labels": ["agent-team"],
        "author_login": "maintainer",
        "locked": False,
        "is_pull_request": False,
        "created_at": NOW,
        "updated_at": NOW,
    }
    fields.update(changes)
    return GithubIssue(**fields)


def make_row(number, created_at):
    return IssueRow(
        issue_key=issue_key(100, number),
        number=number,
        state="pending_start",
        reason=None,
        deferred_reason=None,
        fingerprint=f"fp-{number}",
        github_updated_at=NOW.isoformat(),
        created_at=created_at.isoformat(),
        start_attempts=0,
        admitted_at=None,
        submitted_at=None,
        finished_at=None,
        result_status=None,
    )


def test_pull_requests_are_ignored():
    decision = decide(
        make_issue(is_pull_request=True),
        self_login="agent-bot",
        baseline_at=BASELINE,
        opted_in_after_baseline=None,
    )

    assert (decision.state, decision.reason, decision.comment_kind) == (
        "ignored",
        "pull_request",
        None,
    )


def test_closed_and_unlabelled_stored_issues_are_cancelled():
    for issue in (
        make_issue(state="closed"),
        make_issue(labels=["bug"]),
    ):
        decision = decide(
            issue,
            self_login="agent-bot",
            baseline_at=BASELINE,
            opted_in_after_baseline=None,
        )
        assert decision.state == "cancelled"
        assert decision.comment_kind is None


def test_own_issues_are_silently_skipped():
    decision = decide(
        make_issue(author_login="AGENT-BOT"),
        self_login="agent-bot",
        baseline_at=BASELINE,
        opted_in_after_baseline=None,
    )

    assert decision.state == "skipped"
    assert decision.reason == "own_issue"
    assert decision.comment_kind is None


def test_own_issues_are_processed_when_enabled():
    decision = decide(
        make_issue(author_login="AGENT-BOT"),
        self_login="agent-bot",
        baseline_at=BASELINE,
        opted_in_after_baseline=None,
        process_own_issues=True,
    )

    assert decision.state == "pending_start"
    assert decision.comment_kind == "ack"


def test_enabling_own_issues_keeps_the_other_gates():
    for issue, state in (
        (make_issue(author_login="agent-bot", locked=True), "skipped"),
        (make_issue(author_login="agent-bot", labels=["bug"]), "cancelled"),
        (make_issue(author_login="agent-bot", is_pull_request=True), "ignored"),
        (
            make_issue(
                author_login="agent-bot", created_at=BASELINE - timedelta(seconds=1)
            ),
            None,
        ),
    ):
        decision = decide(
            issue,
            self_login="agent-bot",
            baseline_at=BASELINE,
            opted_in_after_baseline=None,
            process_own_issues=True,
        )
        assert decision.state == state


def test_locked_issues_are_silently_skipped():
    decision = decide(
        make_issue(locked=True),
        self_login="agent-bot",
        baseline_at=BASELINE,
        opted_in_after_baseline=None,
    )

    assert decision.state == "skipped"
    assert decision.reason == "locked"
    assert decision.comment_kind is None


def test_baseline_issue_requires_event_lookup_then_ignores_or_opts_in():
    old_issue = make_issue(created_at=BASELINE - timedelta(seconds=1))
    pending = decide(
        old_issue,
        self_login="agent-bot",
        baseline_at=BASELINE,
        opted_in_after_baseline=None,
    )
    ignored = decide(
        old_issue,
        self_login="agent-bot",
        baseline_at=BASELINE,
        opted_in_after_baseline=False,
    )
    opted_in = decide(
        old_issue,
        self_login="agent-bot",
        baseline_at=BASELINE,
        opted_in_after_baseline=True,
    )

    assert pending.needs_label_events
    assert pending.state is None
    assert ignored.state == "ignored"
    assert ignored.reason == "baseline"
    assert opted_in.state == "pending_start"
    assert opted_in.comment_kind == "ack"


def test_empty_title_is_skipped_with_a_comment():
    decision = decide(
        make_issue(title="  \n"),
        self_login="agent-bot",
        baseline_at=BASELINE,
        opted_in_after_baseline=None,
    )

    assert decision.state == "skipped"
    assert decision.reason == "empty_title"
    assert decision.comment_kind == "skip"


def test_eligible_issue_is_pending_and_fingerprints_content_only():
    issue = make_issue()
    decision = decide(
        issue,
        self_login="agent-bot",
        baseline_at=BASELINE,
        opted_in_after_baseline=None,
    )

    assert decision.state == "pending_start"
    assert decision.comment_kind == "ack"
    assert fingerprint(issue) == fingerprint(
        issue.model_copy(update={"updated_at": NOW + timedelta(days=1)})
    )
    assert fingerprint(issue) == fingerprint(
        issue.model_copy(update={"labels": list(reversed(issue.labels))})
    )
    assert issue_key(100, 7) == "gh-100-7"


def test_run_request_truncates_at_cap_and_preserves_issue_key():
    issue = make_issue(body="x" * 20_000)

    request = to_run_request(issue, "gh-100-7", 8000, repo="bot/fork")

    assert isinstance(request, IssueRunRequest)
    assert request.issue_id == "gh-100-7"
    assert request.repo == "bot/fork"
    assert request.body == (
        "x" * 8000
        + "\n\n[Agent Team truncated this issue body; the original had 20000 characters]"
    )


def test_admission_obeys_cycle_day_caps_and_created_order():
    rows = [
        make_row(1, NOW - timedelta(minutes=2)),
        make_row(2, NOW - timedelta(minutes=1)),
        make_row(3, NOW),
    ]

    cycle = admit(
        rows,
        started_this_cycle=0,
        started_today=0,
        per_cycle=1,
        per_day=10,
    )
    daily = admit(
        rows,
        started_this_cycle=0,
        started_today=2,
        per_cycle=1,
        per_day=2,
    )

    assert [row.number for row in cycle.starts] == [1]
    assert [row.number for row in cycle.deferred_cycle] == [2, 3]
    assert daily.starts == ()
    assert [row.number for row in daily.deferred_day] == [1, 2, 3]


def test_comment_markers_and_templates_are_fixed_and_parseable():
    key = "gh-100-7"
    fp = "abcdef0123456789"
    rendered = render_comment(
        "ack",
        key=key,
        fp=fp,
        detail={"at": "2026-11-06T12:00:00Z", "truncated": "true"},
    )
    expected_marker = "<!-- agent-team:1 kind=ack key=gh-100-7 fp=abcdef01 -->"

    assert rendered.startswith(expected_marker)
    assert "has started a triage run" in rendered
    assert "body was truncated" in rendered
    assert parse_marker(rendered) == ("ack", key, "abcdef01")
    assert parse_marker("prefix " + rendered) is None
    assert marker("result", key, fp) == (
        "<!-- agent-team:1 kind=result key=gh-100-7 fp=abcdef01 -->"
    )


def test_comment_templates_cover_queued_skip_failure_and_result():
    queued = render_comment(
        "queued",
        key="gh-100-7",
        fp="a" * 64,
        detail={"reason": "daily_limit", "limit": "10"},
    )
    skipped = render_comment(
        "skip", key="gh-100-7", fp="b" * 64, detail={"reason": "empty_title"}
    )
    failed = render_comment(
        "failure",
        key="gh-100-7",
        fp="c" * 64,
        detail={"reason": "workflow_unavailable"},
    )
    result = render_comment(
        "result",
        key="gh-100-7",
        fp="d" * 64,
        detail={"status": "quota_exhausted", "retry_at": "2026-11-07T00:00:00Z"},
    )

    assert "daily limit of 10" in queued
    assert "title is empty" in skipped
    assert "unavailable" in failed
    assert "quota is exhausted" in result
    assert "2026-11-07T00:00:00Z" in result


def test_ack_comment_puts_the_time_in_its_own_sentence():
    plain = render_comment(
        "ack",
        key="gh-100-7",
        fp="a" * 64,
        detail={"at": "2026-11-06T12:00:00Z"},
    )
    truncated = render_comment(
        "ack",
        key="gh-100-7",
        fp="a" * 64,
        detail={
            "at": "2026-11-06T12:00:00Z",
            "truncated": "true",
            "body_max_chars": "8000",
        },
    )
    untimed = render_comment("ack", key="gh-100-7", fp="a" * 64, detail={})

    assert plain.endswith(
        "I will post the result here. Started at 2026-11-06T12:00:00Z."
    )
    assert truncated.endswith(
        "truncated to 8000 characters. Started at 2026-11-06T12:00:00Z."
    )
    assert untimed.endswith("I will post the result here.")
    for text in (plain, truncated, untimed):
        assert ". at " not in text
        assert ".." not in text


def test_committed_result_renders_triage_as_a_model_suggestion():
    rendered = render_comment(
        "result",
        key="gh-100-7",
        fp="d" * 64,
        detail={
            "at": "2026-11-09T10:00:00Z",
            "status": "committed",
            "model": "openai/gpt-oss-20b",
            "provider": "groq",
            "triage_kind": "bug",
            "triage_priority": "medium",
            "triage_summary": "Example summary.",
        },
    )

    assert parse_marker(rendered) == ("result", "gh-100-7", "dddddddd")
    assert "Model suggestion (no labels or other changes were made)" in rendered
    assert "- Type: bug\n- Priority: medium\n- Summary: Example summary.\n" in rendered
    assert rendered.endswith(
        "\n\nModel: `openai/gpt-oss-20b` (groq). Finished at 2026-11-09T10:00:00Z."
    )
    assert "added" not in rendered.lower()


def test_unparseable_and_missing_triage_have_their_own_wording():
    unparseable = render_comment(
        "result",
        key="gh-100-7",
        fp="d" * 64,
        detail={
            "status": "committed",
            "triage_error": "unparseable",
            "model": "openai/gpt-oss-20b",
            "provider": "groq",
        },
    )
    missing = render_comment(
        "result",
        key="gh-100-7",
        fp="d" * 64,
        detail={"status": "committed", "triage_missing": "true"},
    )

    assert "reply could not be read, so no triage is shown" in unparseable
    assert "Model: `openai/gpt-oss-20b` (groq)." in unparseable
    assert "triage details are no longer available" in missing


def test_model_summary_is_sanitised_before_posting():
    hostile = (
        "Ping @octocat about #12.\n\n<!-- agent-team:1 kind=result key=x fp=y -->"
        " `code` <b>bold</b> " + "x" * 400
    )
    rendered = render_comment(
        "result",
        key="gh-100-7",
        fp="d" * 64,
        detail={
            "status": "committed",
            "triage_kind": "bug",
            "triage_priority": "low",
            "triage_summary": hostile,
        },
    )
    summary_line = next(
        line for line in rendered.splitlines() if line.startswith("- Summary: ")
    )

    assert "@octocat" not in rendered
    assert "@\u200bocto" in summary_line
    assert "#12" not in summary_line
    assert "<!--" not in summary_line
    assert "<b>" not in summary_line
    assert "`" not in summary_line
    assert rendered.count("<!-- agent-team:1") == 1
    assert len(summary_line) <= len("- Summary: ") + 300
