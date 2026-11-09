import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from agentteam_contracts import IssueRunRequest

from .schemas import GithubIssue
from .settings import GITHUB_LABEL
from .store import CommentKind, IssueRow, IssueState


@dataclass(frozen=True)
class Decision:
    state: IssueState | None
    reason: str | None
    comment_kind: CommentKind | None
    needs_label_events: bool = False


@dataclass(frozen=True)
class AdmissionPlan:
    starts: tuple[IssueRow, ...]
    deferred_cycle: tuple[IssueRow, ...]
    deferred_day: tuple[IssueRow, ...]


_MARKER_RE = re.compile(
    r"^<!-- agent-team:1 kind=(ack|queued|skip|failure|result) "
    r"key=(gh-\d+-\d+) fp=([0-9a-f]{8}) -->"
)
_COMMENT_KINDS = {"ack", "queued", "skip", "failure", "result"}


def fingerprint(issue: GithubIssue) -> str:
    payload = json.dumps(
        [issue.title, issue.body, sorted(issue.labels)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def issue_key(repo_id: int, number: int) -> str:
    return f"gh-{repo_id}-{number}"


def decide(
    issue: GithubIssue,
    *,
    self_login: str,
    baseline_at: datetime,
    opted_in_after_baseline: bool | None,
    process_own_issues: bool = False,
) -> Decision:
    if issue.is_pull_request:
        return Decision("ignored", "pull_request", None)
    if issue.state == "closed" or GITHUB_LABEL not in issue.labels:
        return Decision("cancelled", "closed_or_unlabelled", None)
    if (
        not process_own_issues
        and issue.author_login.casefold() == self_login.casefold()
    ):
        return Decision("skipped", "own_issue", None)
    if issue.locked:
        return Decision("skipped", "locked", None)
    if issue.created_at < baseline_at:
        if opted_in_after_baseline is None:
            return Decision(None, None, None, needs_label_events=True)
        if not opted_in_after_baseline:
            return Decision("ignored", "baseline", None)
    if not issue.title.strip():
        return Decision("skipped", "empty_title", "skip")
    return Decision("pending_start", None, "ack")


def admit(
    pending: Sequence[IssueRow],
    *,
    started_this_cycle: int,
    started_today: int,
    per_cycle: int,
    per_day: int,
) -> AdmissionPlan:
    cycle_left = max(0, per_cycle - started_this_cycle)
    day_left = max(0, per_day - started_today)
    starts: list[IssueRow] = []
    deferred_cycle: list[IssueRow] = []
    deferred_day: list[IssueRow] = []

    for issue in sorted(pending, key=lambda row: (row.created_at, row.issue_key)):
        if day_left == 0:
            deferred_day.append(issue)
        elif cycle_left == 0:
            deferred_cycle.append(issue)
        else:
            starts.append(issue)
            cycle_left -= 1
            day_left -= 1

    return AdmissionPlan(
        starts=tuple(starts),
        deferred_cycle=tuple(deferred_cycle),
        deferred_day=tuple(deferred_day),
    )


def to_run_request(
    issue: GithubIssue, key: str, body_max_chars: int, *, repo: str
) -> IssueRunRequest:
    if body_max_chars < 0:
        raise ValueError("body_max_chars must not be negative")
    body = issue.body
    if len(body) > body_max_chars:
        original_length = len(body)
        body = (
            body[:body_max_chars] + "\n\n[Agent Team truncated this issue body; "
            f"the original had {original_length} characters]"
        )
    return IssueRunRequest(issue_id=key, repo=repo, title=issue.title, body=body)


def marker(kind: str, key: str, fp: str) -> str:
    if kind not in _COMMENT_KINDS:
        raise ValueError(f"unsupported comment kind: {kind}")
    return f"<!-- agent-team:1 kind={kind} key={key} fp={fp[:8]} -->"


def parse_marker(text: str) -> tuple[str, str, str] | None:
    match = _MARKER_RE.match(text)
    return None if match is None else match.groups()


def render_comment(
    kind: str,
    *,
    key: str,
    fp: str,
    detail: Mapping[str, str],
) -> str:
    stamp = detail.get("at")
    if kind == "ack":
        message = (
            f"Agent Team has started a triage run for this issue (run `{key}`). "
            "I will post the result here."
        )
        if detail.get("truncated") == "true":
            limit = detail.get("body_max_chars", "the configured limit")
            message += f" The original issue body was truncated to {limit} characters."
        if stamp:
            message += f" Started at {stamp}."
    elif kind == "queued":
        if detail.get("reason") == "daily_limit":
            limit = detail.get("limit", "10")
            message = (
                f"Queued: the daily limit of {limit} runs has been reached. "
                "This issue will start after the limit resets (about 00:00 UTC)."
            )
        else:
            message = (
                "Queued: this poll cycle's start limit has been reached. "
                "This issue will be reconsidered on the next poll."
            )
        if stamp:
            message += f" Checked at {stamp}."
    elif kind == "skip":
        reason = detail.get("reason")
        messages = {
            "empty_title": "its title is empty",
            "locked": "it is locked, so it cannot be commented on",
            "own_issue": "it was opened by the Agent Team bot",
        }
        message = f"Agent Team skipped this issue: {messages.get(reason, 'it is not eligible')}."
        if stamp:
            message += f" Checked at {stamp}."
    elif kind == "failure":
        reason = detail.get("reason")
        messages = {
            "workflow_rejected": "the workflow rejected the request",
            "workflow_conflict": "the workflow has a conflicting run",
            "workflow_unavailable": "the workflow service was unavailable",
        }
        explanation = messages.get(reason, "the workflow could not start the run")
        message = f"Agent Team could not start this issue: {explanation} (run `{key}`)."
        if stamp:
            message += f" Checked at {stamp}."
    elif kind == "result":
        status = detail.get("status")
        if status == "quota_exhausted":
            retry_at = detail.get("retry_at", "a later time")
            message = (
                f"Agent Team could not complete triage for this issue (run `{key}`) "
                f"because the model quota is exhausted. Try again after {retry_at}."
            )
        elif status == "throttled":
            retry_at = detail.get("retry_at")
            retry_in = detail.get("retry_in", "a short wait")
            retry = f"after {retry_at}" if retry_at else f"in {retry_in} seconds"
            message = (
                f"Agent Team could not complete triage for this issue (run `{key}`) "
                f"because the model was throttled. Try again {retry}."
            )
        else:
            message = _committed_result_message(key, detail)
        if stamp:
            message += f" Finished at {stamp}."
    else:
        raise ValueError(f"unsupported comment kind: {kind}")

    return f"{marker(kind, key, fp)}\n{message}"


SUMMARY_MAX_CHARS = 300
_ZWSP = "\u200b"


def _suggestion_area(detail: Mapping[str, str]) -> str:
    raw = detail.get("triage_area")
    if not raw or not raw.strip():
        return "unknown"
    return sanitize_summary(raw) or "unknown"


def sanitize_summary(text: str) -> str:
    """Make model-written text inert before it is posted to GitHub."""
    text = " ".join(text.split())
    text = text.replace("<", "&lt;").replace(">", "&gt;").replace("`", "")
    text = text.replace("@", "@" + _ZWSP)
    text = re.sub(r"#(?=\d)", "#" + _ZWSP, text)
    return text[:SUMMARY_MAX_CHARS].strip()


def _committed_result_message(key: str, detail: Mapping[str, str]) -> str:
    kind = detail.get("triage_kind")
    priority = detail.get("triage_priority")
    summary = sanitize_summary(detail.get("triage_summary", ""))
    if kind and priority and summary:
        message = (
            f"Agent Team completed triage for this issue (run `{key}`). "
            "Model suggestion (no labels or other changes were made):\n"
            f"- Type: {kind}\n- Priority: {priority}\n- Summary: {summary}\n"
            f"- Area: {_suggestion_area(detail)}\n\n"
        )
    elif detail.get("triage_error") == "unparseable":
        message = (
            f"Agent Team completed the triage call for this issue (run `{key}`), "
            "but the model's reply could not be read, so no triage is shown."
        )
    elif detail.get("triage_missing") == "true":
        message = (
            f"Agent Team completed triage for this issue (run `{key}`). "
            "The triage details are no longer available."
        )
    else:
        message = f"Agent Team completed triage for this issue (run `{key}`)."
    model = detail.get("model")
    if model:
        provider = detail.get("provider")
        separator = "" if message.endswith("\n") else " "
        message += f"{separator}Model: `{model}`"
        message += f" ({provider})." if provider else "."
    return message
