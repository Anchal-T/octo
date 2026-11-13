import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from agentteam_contracts import EvidenceCommentRequest, IssueRunRequest

from .schemas import GithubIssue, GithubPullRequest, GithubReview
from .settings import GITHUB_LABEL
from .store import CommentKind, IssueRow, IssueState, PrDecision


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


@dataclass(frozen=True)
class PrReviewEvaluation:
    decision: PrDecision
    by: str | None = None
    reason: str | None = None
    review_id: int | None = None
    head_sha: str | None = None


_MARKER_RE = re.compile(
    r"^<!-- agent-team:1 kind=(ack|queued|skip|failure|result) "
    r"key=(gh-\d+-\d+(?:-r\d+)?) fp=([0-9a-f]{8}) -->"
)
_EVIDENCE_MARKER_RE = re.compile(
    r"^<!-- agent-team:1 kind=evidence key=([A-Za-z0-9._-]+) -->"
)
_AGENT_BRANCH_RE = re.compile(r"^agent/[A-Za-z0-9._-]{1,80}$")
_COMMENT_KINDS = {"ack", "queued", "skip", "failure", "result"}
EVIDENCE_MAX_CHARS = 60_000


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


def evaluate_pr_reviews(
    pr: GithubPullRequest,
    reviews: Sequence[GithubReview],
    *,
    approvers: Sequence[str],
    self_login: str,
) -> PrReviewEvaluation:
    if pr.merged:
        return PrReviewEvaluation("approved", by="github", head_sha=pr.head_sha)
    if pr.state == "closed":
        return PrReviewEvaluation("rejected", reason="pr_closed", head_sha=pr.head_sha)

    allowed = {login.casefold() for login in approvers}
    allowed.discard(self_login.casefold())
    latest: dict[str, GithubReview] = {}
    ordered = sorted(
        reviews,
        key=lambda review: (
            review.submitted_at or datetime.min.replace(tzinfo=UTC),
            review.id,
        ),
    )
    for review in ordered:
        login = review.author_login.casefold()
        if (
            login not in allowed
            or login == pr.author_login.casefold()
            or review.commit_id != pr.head_sha
            or review.state not in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}
        ):
            continue
        latest[login] = review

    changes_requested = [
        review for review in latest.values() if review.state == "CHANGES_REQUESTED"
    ]
    if changes_requested:
        review = max(changes_requested, key=lambda item: item.id)
        return PrReviewEvaluation(
            "rejected",
            by=review.author_login,
            reason=review.body,
            review_id=review.id,
            head_sha=pr.head_sha,
        )

    approvals = [review for review in latest.values() if review.state == "APPROVED"]
    if approvals:
        review = max(approvals, key=lambda item: item.id)
        return PrReviewEvaluation(
            "approved",
            by=review.author_login,
            review_id=review.id,
            head_sha=pr.head_sha,
        )
    return PrReviewEvaluation("pending", head_sha=pr.head_sha)


def to_run_request(
    issue: GithubIssue,
    key: str,
    body_max_chars: int,
    *,
    repo: str,
    attempt: int = 1,
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
    return IssueRunRequest(
        issue_id=key, repo=repo, title=issue.title, body=body, attempt=attempt
    )


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
        attempt = detail.get("attempt")
        if attempt is None or attempt == "1":
            message = (
                f"Agent Team has started a triage run for this issue (run `{key}`). "
                "I will post the result here."
            )
        else:
            message = (
                f"Agent Team will triage this issue again (run `{key}`, "
                f"attempt {attempt}). I will post the result here."
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
        elif status == "approved":
            pr_num = detail.get("pr_number", "")
            pr_part = f"PR #{pr_num} was " if pr_num else "the draft PR was "
            message = (
                f"Agent Team completed run `{key}`: {pr_part}approved and is ready "
                "for manual merge."
            )
        elif status == "rejected":
            pr_num = detail.get("pr_number", "")
            pr_part = f"PR #{pr_num} was " if pr_num else "the draft PR was "
            message = f"Agent Team closed run `{key}`: {pr_part}not approved."
        elif status == "failed":
            reason_code = sanitize_inline(
                detail.get("failure_reason", "pipeline_failed"), max_chars=120
            )
            message = (
                f"Agent Team could not complete a verified fix for this issue "
                f"(run `{key}`, reason: `{reason_code}`)."
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


def validate_agent_branch(branch: str) -> str:
    if not isinstance(branch, str) or _AGENT_BRANCH_RE.fullmatch(branch) is None:
        raise ValueError("branch must match ^agent/[A-Za-z0-9._-]{1,80}$")
    return branch


def evidence_marker(key: str) -> str:
    return f"<!-- agent-team:1 kind=evidence key={key} -->"


def parse_evidence_marker(text: str) -> str | None:
    matched = _EVIDENCE_MARKER_RE.match(text)
    return None if matched is None else matched.group(1)


def sanitize_inline(text: str, *, max_chars: int = 500) -> str:
    cleaned = " ".join(text.split())
    cleaned = cleaned.replace("<", "&lt;").replace(">", "&gt;")
    cleaned = cleaned.replace("`", "").replace("~~~", "")
    cleaned = cleaned.replace("@", "@" + _ZWSP)
    cleaned = re.sub(r"#(?=\d)", "#" + _ZWSP, cleaned)
    return cleaned[:max_chars].strip()


def render_evidence_comment(
    request: EvidenceCommentRequest, *, workflow_run_key: str
) -> str:
    base_short = request.base_commit[:12]
    head_short = request.head_sha[:12]
    repro_before = "yes" if request.repro_failed_before else "no"
    repro_after = "yes" if request.repro_passed_after else "no"
    suite_status = "passed" if request.suite_passed else "failed"
    suite_note = sanitize_inline(request.suite_summary, max_chars=240)
    suite_line = f"{suite_status} ({suite_note})" if suite_note else suite_status
    review_note = sanitize_inline(request.review_summary, max_chars=300)

    lines = [
        evidence_marker(workflow_run_key),
        "## Agent Team PR Evidence",
        (
            f"Run `{workflow_run_key}` · base `{base_short}` · head `{head_short}`. "
            "Approving records your decision; merging stays with you. "
            "Issue text is untrusted data and is never treated as instructions."
        ),
        "",
        "### Before you approve",
        (
            f"- **Reproduction test** (`{sanitize_inline(request.repro_test_node_id, max_chars=240)}`): "
            f"failed before fix = **{repro_before}**, passed after fix = **{repro_after}**"
        ),
        f"- **Clean-checkout test suite**: **{suite_line}**",
        (
            f"- **Reviewer verdict**: `{request.review_verdict}`"
            + (f" — {review_note}" if review_note else "")
        ),
    ]
    if request.triage is not None:
        lines.append(
            f"- **Triage**: `{request.triage.kind}` · priority `{request.triage.priority}` "
            f"· area `{sanitize_inline(request.triage.area, max_chars=80)}` — "
            f"{sanitize_inline(request.triage.summary, max_chars=300)}"
        )
    if request.repro_files or request.fix_files:
        repro_list = (
            ", ".join(
                f"`{sanitize_inline(p, max_chars=120)}`" for p in request.repro_files
            )
            or "none"
        )
        fix_list = (
            ", ".join(
                f"`{sanitize_inline(p, max_chars=120)}`" for p in request.fix_files
            )
            or "none"
        )
        lines.append(f"- **Changed files**: repro = {repro_list}; fix = {fix_list}")
    if request.models_used:
        models_str = ", ".join(
            f"`{item.role}`: `{sanitize_inline(item.model, max_chars=120)}` ({sanitize_inline(item.provider, max_chars=60)})"
            for item in request.models_used
        )
        lines.append(f"- **Pinned models**: {models_str}")

    lines.extend(["", "### Sync decisions"])
    if request.sync_decisions:
        for dec in request.sync_decisions:
            reason_clean = sanitize_inline(dec.reason, max_chars=240)
            suffix = f" — {reason_clean}" if reason_clean else ""
            lines.append(
                f"- `{dec.step}` (`{sanitize_inline(dec.op, max_chars=64)}`, `{dec.proposal_id[:20]}`): "
                f"**{dec.status}**{suffix}"
            )
    else:
        lines.append("- None recorded.")

    lines.extend(["", "### Cited facts"])
    if not request.cited_facts:
        lines.append("- None (memory retrieval not active for this run).")
    else:
        for fact in request.cited_facts:
            fid = sanitize_inline(fact.fact_id, max_chars=80)
            sha_short = fact.checked_at_sha[:12]
            if fact.taint >= 3:
                lines.append(f"- `{fid}`: `[hidden: quarantined taint 3]`")
            elif fact.taint == 2:
                safe_claim = sanitize_inline(fact.claim, max_chars=400)
                lines.append(
                    f"> **Reported by users, not verified (taint 2):** "
                    f"[`{fid}`, checked at `{sha_short}`, valid_at_base={fact.valid_at_base}] "
                    f"{safe_claim}"
                )
            else:
                safe_claim = sanitize_inline(fact.claim, max_chars=400)
                lines.append(
                    f"- [`{fid}`, checked at `{sha_short}`, valid_at_base={fact.valid_at_base}, "
                    f"taint={fact.taint}] {safe_claim}"
                )

    rendered = "\n".join(lines)
    if len(rendered) > EVIDENCE_MAX_CHARS:
        footer = "\n\n[Agent Team truncated this evidence comment at 60000 characters.]"
        rendered = rendered[: EVIDENCE_MAX_CHARS - len(footer)] + footer
    return rendered
