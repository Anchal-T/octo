from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from agentteam_common import workflow_key
from agentteam_contracts import (
    CommitStatusRequest,
    EvidenceCommentRequest,
    GithubWriteOutcome,
    OpenPrRequest,
    RepoHeadResponse,
)

from .client import GithubClient, GithubFailure
from .rules import (
    parse_evidence_marker,
    render_evidence_comment,
    validate_agent_branch,
)
from .schemas import Clock
from .settings import GithubSettings
from .store import PullRequestRow, Store


def _request_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _map_github_failure(error: GithubFailure, now: datetime) -> GithubWriteOutcome:
    out = error.outcome
    if out.outcome == "rate_limited":
        return GithubWriteOutcome(
            outcome="throttled",
            decided_at=now,
            retry_in=out.retry_in if out.retry_in is not None else 5.0,
            retry_at=out.retry_at,
        )
    if out.outcome == "unavailable":
        return GithubWriteOutcome(
            outcome="throttled",
            decided_at=now,
            retry_in=5.0,
        )
    return GithubWriteOutcome(
        outcome="failed",
        decided_at=now,
        error_code=out.error_code or out.outcome,
        reason=out.reason or "GitHub rejected the write request",
    )


def _utc_now(clock: Clock) -> datetime:
    now = clock()
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return now.astimezone(UTC)


async def fetch_repo_head(github: GithubClient) -> RepoHeadResponse:
    repo = await github.get_repo()
    head_sha = await github.get_branch_head(repo.default_branch)
    return RepoHeadResponse(
        repo=repo.full_name,
        default_branch=repo.default_branch,
        head_sha=head_sha,
    )


async def open_draft_pull_request(
    *,
    settings: GithubSettings,
    github: GithubClient,
    store: Store,
    clock: Clock,
    request: OpenPrRequest,
) -> GithubWriteOutcome:
    now = _utc_now(clock)
    body_no_key = request.model_dump(mode="json", exclude={"idempotency_key"})
    req_hash = _request_hash(body_no_key)
    cached = store.get_write_log(
        request.idempotency_key, endpoint="POST /v1/prs", request_sha256=req_hash
    )
    if cached is not None:
        return GithubWriteOutcome.model_validate_json(cached)

    if github.auth_mode != "app":
        return GithubWriteOutcome(
            outcome="failed",
            decided_at=now,
            error_code="app_auth_required",
            reason="PR authoring requires GitHub App installation token authentication",
        )

    try:
        branch = validate_agent_branch(f"agent/{request.issue_id}")
    except ValueError as error:
        return GithubWriteOutcome(
            outcome="failed",
            decided_at=now,
            error_code="invalid_branch",
            reason=str(error),
        )

    try:
        self_login = await github.get_self()
        repo = await github.get_repo()
        if not self_login.endswith("[bot]") or (
            self_login.casefold() == repo.owner_login.casefold()
        ):
            return GithubWriteOutcome(
                outcome="failed",
                decided_at=now,
                error_code="self_author_forbidden",
                reason="PR author must be the GitHub App bot, not the fork owner",
            )
        base_branch = request.base_branch or repo.default_branch
        approvers = [
            login
            for login in (settings.approvers or (repo.owner_login,))
            if login.casefold() != self_login.casefold()
        ]
        run_key = workflow_key(request.issue_id, request.attempt)

        tree_sha = await github.create_tree(
            base_tree=request.base_commit,
            files=request.files,
        )
        commit_sha = await github.create_commit(
            message=f"fix: {request.title}\n\nAgent Team run {run_key}",
            tree_sha=tree_sha,
            parents=[request.base_commit],
        )
        await github.upsert_branch_ref(branch, commit_sha)

        existing_pr = await github.find_open_pr(head_branch=branch)
        if existing_pr is None:
            pr = await github.create_draft_pr(
                title=request.title,
                body=request.body,
                head_branch=branch,
                base_branch=base_branch,
            )
            previous_row = None
        else:
            pr = existing_pr.model_copy(update={"head_sha": commit_sha})
            previous_row = store.get_pull_request(pr.number)

        need_review_request = (
            previous_row is None
            or previous_row.head_sha != commit_sha
            or previous_row.review_requested_sha != commit_sha
        )
        if need_review_request and approvers:
            await github.request_reviewers(pr.number, approvers)

        now_iso = now.isoformat()
        created_at = previous_row.created_at if previous_row is not None else now_iso
        store.upsert_pull_request(
            PullRequestRow(
                pr_number=pr.number,
                issue_key=request.issue_id,
                attempt=request.attempt,
                branch=branch,
                base_branch=base_branch,
                head_sha=commit_sha,
                html_url=pr.html_url,
                author_login=pr.author_login or self_login,
                state="open",
                review_requested_sha=commit_sha if approvers else None,
                decision="pending",
                decided_by=None,
                decision_reason=None,
                decided_review_id=None,
                decided_head_sha=None,
                decided_at=None,
                delivered_at=None,
                created_at=created_at,
                updated_at=now_iso,
            )
        )
    except GithubFailure as error:
        return _map_github_failure(error, _utc_now(clock))

    outcome = GithubWriteOutcome(
        outcome="ok",
        decided_at=_utc_now(clock),
        pr_number=pr.number,
        html_url=pr.html_url,
        head_sha=commit_sha,
        branch=branch,
    )
    store.record_write_log(
        request.idempotency_key,
        endpoint="POST /v1/prs",
        request_sha256=req_hash,
        response_json=outcome.model_dump_json(),
        created_at=outcome.decided_at,
    )
    return outcome


async def upsert_pr_evidence_comment(
    *,
    github: GithubClient,
    store: Store,
    clock: Clock,
    pr_number: int,
    request: EvidenceCommentRequest,
) -> GithubWriteOutcome:
    endpoint = f"PUT /v1/prs/{pr_number}/evidence"
    body_no_key = request.model_dump(mode="json", exclude={"idempotency_key"})
    req_hash = _request_hash(body_no_key)
    cached = store.get_write_log(
        request.idempotency_key, endpoint=endpoint, request_sha256=req_hash
    )
    if cached is not None:
        return GithubWriteOutcome.model_validate_json(cached)

    run_key = workflow_key(request.issue_id, request.attempt)
    rendered = render_evidence_comment(request, workflow_run_key=run_key)
    try:
        self_login = await github.get_self()
        comments = await github.list_comments(pr_number)
        existing = next(
            (
                comment
                for comment in comments
                if comment.author_login.casefold() == self_login.casefold()
                and parse_evidence_marker(comment.body) == run_key
            ),
            None,
        )
        if existing is not None:
            comment_id = await github.update_comment(existing.id, rendered)
        else:
            comment_id = await github.post_comment(pr_number, rendered)
    except GithubFailure as error:
        return _map_github_failure(error, _utc_now(clock))

    outcome = GithubWriteOutcome(
        outcome="ok",
        decided_at=_utc_now(clock),
        pr_number=pr_number,
        head_sha=request.head_sha,
        comment_id=comment_id,
    )
    store.record_write_log(
        request.idempotency_key,
        endpoint=endpoint,
        request_sha256=req_hash,
        response_json=outcome.model_dump_json(),
        created_at=outcome.decided_at,
    )
    return outcome


async def set_commit_status(
    *,
    github: GithubClient,
    store: Store,
    clock: Clock,
    sha: str,
    request: CommitStatusRequest,
) -> GithubWriteOutcome:
    endpoint = f"POST /v1/commits/{sha}/status"
    body_no_key = request.model_dump(mode="json", exclude={"idempotency_key"})
    req_hash = _request_hash(body_no_key)
    cached = store.get_write_log(
        request.idempotency_key, endpoint=endpoint, request_sha256=req_hash
    )
    if cached is not None:
        return GithubWriteOutcome.model_validate_json(cached)

    try:
        await github.create_commit_status(
            sha,
            state=request.state,
            description=request.description,
            context=request.context,
        )
    except GithubFailure as error:
        return _map_github_failure(error, _utc_now(clock))

    outcome = GithubWriteOutcome(
        outcome="ok",
        decided_at=_utc_now(clock),
        head_sha=sha,
    )
    store.record_write_log(
        request.idempotency_key,
        endpoint=endpoint,
        request_sha256=req_hash,
        response_json=outcome.model_dump_json(),
        created_at=outcome.decided_at,
    )
    return outcome
