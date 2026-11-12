from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import restate
from agentteam_common import Settings, load_agentteam_config, stable_id, workflow_key
from agentteam_contracts import (
    ApprovalDecision,
    ApprovalResult,
    CommitStatusRequest,
    EventWrite,
    EvidenceCommentRequest,
    GithubWriteOutcome,
    IssueRunRequestV2,
    IssueRunResultV2,
    ModelCallRequest,
    ModelOutcome,
    ModelUsageItem,
    OpenPrRequest,
    ProposalWrite,
    RepoHeadResponse,
    RoundRecord,
    RoundWrite,
    RunStatusV2,
    RunWrite,
    Sampling,
    SyncDecisionItem,
)

from .models import Ledger, ModelClient, Store
from .models.pools import pin_for
from .retries import WaitDecision, decide_delay, decide_wait
from .roles import (
    SandboxSessionManager,
    run_fixer_step,
    run_reproducer_step,
    run_tester_step,
)
from .sandbox import DockerSandbox, Sandbox
from .triage import parse_triage, triage_messages

settings = Settings.from_env()
agentteam_config = load_agentteam_config()
issue_run_v2 = restate.Workflow("IssueRunV2", journal_retention=timedelta(days=7))
_model_client: ModelClient | None = None


def current_basis() -> str:
    return datetime.now(UTC).isoformat()


def call_model(request: ModelCallRequest) -> ModelOutcome:
    global _model_client
    if _model_client is None:
        store = Store(settings)
        _model_client = ModelClient(
            settings=settings,
            ledger=Ledger(settings, store),
            store=store,
            http=httpx.Client(),
            now=lambda: datetime.now(UTC),
        )
    return _model_client.call(request)


async def post_memory(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=settings.memory_url, timeout=15) as client:
        response = await client.post(path, json=payload)
        response.raise_for_status()
        return response.json()


async def call_github_api(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=settings.github_url, timeout=30) as client:
        response = await client.request(method, path, json=payload)
        response.raise_for_status()
        return response.json()


def make_sandbox(base_commit: str) -> Sandbox:
    repo_path = settings.sandbox_repo_path or settings.scip_repo_path
    if not repo_path:
        raise RuntimeError(
            "SANDBOX_REPO_PATH (or SCIP_REPO_PATH) must be set for DockerSandbox"
        )
    return DockerSandbox(
        image=settings.sandbox_image,
        repo_path=repo_path,
        base_commit=base_commit,
        network=settings.sandbox_network,
    )


def _key_prefix_v2(issue_id: str, attempt: int) -> str:
    if attempt == 1:
        return f"{issue_id}:v2"
    return f"{issue_id}:v2:r{attempt}"


class RestateStepJournal:
    def __init__(self, ctx: restate.WorkflowContext) -> None:
        self.ctx = ctx

    async def call_model(
        self, name: str, request: ModelCallRequest
    ) -> tuple[ModelOutcome, str]:
        provider = pin_for(request.role, settings).pool
        for call_num in (1, 2, 3):
            outcome = await self.ctx.run_typed(
                f"{name}:{call_num}",
                call_model,
                restate.RunOptions(type_hint=ModelOutcome, max_attempts=2),
                request=request,
            )
            if outcome.outcome == "completed":
                return outcome, provider
            if outcome.outcome in ("failed", "ledger_unavailable"):
                raise restate.TerminalError(
                    f"{request.role} {outcome.outcome}: {outcome.error_code or 'unspecified'}"
                )
            if outcome.outcome == "throttled":
                if outcome.retry_in is None:
                    raise restate.TerminalError(
                        f"{request.role} throttle is missing retry_in"
                    )
                if call_num == 3:
                    return outcome, provider
                wait = await self.ctx.run_typed(
                    f"{name}:throttle-wait:{call_num}",
                    decide_delay,
                    restate.RunOptions(type_hint=WaitDecision),
                    delay_s=outcome.retry_in,
                    cap_s=settings.triager_wait_cap_s,
                )
                if wait["action"] == "return":
                    return outcome, provider
                wait_s = wait["wait_s"]
                if wait_s is None:
                    raise restate.TerminalError("wait decision is missing wait_s")
                await self.ctx.sleep(
                    timedelta(seconds=wait_s),
                    name=f"{name}:throttle:{call_num}",
                )
                continue

            if outcome.retry_at is None:
                raise restate.TerminalError(
                    f"{request.role} quota stop is missing retry_at"
                )
            if call_num == 3:
                return outcome, provider
            wait = await self.ctx.run_typed(
                f"{name}:quota-wait:{call_num}",
                decide_wait,
                restate.RunOptions(type_hint=WaitDecision),
                retry_at=outcome.retry_at,
                decided_at=outcome.decided_at,
                cap_s=settings.triager_wait_cap_s,
            )
            if wait["action"] == "return":
                return outcome, provider
            wait_s = wait["wait_s"]
            if wait_s is None:
                raise restate.TerminalError("wait decision is missing wait_s")
            await self.ctx.sleep(
                timedelta(seconds=wait_s),
                name=f"{name}:quota:{call_num}",
            )
        raise restate.TerminalError(f"{request.role} did not complete within 3 calls")

    async def run_step(
        self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        return await self.ctx.run_typed(
            name,
            fn,
            restate.RunOptions(max_attempts=2),
            *args,
            **kwargs,
        )

    async def record_event(self, name: str, event: EventWrite) -> str:
        raw = await self.ctx.run_typed(
            name,
            post_memory,
            path="/v1/events",
            payload=event.model_dump(mode="json"),
        )
        return str(raw.get("id") or stable_id("ev", event.idempotency_key))


async def _durable_github_write(
    ctx: restate.WorkflowContext,
    step_name: str,
    method: str,
    path: str,
    payload: dict[str, Any],
) -> GithubWriteOutcome:
    for call_num in (1, 2):
        raw = await ctx.run_typed(
            f"{step_name}:{call_num}",
            call_github_api,
            restate.RunOptions(max_attempts=2),
            method=method,
            path=path,
            payload=payload,
        )
        outcome = GithubWriteOutcome.model_validate(raw)
        if outcome.outcome == "ok":
            return outcome
        if outcome.outcome == "failed":
            raise restate.TerminalError(
                f"GitHub write failed ({outcome.error_code}): {outcome.reason}"
            )
        if call_num == 2 or outcome.retry_in is None:
            raise restate.TerminalError("GitHub write throttled past retry budget")
        wait = await ctx.run_typed(
            f"{step_name}:wait:{call_num}",
            decide_delay,
            restate.RunOptions(type_hint=WaitDecision),
            delay_s=outcome.retry_in,
            cap_s=settings.triager_wait_cap_s,
        )
        if wait["action"] == "return" or wait["wait_s"] is None:
            raise restate.TerminalError("GitHub write throttle exceeded wait cap")
        await ctx.sleep(
            timedelta(seconds=wait["wait_s"]),
            name=f"{step_name}:sleep:{call_num}",
        )
    raise restate.TerminalError("GitHub write did not succeed")


def _retryable_result_v2(
    request: IssueRunRequestV2,
    outcome: ModelOutcome,
    *,
    provider: str,
    base_commit: str | None,
    area: str = "unknown",
) -> IssueRunResultV2:
    status = "quota_exhausted" if outcome.outcome == "quota_exhausted" else "throttled"
    return IssueRunResultV2(
        issue_id=request.issue_id,
        attempt=request.attempt,
        status=status,
        stage="failed",
        base_commit=base_commit,
        area=area,
        model=outcome.model_reported or outcome.model_pinned,
        provider=provider,
        retry_at=outcome.retry_at,
        retry_in=outcome.retry_in,
        scope=outcome.scope,
        decided_at=outcome.decided_at,
        started_at=outcome.started_at,
        finished_at=outcome.finished_at,
    )


@issue_run_v2.main(name="run", workflow_retention=timedelta(days=14))
async def run_issue_v2(
    ctx: restate.WorkflowContext, request: IssueRunRequestV2
) -> IssueRunResultV2:
    ctx.set("issue_id", request.issue_id)
    ctx.set("attempt", request.attempt)
    ctx.set("stage", "running")

    try:
        allowed_areas = agentteam_config.repo(request.repo).allowed_area_names
    except ValueError as error:
        ctx.set("stage", "failed")
        raise restate.TerminalError("repository is not configured") from error

    if request.read_basis is None:
        basis_iso = await ctx.run_typed("read-basis", current_basis)
        read_basis = datetime.fromisoformat(basis_iso)
    else:
        read_basis = request.read_basis

    if request.base_commit is not None:
        base_commit = request.base_commit
        base_branch: str | None = None
    else:
        raw_head = await ctx.run_typed(
            "resolve-repo-head",
            call_github_api,
            method="GET",
            path="/v1/repo/head",
            payload=None,
        )
        repo_head = RepoHeadResponse.model_validate(raw_head)
        base_commit = repo_head.head_sha
        base_branch = repo_head.default_branch

    prefix = _key_prefix_v2(request.issue_id, request.attempt)
    wf_key = workflow_key(request.issue_id, request.attempt)
    journal = RestateStepJournal(ctx)
    sync_decisions: list[SyncDecisionItem] = []
    models_used: dict[str, ModelUsageItem] = {}

    # --- Round 1: Triage ---
    r1_key = f"{prefix}:round:1"
    r1_id = stable_id("round", r1_key)
    await ctx.run_typed(
        "open-round:1",
        post_memory,
        path="/v1/rounds",
        payload=RoundWrite(
            idempotency_key=r1_key,
            seq=1,
            base_commit=base_commit,
            read_basis=read_basis,
        ).model_dump(mode="json"),
    )
    issue_ev_key = f"{prefix}:issue"
    issue_ev_id = await journal.record_event(
        "record-issue",
        EventWrite(
            idempotency_key=issue_ev_key,
            source="github_issue",
            kind="issue",
            taint=2,
            payload={
                "issue_id": request.issue_id,
                "title": request.title,
                "body": request.body,
            },
        ),
    )

    triage_run_key = f"{prefix}:triage:1"
    triage_run_id = stable_id("run", triage_run_key)
    triage_outcome, triage_provider = await journal.call_model(
        "triage-call",
        ModelCallRequest(
            run_id=triage_run_id,
            step="triage",
            role="triager",
            messages=triage_messages(
                request.title, request.body, allowed_areas=allowed_areas
            ),
            sampling=Sampling(),
            taint=2,
            source="github_issue",
        ),
    )
    if triage_outcome.outcome != "completed":
        ctx.set("stage", "failed")
        return _retryable_result_v2(
            request,
            triage_outcome,
            provider=triage_provider,
            base_commit=base_commit,
        )

    triage_model = triage_outcome.model_reported or triage_outcome.model_pinned
    models_used["triager"] = ModelUsageItem(
        role="triager", model=triage_model, provider=triage_provider
    )
    triage = parse_triage(triage_outcome.text, allowed_areas=allowed_areas)
    area = triage.area if triage is not None else "unknown"
    triage_error = None if triage is not None else "unparseable"

    await ctx.run_typed(
        "record-triage-run",
        post_memory,
        path="/v1/runs",
        payload=RunWrite(
            idempotency_key=triage_run_key,
            agent_id="triager",
            role="triager",
            round_id=r1_id,
            workflow_id=wf_key,
            step="triage",
            model=triage_model,
            base_commit=base_commit,
            tokens_in=triage_outcome.tokens_in,
            tokens_out=triage_outcome.tokens_out,
            outcome="routed",
            event_ids=[issue_ev_id],
            started_at=triage_outcome.started_at or triage_outcome.decided_at,
            ended_at=triage_outcome.finished_at or triage_outcome.decided_at,
        ).model_dump(mode="json"),
    )
    triage_prop_key = f"{prefix}:triage-proposal:1"
    triage_prop_raw = await ctx.run_typed(
        "write-triage-proposal",
        post_memory,
        path=f"/v1/rounds/{r1_id}/proposals",
        payload=ProposalWrite(
            idempotency_key=triage_prop_key,
            round_id=r1_id,
            agent_id="triager",
            run_id=triage_run_id,
            op="route_issue",
            evidence_event_ids=[issue_ev_id],
            payload={
                "issue_id": request.issue_id,
                "attempt": request.attempt,
                "triage": None if triage is None else triage.model_dump(mode="json"),
                "triage_error": triage_error,
                "model": triage_model,
                "request_id": triage_outcome.request_id,
            },
        ).model_dump(mode="json"),
    )
    triage_prop_id = str(
        triage_prop_raw.get("id") or stable_id("prop", triage_prop_key)
    )
    r1_commit_raw = await ctx.run_typed(
        "commit-round:1",
        post_memory,
        path=f"/v1/rounds/{r1_id}/commit",
        payload={},
    )
    r1_committed = RoundRecord.model_validate(r1_commit_raw)
    sync_decisions.append(
        SyncDecisionItem(
            step="triage",
            round_id=r1_id,
            proposal_id=triage_prop_id,
            op="route_issue",
            status="accepted"
            if triage_prop_id in r1_committed.committed_proposals
            else "rejected",
            reason="supported triage routing proposal",
        )
    )

    # --- Round 2: Reproducer ---
    sessions = SandboxSessionManager(make_sandbox)
    repro_basis_iso = await ctx.run_typed("read-basis:repro", current_basis)
    r2_key = f"{prefix}:round:2"
    r2_id = stable_id("round", r2_key)
    await ctx.run_typed(
        "open-round:2",
        post_memory,
        path="/v1/rounds",
        payload=RoundWrite(
            idempotency_key=r2_key,
            seq=2,
            base_commit=base_commit,
            read_basis=datetime.fromisoformat(repro_basis_iso),
        ).model_dump(mode="json"),
    )
    repro_run_key = f"{prefix}:repro:1"
    repro_run_id = stable_id("run", repro_run_key)
    repro = await run_reproducer_step(
        journal=journal,
        sessions=sessions,
        session_key=f"{wf_key}:repro",
        key_prefix=prefix,
        run_id=repro_run_id,
        repo=request.repo,
        base_commit=base_commit,
        title=request.title,
        body=request.body,
        triage=triage,
        turn_cap=settings.reproducer_turn_cap,
        prompt_token_cap=settings.reproducer_prompt_token_cap,
        cmd_timeout_s=settings.sandbox_cmd_timeout_s,
    )
    if repro.retry_outcome is not None:
        ctx.set("stage", "failed")
        return _retryable_result_v2(
            request,
            repro.retry_outcome,
            provider=repro.provider or "groq",
            base_commit=base_commit,
            area=area,
        )
    if repro.model and repro.provider:
        models_used["reproducer"] = ModelUsageItem(
            role="reproducer", model=repro.model, provider=repro.provider
        )
    if not repro.succeeded or not repro.files:
        ctx.set("stage", "failed")
        return IssueRunResultV2(
            issue_id=request.issue_id,
            attempt=request.attempt,
            status="failed",
            stage="failed",
            base_commit=base_commit,
            area=area,
            triage=triage,
            triage_error=triage_error,
            failure_reason=repro.failure_reason or "repro_not_established",
            model=repro.model or triage_model,
            provider=repro.provider or triage_provider,
        )

    await ctx.run_typed(
        "record-repro-run",
        post_memory,
        path="/v1/runs",
        payload=RunWrite(
            idempotency_key=repro_run_key,
            agent_id="reproducer",
            role="reproducer",
            round_id=r2_id,
            workflow_id=wf_key,
            step="repro",
            model=repro.model,
            base_commit=base_commit,
            tokens_in=repro.tokens_in,
            tokens_out=repro.tokens_out,
            outcome="reproduced",
            event_ids=list(repro.event_ids),
            started_at=repro.started_at or read_basis,
            ended_at=repro.ended_at or read_basis,
        ).model_dump(mode="json"),
    )
    repro_prop_key = f"{prefix}:repro-proposal:1"
    repro_prop_raw = await ctx.run_typed(
        "write-repro-proposal",
        post_memory,
        path=f"/v1/rounds/{r2_id}/proposals",
        payload=ProposalWrite(
            idempotency_key=repro_prop_key,
            round_id=r2_id,
            agent_id="reproducer",
            run_id=repro_run_id,
            op="add_repro",
            evidence_event_ids=list(repro.event_ids),
            payload={
                "patch": repro.patch,
                "files": repro.files,
                "test_node_id": repro.test_node_id,
                "test_passed": False,
                "summary": repro.summary,
            },
        ).model_dump(mode="json"),
    )
    repro_prop_id = str(repro_prop_raw.get("id") or stable_id("prop", repro_prop_key))
    r2_commit_raw = await ctx.run_typed(
        "commit-round:2",
        post_memory,
        path=f"/v1/rounds/{r2_id}/commit",
        payload={},
    )
    r2_committed = RoundRecord.model_validate(r2_commit_raw)
    if repro_prop_id not in r2_committed.committed_proposals:
        ctx.set("stage", "failed")
        return IssueRunResultV2(
            issue_id=request.issue_id,
            attempt=request.attempt,
            status="failed",
            stage="failed",
            base_commit=base_commit,
            area=area,
            triage=triage,
            triage_error=triage_error,
            failure_reason="repro_rejected_by_sync",
        )
    sync_decisions.append(
        SyncDecisionItem(
            step="repro",
            round_id=r2_id,
            proposal_id=repro_prop_id,
            op="add_repro",
            status="accepted",
            reason="supported reproducer add_repro proposal",
        )
    )

    # --- Rounds 3+: Fixer -> Tester (up to fixer_max_attempts) ---
    review_notes: str | None = None
    approved_fix_files: list[str] = []
    approved_test = None
    used_fix_attempts = 0

    for fix_attempt in range(1, settings.fixer_max_attempts + 1):
        used_fix_attempts = fix_attempt
        fix_seq = 3 + 2 * (fix_attempt - 1)
        fix_basis_iso = await ctx.run_typed(
            f"read-basis:fix:{fix_attempt}", current_basis
        )
        fix_round_key = f"{prefix}:round:{fix_seq}"
        fix_round_id = stable_id("round", fix_round_key)
        await ctx.run_typed(
            f"open-round:{fix_seq}",
            post_memory,
            path="/v1/rounds",
            payload=RoundWrite(
                idempotency_key=fix_round_key,
                seq=fix_seq,
                base_commit=base_commit,
                read_basis=datetime.fromisoformat(fix_basis_iso),
            ).model_dump(mode="json"),
        )
        fix_run_key = f"{prefix}:fix:{fix_attempt}"
        fix_run_id = stable_id("run", fix_run_key)
        fix = await run_fixer_step(
            journal=journal,
            sessions=sessions,
            session_key=f"{wf_key}:fix:{fix_attempt}",
            key_prefix=prefix,
            run_id=fix_run_id,
            fix_attempt=fix_attempt,
            repo=request.repo,
            base_commit=base_commit,
            title=request.title,
            body=request.body,
            triage=triage,
            test_node_id=repro.test_node_id,
            repro_diff=repro.patch,
            review_notes=review_notes,
            turn_cap=settings.fixer_turn_cap,
            prompt_token_cap=settings.fixer_prompt_token_cap,
            cmd_timeout_s=settings.sandbox_cmd_timeout_s,
        )
        if fix.retry_outcome is not None:
            ctx.set("stage", "failed")
            return _retryable_result_v2(
                request,
                fix.retry_outcome,
                provider=fix.provider or "groq",
                base_commit=base_commit,
                area=area,
            )
        if fix.model and fix.provider:
            models_used["fixer"] = ModelUsageItem(
                role="fixer", model=fix.model, provider=fix.provider
            )
        if not fix.succeeded or not fix.files:
            if fix_attempt < settings.fixer_max_attempts:
                review_notes = (
                    "Previous attempt did not produce a passing source fix; "
                    "ensure non-test source files make the reproduction test pass."
                )
                continue
            ctx.set("stage", "failed")
            return IssueRunResultV2(
                issue_id=request.issue_id,
                attempt=request.attempt,
                status="failed",
                stage="failed",
                base_commit=base_commit,
                area=area,
                triage=triage,
                triage_error=triage_error,
                repro_test_node_id=repro.test_node_id,
                repro_files=repro.files,
                fix_attempts=used_fix_attempts,
                failure_reason=fix.failure_reason or "fix_not_established",
            )

        await ctx.run_typed(
            f"record-fix-run:{fix_attempt}",
            post_memory,
            path="/v1/runs",
            payload=RunWrite(
                idempotency_key=fix_run_key,
                agent_id="fixer",
                role="fixer",
                round_id=fix_round_id,
                workflow_id=wf_key,
                step="fix",
                model=fix.model,
                base_commit=base_commit,
                tokens_in=fix.tokens_in,
                tokens_out=fix.tokens_out,
                outcome="patched",
                event_ids=list(fix.event_ids),
                started_at=fix.started_at or read_basis,
                ended_at=fix.ended_at or read_basis,
            ).model_dump(mode="json"),
        )
        fix_prop_key = f"{prefix}:fix-proposal:{fix_attempt}"
        fix_prop_raw = await ctx.run_typed(
            f"write-fix-proposal:{fix_attempt}",
            post_memory,
            path=f"/v1/rounds/{fix_round_id}/proposals",
            payload=ProposalWrite(
                idempotency_key=fix_prop_key,
                round_id=fix_round_id,
                agent_id="fixer",
                run_id=fix_run_id,
                op="edit_file",
                evidence_event_ids=list(fix.event_ids),
                payload={
                    "patch": fix.patch,
                    "files": fix.files,
                    "test_node_id": repro.test_node_id,
                    "test_passed": True,
                    "fix_attempt": fix_attempt,
                    "summary": fix.summary,
                },
            ).model_dump(mode="json"),
        )
        fix_prop_id = str(fix_prop_raw.get("id") or stable_id("prop", fix_prop_key))
        fix_commit_raw = await ctx.run_typed(
            f"commit-round:{fix_seq}",
            post_memory,
            path=f"/v1/rounds/{fix_round_id}/commit",
            payload={},
        )
        fix_committed = RoundRecord.model_validate(fix_commit_raw)
        if fix_prop_id not in fix_committed.committed_proposals:
            ctx.set("stage", "failed")
            return IssueRunResultV2(
                issue_id=request.issue_id,
                attempt=request.attempt,
                status="failed",
                stage="failed",
                base_commit=base_commit,
                area=area,
                triage=triage,
                triage_error=triage_error,
                repro_test_node_id=repro.test_node_id,
                repro_files=repro.files,
                fix_attempts=used_fix_attempts,
                failure_reason="fix_rejected_by_sync",
            )
        sync_decisions.append(
            SyncDecisionItem(
                step="fix",
                round_id=fix_round_id,
                proposal_id=fix_prop_id,
                op="edit_file",
                status="accepted",
                reason="supported fixer edit_file proposal",
            )
        )

        # Tester round
        test_seq = fix_seq + 1
        test_basis_iso = await ctx.run_typed(
            f"read-basis:test:{fix_attempt}", current_basis
        )
        test_round_key = f"{prefix}:round:{test_seq}"
        test_round_id = stable_id("round", test_round_key)
        await ctx.run_typed(
            f"open-round:{test_seq}",
            post_memory,
            path="/v1/rounds",
            payload=RoundWrite(
                idempotency_key=test_round_key,
                seq=test_seq,
                base_commit=base_commit,
                read_basis=datetime.fromisoformat(test_basis_iso),
            ).model_dump(mode="json"),
        )
        test_run_key = f"{prefix}:test:{fix_attempt}"
        test_run_id = stable_id("run", test_run_key)
        test_res = await run_tester_step(
            journal=journal,
            sandbox_factory=make_sandbox,
            key_prefix=prefix,
            run_id=test_run_id,
            fix_attempt=fix_attempt,
            repo=request.repo,
            base_commit=base_commit,
            title=request.title,
            body=request.body,
            test_node_id=repro.test_node_id,
            repro_diff=repro.patch,
            fix_diff=fix.patch,
            repro_files=repro.files,
            fix_files=fix.files,
            prompt_token_cap=settings.tester_prompt_token_cap,
            cmd_timeout_s=settings.sandbox_cmd_timeout_s,
        )
        if test_res.retry_outcome is not None:
            ctx.set("stage", "failed")
            return _retryable_result_v2(
                request,
                test_res.retry_outcome,
                provider=test_res.provider or "groq",
                base_commit=base_commit,
                area=area,
            )
        if test_res.model and test_res.provider:
            models_used["tester"] = ModelUsageItem(
                role="tester", model=test_res.model, provider=test_res.provider
            )
        assert test_res.verdict is not None

        await ctx.run_typed(
            f"record-test-run:{fix_attempt}",
            post_memory,
            path="/v1/runs",
            payload=RunWrite(
                idempotency_key=test_run_key,
                agent_id="tester",
                role="tester",
                round_id=test_round_id,
                workflow_id=wf_key,
                step="test",
                model=test_res.model,
                base_commit=base_commit,
                tokens_in=test_res.tokens_in,
                tokens_out=test_res.tokens_out,
                outcome="reviewed",
                event_ids=list(test_res.event_ids),
                started_at=test_res.started_at or read_basis,
                ended_at=test_res.ended_at or read_basis,
            ).model_dump(mode="json"),
        )
        test_prop_key = f"{prefix}:test-proposal:{fix_attempt}"
        test_prop_raw = await ctx.run_typed(
            f"write-test-proposal:{fix_attempt}",
            post_memory,
            path=f"/v1/rounds/{test_round_id}/proposals",
            payload=ProposalWrite(
                idempotency_key=test_prop_key,
                round_id=test_round_id,
                agent_id="tester",
                run_id=test_run_id,
                op="review",
                evidence_event_ids=list(test_res.event_ids),
                payload={
                    "verdict": test_res.verdict,
                    "notes": test_res.notes,
                    "summary": test_res.summary,
                    "repro_failed_before": test_res.repro_failed_before,
                    "repro_passed_after": test_res.repro_passed_after,
                    "suite_passed": test_res.suite_passed,
                },
            ).model_dump(mode="json"),
        )
        test_prop_id = str(test_prop_raw.get("id") or stable_id("prop", test_prop_key))
        test_commit_raw = await ctx.run_typed(
            f"commit-round:{test_seq}",
            post_memory,
            path=f"/v1/rounds/{test_round_id}/commit",
            payload={},
        )
        test_committed = RoundRecord.model_validate(test_commit_raw)
        if test_prop_id not in test_committed.committed_proposals:
            ctx.set("stage", "failed")
            return IssueRunResultV2(
                issue_id=request.issue_id,
                attempt=request.attempt,
                status="failed",
                stage="failed",
                base_commit=base_commit,
                area=area,
                triage=triage,
                triage_error=triage_error,
                repro_test_node_id=repro.test_node_id,
                repro_files=repro.files,
                fix_files=fix.files,
                fix_attempts=used_fix_attempts,
                failure_reason="review_rejected_by_sync",
            )
        sync_decisions.append(
            SyncDecisionItem(
                step="test",
                round_id=test_round_id,
                proposal_id=test_prop_id,
                op="review",
                status="accepted",
                reason="supported tester review proposal",
            )
        )

        if test_res.verdict == "approve":
            approved_fix_files = fix.files
            approved_test = test_res
            break
        review_notes = test_res.notes or "Clean-checkout review requested changes."

    if approved_test is None:
        ctx.set("stage", "failed")
        return IssueRunResultV2(
            issue_id=request.issue_id,
            attempt=request.attempt,
            status="failed",
            stage="failed",
            base_commit=base_commit,
            area=area,
            triage=triage,
            triage_error=triage_error,
            repro_test_node_id=repro.test_node_id,
            repro_files=repro.files,
            fix_attempts=used_fix_attempts,
            failure_reason="review_changes_requested",
        )

    # --- Open Draft PR, Evidence Comment, Pending Commit Status ---
    pr_title = f"fix({area}): {request.title}"[:240]
    pr_body = (
        f"Automated draft PR prepared by Agent Team (`{wf_key}`).\n\n"
        f"- Issue: `{request.issue_id}` (attempt {request.attempt})\n"
        f"- Reproduction test: `{repro.test_node_id}`\n"
        "- Approving records your decision; merging remains manual."
    )
    pr_out = await _durable_github_write(
        ctx,
        "open-pr",
        "POST",
        "/v1/prs",
        OpenPrRequest(
            idempotency_key=f"{prefix}:open-pr",
            issue_id=request.issue_id,
            attempt=request.attempt,
            base_commit=base_commit,
            base_branch=base_branch,
            title=pr_title,
            body=pr_body,
            files=list(approved_test.resulting_files),
        ).model_dump(mode="json"),
    )
    if pr_out.pr_number is None or not pr_out.head_sha:
        ctx.set("stage", "failed")
        raise restate.TerminalError("GitHub PR creation did not return PR metadata")

    await _durable_github_write(
        ctx,
        "upsert-evidence",
        "PUT",
        f"/v1/prs/{pr_out.pr_number}/evidence",
        EvidenceCommentRequest(
            idempotency_key=f"{prefix}:evidence",
            issue_id=request.issue_id,
            attempt=request.attempt,
            base_commit=base_commit,
            head_sha=pr_out.head_sha,
            triage=triage,
            repro_test_node_id=repro.test_node_id,
            repro_failed_before=approved_test.repro_failed_before,
            repro_passed_after=approved_test.repro_passed_after,
            suite_passed=approved_test.suite_passed,
            suite_summary=approved_test.suite_summary,
            review_verdict="approve",
            review_summary=approved_test.summary,
            repro_files=repro.files,
            fix_files=approved_fix_files,
            sync_decisions=sync_decisions,
            cited_facts=[],
            models_used=list(models_used.values()),
        ).model_dump(mode="json"),
    )
    await _durable_github_write(
        ctx,
        "status-pending",
        "POST",
        f"/v1/commits/{pr_out.head_sha}/status",
        CommitStatusRequest(
            idempotency_key=f"{prefix}:status:pending",
            state="pending",
            description="Checks passed; awaiting maintainer PR review.",
        ).model_dump(mode="json"),
    )

    ctx.set("pr_number", pr_out.pr_number)
    ctx.set("pr_url", pr_out.html_url or "")
    ctx.set("pr_head_sha", pr_out.head_sha)
    ctx.set("stage", "awaiting_approval")

    decision = await ctx.promise("pr_approval", type_hint=ApprovalDecision).value()
    ctx.set("approval_recorded", True)
    effective_sha = decision.head_sha or pr_out.head_sha

    await journal.record_event(
        "record-decision",
        EventWrite(
            idempotency_key=f"{prefix}:decision",
            source="maintainer",
            kind="decision",
            taint=0,
            payload={
                "issue_id": request.issue_id,
                "attempt": request.attempt,
                "pr_number": pr_out.pr_number,
                "head_sha": effective_sha,
                "approved": decision.approved,
                "by": decision.by,
                "reason": decision.reason,
                "source": decision.source,
            },
        ),
    )

    if decision.approved:
        ctx.set("stage", "merged")
        await _durable_github_write(
            ctx,
            "status-final",
            "POST",
            f"/v1/commits/{effective_sha}/status",
            CommitStatusRequest(
                idempotency_key=f"{prefix}:status:final",
                state="success",
                description=f"Approved by {decision.by}; ready to merge."[:140],
            ).model_dump(mode="json"),
        )
        return IssueRunResultV2(
            issue_id=request.issue_id,
            attempt=request.attempt,
            status="approved",
            stage="merged",
            base_commit=base_commit,
            area=area,
            triage=triage,
            triage_error=triage_error,
            repro_test_node_id=repro.test_node_id,
            repro_files=repro.files,
            fix_files=approved_fix_files,
            fix_attempts=used_fix_attempts,
            pr_number=pr_out.pr_number,
            pr_url=pr_out.html_url,
            pr_head_sha=effective_sha,
            approval=decision,
        )

    ctx.set("stage", "closed")
    await _durable_github_write(
        ctx,
        "status-final",
        "POST",
        f"/v1/commits/{effective_sha}/status",
        CommitStatusRequest(
            idempotency_key=f"{prefix}:status:final",
            state="failure",
            description=f"Rejected by {decision.by}."[:140],
        ).model_dump(mode="json"),
    )
    return IssueRunResultV2(
        issue_id=request.issue_id,
        attempt=request.attempt,
        status="rejected",
        stage="closed",
        base_commit=base_commit,
        area=area,
        triage=triage,
        triage_error=triage_error,
        repro_test_node_id=repro.test_node_id,
        repro_files=repro.files,
        fix_files=approved_fix_files,
        fix_attempts=used_fix_attempts,
        pr_number=pr_out.pr_number,
        pr_url=pr_out.html_url,
        pr_head_sha=effective_sha,
        approval=decision,
        failure_reason=decision.reason or "pr_rejected",
    )


@issue_run_v2.handler(name="approve")
async def approve_issue_v2(
    ctx: restate.WorkflowSharedContext,
    decision: ApprovalDecision,
) -> ApprovalResult:
    issue_id = await ctx.get("issue_id", type_hint=str) or ctx.key()
    attempt = await ctx.get("attempt", type_hint=int) or 1
    stage = await ctx.get("stage", type_hint=str) or "running"
    already = await ctx.get("approval_recorded", type_hint=bool) or False
    if already or stage in {"merged", "closed"}:
        return ApprovalResult(
            issue_id=issue_id,
            attempt=attempt,
            status="already_decided",
            stage=stage,  # type: ignore[arg-type]
        )

    stored_head = await ctx.get("pr_head_sha", type_hint=str)
    pr_number = await ctx.get("pr_number", type_hint=int)
    if (
        decision.head_sha is not None
        and stored_head is not None
        and decision.head_sha != stored_head
    ):
        live_head = stored_head
        if pr_number is not None:
            pr_info = await ctx.run_typed(
                f"verify-pr-head:{decision.head_sha[:12]}",
                call_github_api,
                method="GET",
                path=f"/v1/prs/{pr_number}",
                payload=None,
            )
            live_head = str(pr_info.get("head_sha") or stored_head)
        if decision.head_sha != live_head:
            return ApprovalResult(
                issue_id=issue_id,
                attempt=attempt,
                status="stale_head",
                stage=stage,  # type: ignore[arg-type]
            )

    try:
        await ctx.promise("pr_approval", type_hint=ApprovalDecision).resolve(decision)
    except restate.TerminalError as error:
        if error.status_code == 409 or "already completed" in str(error).lower():
            return ApprovalResult(
                issue_id=issue_id,
                attempt=attempt,
                status="already_decided",
                stage=stage,  # type: ignore[arg-type]
            )
        raise

    next_stage = "merged" if decision.approved else "closed"
    return ApprovalResult(
        issue_id=issue_id,
        attempt=attempt,
        status="recorded",
        stage=next_stage,
    )


@issue_run_v2.handler(name="info")
async def info_issue_v2(
    ctx: restate.WorkflowSharedContext,
) -> RunStatusV2:
    issue_id = await ctx.get("issue_id", type_hint=str) or ctx.key()
    attempt = await ctx.get("attempt", type_hint=int) or 1
    stage = await ctx.get("stage", type_hint=str) or "running"
    pr_number = await ctx.get("pr_number", type_hint=int)
    pr_url = await ctx.get("pr_url", type_hint=str)
    head_sha = await ctx.get("pr_head_sha", type_hint=str)
    state = "awaiting_approval" if stage == "awaiting_approval" else "running"
    return RunStatusV2(
        issue_id=issue_id,
        attempt=attempt,
        state=state,
        pr_number=pr_number,
        pr_url=pr_url or None,
        head_sha=head_sha,
    )
