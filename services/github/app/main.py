import asyncio
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from agentteam_contracts import (
    CommitStatusRequest,
    EvidenceCommentRequest,
    GithubWriteOutcome,
    OpenPrRequest,
    RepoHeadResponse,
)
from fastapi import FastAPI, HTTPException, Request

from .auth import GithubAppTokenSource, StaticTokenSource, TokenSource
from .client import GithubClient, GithubFailure
from .poller import Poller
from .pr_service import (
    fetch_repo_head,
    open_draft_pull_request,
    set_commit_status,
    upsert_pr_evidence_comment,
)
from .schemas import Clock, CycleSummary, IssueView, Sleep
from .settings import GithubSettings, SettingsError
from .store import IssueState, Store, WriteConflictError
from .workflow_client import WorkflowClient


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Runtime:
    settings: GithubSettings
    store: Store
    poller: Poller
    github: GithubClient
    workflow: WorkflowClient
    github_http: httpx.AsyncClient
    workflow_http: httpx.AsyncClient


async def close_runtime(runtime: Runtime) -> None:
    await runtime.github_http.aclose()
    if runtime.workflow_http is not runtime.github_http:
        await runtime.workflow_http.aclose()


def build_runtime(
    settings: GithubSettings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Clock = utc_now,
    sleep: Sleep = asyncio.sleep,
) -> Runtime:
    store = Store(settings.sqlite_path)
    github_http = httpx.AsyncClient(transport=transport)
    workflow_http = httpx.AsyncClient(transport=transport)
    token_source: TokenSource
    if settings.auth_mode == "app":
        token_source = GithubAppTokenSource(
            app_id=settings.app_id,
            private_key_pem=settings.load_private_key_pem(),
            repo=settings.repo,
            clock=clock,
            installation_id=settings.app_installation_id,
        )
    else:
        token_source = StaticTokenSource(settings.token)
    github = GithubClient(
        token=settings.token,
        repo=settings.repo,
        http=github_http,
        clock=clock,
        sleep=sleep,
        token_source=token_source,
    )
    workflow = WorkflowClient(base_url=settings.workflow_url, http=workflow_http)
    poller = Poller(
        settings=settings,
        github=github,
        workflow=workflow,
        store=store,
        clock=clock,
        sleep=sleep,
        holder=uuid.uuid4().hex,
    )
    return Runtime(
        settings=settings,
        store=store,
        poller=poller,
        github=github,
        workflow=workflow,
        github_http=github_http,
        workflow_http=workflow_http,
    )


def create_app(runtime: Runtime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if runtime is not None:
            yield
            return

        try:
            settings = GithubSettings.from_env()
            settings.validate_for_start()
        except SettingsError as error:
            print(str(error), file=sys.stderr)
            raise SystemExit(2) from error

        active_runtime = build_runtime(settings)
        prepared = False
        try:
            await active_runtime.poller.prepare()
            prepared = True
        finally:
            if not prepared:
                await close_runtime(active_runtime)

        app.state.runtime = active_runtime
        stop = asyncio.Event()
        task = asyncio.create_task(active_runtime.poller.run_forever(stop))
        try:
            yield
        finally:
            stop.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await close_runtime(active_runtime)

    app = FastAPI(title="Agent Team GitHub", lifespan=lifespan)
    app.state.runtime = runtime

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/status")
    def status(request: Request) -> dict[str, object]:
        active = _runtime(request)
        now = active.poller.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise HTTPException(
                status_code=500, detail="poller clock is not timezone-aware"
            )
        day_start = now.astimezone(UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return {
            "repo": active.store.get_meta("repo_full_name") or active.settings.repo,
            "self_login": active.store.get_meta("self_login"),
            "auth_state": active.poller.auth_state,
            "last_cycle": active.poller.last_cycle,
            "starts_today": active.store.count_admitted_since(day_start),
        }

    @app.get("/v1/issues", response_model=list[IssueView])
    def issues(request: Request, state: IssueState | None = None) -> list[IssueView]:
        active = _runtime(request)
        return [
            IssueView(
                issue_key=row.issue_key,
                number=row.number,
                state=row.state,
                reason=row.reason,
                start_attempts=row.start_attempts,
                result_status=row.result_status,
            )
            for row in active.store.list_issues(state=state)
        ]

    @app.post("/v1/poll/once", response_model=CycleSummary)
    async def poll_once(request: Request) -> CycleSummary:
        summary = await _runtime(request).poller.run_cycle()
        if summary.aborted_reason == "cycle_in_progress":
            raise HTTPException(status_code=409, detail="cycle_in_progress")
        return summary

    @app.get("/v1/repo/head", response_model=RepoHeadResponse)
    async def repo_head(request: Request) -> RepoHeadResponse:
        active = _runtime(request)
        try:
            return await fetch_repo_head(active.github)
        except GithubFailure as error:
            raise HTTPException(
                status_code=502,
                detail=error.outcome.reason or error.outcome.outcome,
            ) from error

    @app.post("/v1/prs", response_model=GithubWriteOutcome)
    async def open_pr(request: Request, body: OpenPrRequest) -> GithubWriteOutcome:
        active = _runtime(request)
        try:
            return await open_draft_pull_request(
                settings=active.settings,
                github=active.github,
                store=active.store,
                clock=active.poller.clock,
                request=body,
            )
        except WriteConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.put("/v1/prs/{pr_number}/evidence", response_model=GithubWriteOutcome)
    async def upsert_evidence(
        pr_number: int,
        request: Request,
        body: EvidenceCommentRequest,
    ) -> GithubWriteOutcome:
        active = _runtime(request)
        try:
            return await upsert_pr_evidence_comment(
                github=active.github,
                store=active.store,
                clock=active.poller.clock,
                pr_number=pr_number,
                request=body,
            )
        except WriteConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/v1/commits/{sha}/status", response_model=GithubWriteOutcome)
    async def commit_status(
        sha: str,
        request: Request,
        body: CommitStatusRequest,
    ) -> GithubWriteOutcome:
        active = _runtime(request)
        try:
            return await set_commit_status(
                github=active.github,
                store=active.store,
                clock=active.poller.clock,
                sha=sha,
                request=body,
            )
        except WriteConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    return app


def _runtime(request: Request) -> Runtime:
    active = getattr(request.app.state, "runtime", None)
    if active is None:
        raise HTTPException(status_code=503, detail="GitHub service is starting")
    return active


app = create_app()
