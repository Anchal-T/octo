import asyncio
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from agentteam_common.ids import workflow_key
from agentteam_contracts import IssueRunResult

from .client import GithubClient, GithubFailure
from .rules import (
    admit,
    decide,
    fingerprint,
    issue_key,
    parse_marker,
    render_comment,
    to_run_request,
)
from .schemas import Clock, CycleSummary, GithubIssue, RepoInfo, Sleep
from .settings import (
    GITHUB_LABEL,
    GITHUB_LEASE_TTL_S,
    GITHUB_MAX_START_ATTEMPTS,
    GITHUB_POLL_OVERLAP_S,
    GithubSettings,
)
from .store import CommentKind, IssueRow, Store
from .workflow_client import RunState, WorkflowClient, WorkflowUnavailable

logger = logging.getLogger(__name__)

_REVALIDATE_STATES = {"ignored", "skipped", "pending_start", "submitted"}


class _CycleAbort(Exception):
    pass


class Poller:
    def __init__(
        self,
        *,
        settings: GithubSettings,
        github: GithubClient,
        workflow: WorkflowClient,
        store: Store,
        clock: Clock,
        sleep: Sleep,
        holder: str,
    ) -> None:
        self.settings = settings
        self.github = github
        self.workflow = workflow
        self.store = store
        self.clock = clock
        self.sleep = sleep
        self.holder = holder
        self.self_login = ""
        self.repo_info: RepoInfo | None = None
        self.auth_state = "unknown"
        self.last_cycle: CycleSummary | None = None
        self._cycle_lock = asyncio.Lock()
        self._prepared = False

    async def prepare(self) -> None:
        self.settings.validate_for_start()
        login = await self.github.get_self()
        repo = await self.github.get_repo()
        if (
            not repo.fork
            or repo.owner_login.casefold() != login.casefold()
            or not repo.has_issues
        ):
            raise RuntimeError(
                "GITHUB_REPO must be an issues-enabled fork owned by the bot"
            )

        stored_repo_id = self.store.get_meta("repo_id")
        if stored_repo_id is not None and int(stored_repo_id) != repo.id:
            raise RuntimeError(
                "stored repo_id does not match GITHUB_REPO; use a new GITHUB_SQLITE_PATH"
            )

        await self.github.ensure_label(GITHUB_LABEL)
        now = self._now()
        baseline = self.store.get_meta("baseline_at")
        if baseline is None:
            baseline = self._iso(now)
            self.store.set_meta("baseline_at", baseline)
        if self.store.get_meta("watermark") is None:
            self.store.set_meta("watermark", baseline)
        self.store.set_meta("repo_id", str(repo.id))
        self.store.set_meta("repo_full_name", repo.full_name)
        self.store.set_meta("self_login", login)
        self.self_login = login
        self.repo_info = repo
        self.auth_state = "ok"
        self._prepared = True

    async def run_cycle(self) -> CycleSummary:
        now = self._now()
        summary = CycleSummary(ran_at=now)
        if self._cycle_lock.locked():
            summary.aborted_reason = "cycle_in_progress"
            self.last_cycle = summary
            return summary

        async with self._cycle_lock:
            if not self.store.acquire_lease(self.holder, now, GITHUB_LEASE_TTL_S):
                summary.aborted_reason = "cycle_in_progress"
                self.last_cycle = summary
                return summary
            try:
                if not self._prepared:
                    raise RuntimeError("poller.prepare() must run before polling")
                try:
                    await self._run_locked(summary)
                except _CycleAbort as error:
                    summary.aborted_reason = str(error)
                except GithubFailure as error:
                    self._record_auth_failure(error)
                    summary.aborted_reason = (
                        error.outcome.error_code or error.outcome.outcome
                    )
                self.last_cycle = summary
                return summary
            finally:
                self.store.release_lease(self.holder)

    async def run_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.run_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - keep the poll loop alive
                logger.error("poll cycle failed error_type=%s", type(error).__name__)

            if stop.is_set():
                break
            interval = max(
                self.settings.poll_interval_s,
                self.github.poll_interval_s or 0,
            )
            sleep_task = asyncio.ensure_future(self.sleep(interval))
            stop_task = asyncio.create_task(stop.wait())
            done, pending = await asyncio.wait(
                (sleep_task, stop_task), return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()

    async def _run_locked(self, summary: CycleSummary) -> None:
        watermark = self.store.get_meta("watermark")
        since = None
        if watermark is not None:
            since = self._datetime(watermark) - timedelta(seconds=GITHUB_POLL_OVERLAP_S)

        discovery = await self.github.list_issues(since=since)
        self.auth_state = "ok"
        self._check_rate_limit()
        current = {
            issue.number: issue
            for issue in discovery.issues
            if not issue.is_pull_request
        }

        for row in self.store.list_issues():
            if row.state not in _REVALIDATE_STATES:
                continue
            try:
                issue = await self.github.get_issue(row.number)
            except GithubFailure as error:
                self._record_auth_failure(error)
                current.pop(row.number, None)
                self._check_rate_limit()
                if self._should_abort(error):
                    raise
                continue

            self.auth_state = "ok"
            self._check_rate_limit()
            if issue.is_pull_request:
                current.pop(row.number, None)
            elif issue.state == "closed" or GITHUB_LABEL not in issue.labels:
                self.store.upsert_issue(
                    replace(
                        row,
                        state="cancelled",
                        reason="closed_or_unlabelled",
                        deferred_reason=None,
                        fingerprint=fingerprint(issue),
                        github_updated_at=self._iso(issue.updated_at),
                    )
                )
                current.pop(row.number, None)
            else:
                current[row.number] = issue

        baseline = self._datetime(self.store.get_meta("baseline_at"))
        for discovered in discovery.issues:
            issue = current.get(discovered.number)
            if issue is None or issue.is_pull_request:
                continue
            await self._process_issue(issue, baseline, summary)
            self._check_rate_limit()

        result_details: dict[str, dict[str, str]] = {}
        await self._process_submitted(summary, result_details)
        await self._admit_pending(current, summary)
        await self._post_due_comments(current, summary, result_details)

        if not discovery.truncated and discovery.max_updated_at is not None:
            latest = discovery.max_updated_at
            if watermark is not None:
                latest = max(latest, self._datetime(watermark))
            self.store.set_meta("watermark", self._iso(latest))

    async def _process_issue(
        self, issue: GithubIssue, baseline: datetime, summary: CycleSummary
    ) -> None:
        if self.repo_info is None:
            raise RuntimeError("poller.prepare() must run before processing issues")
        key = issue_key(self.repo_info.id, issue.number)
        fp = fingerprint(issue)
        updated_at = self._iso(issue.updated_at)
        existing = self.store.get_issue(key)

        if existing is not None:
            if existing.state == "submitted":
                if existing.github_updated_at != updated_at:
                    self.store.upsert_issue(
                        replace(existing, github_updated_at=updated_at)
                    )
                return
            if existing.state == "finished" or (
                existing.state == "failed" and existing.fingerprint == fp
            ):
                if existing.github_updated_at == updated_at:
                    return
                self.store.upsert_issue(replace(existing, github_updated_at=updated_at))
                await self._maybe_retriage(existing, issue)
                return
            if (
                existing.state != "cancelled"
                and existing.fingerprint == fp
                and existing.github_updated_at == updated_at
            ):
                return

        decision = decide(
            issue,
            self_login=self.self_login,
            baseline_at=baseline,
            opted_in_after_baseline=None,
            process_own_issues=self.settings.process_own_issues,
        )
        if decision.needs_label_events:
            try:
                events = await self.github.list_label_events(issue.number)
            except GithubFailure as error:
                self._record_auth_failure(error)
                if self._should_abort(error):
                    raise
                if existing is None:
                    raise _CycleAbort("label_event_lookup_failed") from error
                return
            self._check_rate_limit()
            opted_in = any(
                event.event == "labeled"
                and event.label_name.casefold() == GITHUB_LABEL.casefold()
                and event.created_at >= baseline
                for event in events
            )
            decision = decide(
                issue,
                self_login=self.self_login,
                baseline_at=baseline,
                opted_in_after_baseline=opted_in,
                process_own_issues=self.settings.process_own_issues,
            )

        if decision.state is None:
            return

        changed = (
            existing is None
            or existing.fingerprint != fp
            or existing.state != decision.state
        )
        if existing is None or changed:
            start_attempts = 0
            admitted_at = None
            submitted_at = None
            finished_at = None
            result_status = None
        else:
            start_attempts = existing.start_attempts
            admitted_at = existing.admitted_at
            submitted_at = existing.submitted_at
            finished_at = existing.finished_at
            result_status = existing.result_status
        row = IssueRow(
            issue_key=key,
            number=issue.number,
            state=decision.state,
            reason=decision.reason,
            deferred_reason=None,
            fingerprint=fp,
            github_updated_at=updated_at,
            created_at=self._iso(issue.created_at),
            start_attempts=start_attempts,
            admitted_at=admitted_at,
            submitted_at=submitted_at,
            finished_at=finished_at,
            result_status=result_status,
            attempt=1 if existing is None else existing.attempt,
            label_seen_at=None if existing is None else existing.label_seen_at,
        )
        self.store.upsert_issue(row)
        if decision.state == "skipped":
            summary.skipped += 1

    async def _process_submitted(
        self, summary: CycleSummary, result_details: dict[str, dict[str, str]]
    ) -> None:
        now = self._now()
        for row in self.store.submitted_issues():
            result: IssueRunResult | None = None
            status: RunState | None = None
            try:
                status = await self.workflow.status(row.issue_key, attempt=row.attempt)
                if isinstance(status, IssueRunResult):
                    result = status
            except WorkflowUnavailable:
                pass

            if result is not None:
                finished_at = self._iso(now)
                self.store.upsert_issue(
                    replace(
                        row,
                        state="finished",
                        reason=None,
                        deferred_reason=None,
                        finished_at=finished_at,
                        result_status=result.status,
                    )
                )
                result_details[row.issue_key] = self._result_detail(result, finished_at)
                summary.finished += 1
            elif (
                status != "running"
                and row.submitted_at is not None
                and now - self._datetime(row.submitted_at) >= timedelta(minutes=10)
            ):
                self.store.upsert_issue(
                    replace(
                        row,
                        state="failed",
                        reason="workflow_unavailable",
                        deferred_reason=None,
                    )
                )
                summary.failed += 1

    async def _admit_pending(
        self, current: dict[int, GithubIssue], summary: CycleSummary
    ) -> None:
        pending = [
            row
            for row in self.store.list_issues(state="pending_start")
            if row.number in current
        ]
        if not pending:
            return
        now = self._now()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        admission = admit(
            pending,
            started_this_cycle=0,
            started_today=self.store.count_admitted_since(day_start),
            per_cycle=self.settings.max_starts_per_cycle,
            per_day=self.settings.max_starts_per_day,
        )

        for row in admission.deferred_cycle:
            self.store.upsert_issue(replace(row, deferred_reason="cycle_limit"))
        for row in admission.deferred_day:
            self.store.upsert_issue(replace(row, deferred_reason="daily_limit"))
        summary.deferred = len(admission.deferred_cycle) + len(admission.deferred_day)

        for row in admission.starts:
            issue = current[row.number]
            admitted_at = row.admitted_at or self._iso(now)
            admitted = replace(
                row,
                deferred_reason=None,
                admitted_at=admitted_at,
            )
            self.store.upsert_issue(admitted)
            self.store.record_admission(
                admitted.issue_key, admitted.attempt, admitted_at
            )
            try:
                result = await self.workflow.start(
                    to_run_request(
                        issue,
                        row.issue_key,
                        self.settings.body_max_chars,
                        repo=self.settings.repo,
                        attempt=row.attempt,
                    )
                )
            except WorkflowUnavailable:
                result = "unavailable"

            if result == "submitted":
                self.store.upsert_issue(
                    replace(
                        admitted,
                        state="submitted",
                        reason=None,
                        submitted_at=self._iso(now),
                    )
                )
                summary.started += 1
            elif result in {"rejected", "conflict"}:
                reason = (
                    "workflow_rejected" if result == "rejected" else "workflow_conflict"
                )
                self.store.upsert_issue(
                    replace(
                        admitted,
                        state="failed",
                        reason=reason,
                        start_attempts=admitted.start_attempts + 1,
                    )
                )
                summary.failed += 1
            else:
                attempts = admitted.start_attempts + 1
                if attempts >= GITHUB_MAX_START_ATTEMPTS:
                    self.store.upsert_issue(
                        replace(
                            admitted,
                            state="failed",
                            reason="workflow_unavailable",
                            start_attempts=attempts,
                        )
                    )
                    summary.failed += 1
                else:
                    self.store.upsert_issue(
                        replace(
                            admitted,
                            start_attempts=attempts,
                            reason=None,
                        )
                    )

    async def _post_due_comments(
        self,
        current: dict[int, GithubIssue],
        summary: CycleSummary,
        result_details: dict[str, dict[str, str]],
    ) -> None:
        due: list[tuple[IssueRow, CommentKind, dict[str, str]]] = []
        for row in self.store.list_issues():
            at = self._iso(self._now())
            if row.state == "submitted":
                issue = current.get(row.number)
                detail = {"at": at, "attempt": str(row.attempt)}
                if issue is not None and len(issue.body) > self.settings.body_max_chars:
                    detail.update(
                        truncated="true",
                        body_max_chars=str(self.settings.body_max_chars),
                    )
                due.append((row, "ack", detail))
            elif row.state == "pending_start" and row.deferred_reason:
                due.append(
                    (
                        row,
                        "queued",
                        {
                            "at": at,
                            "reason": row.deferred_reason,
                            "limit": str(self.settings.max_starts_per_day),
                        },
                    )
                )
            elif row.state == "skipped" and row.reason == "empty_title":
                due.append((row, "skip", {"at": at, "reason": row.reason}))
            elif row.state == "failed" and row.reason in {
                "workflow_rejected",
                "workflow_conflict",
                "workflow_unavailable",
            }:
                due.append((row, "failure", {"at": at, "reason": row.reason}))
            elif row.state == "finished" and row.result_status:
                detail = result_details.get(row.issue_key)
                if detail is None:
                    detail = {"at": at, "status": row.result_status, "refetch": "true"}
                due.append((row, "result", detail))

        posted = 0
        for row, kind, detail in due:
            if self.store.comment_exists(
                row.issue_key, row.attempt, kind, row.fingerprint
            ):
                continue
            try:
                comments = await self.github.list_comments(row.number)
            except GithubFailure as error:
                self._record_auth_failure(error)
                if self._should_abort(error):
                    raise
                continue
            self.auth_state = "ok"
            self._check_rate_limit()
            run_key = workflow_key(row.issue_key, row.attempt)
            existing = next(
                (
                    comment
                    for comment in comments
                    if comment.author_login.casefold() == self.self_login.casefold()
                    and parse_marker(comment.body)
                    == (kind, run_key, row.fingerprint[:8])
                ),
                None,
            )
            if existing is not None:
                self.store.record_comment(
                    row.issue_key,
                    row.attempt,
                    kind,
                    row.fingerprint,
                    existing.id,
                    existing.created_at,
                )
                continue

            if detail.get("refetch") == "true":
                refetched = await self._refetch_result_detail(row, detail)
                if refetched is None:
                    continue
                detail = refetched

            if posted:
                await self.sleep(1)
            body = render_comment(
                kind,
                key=run_key,
                fp=row.fingerprint,
                detail=detail,
            )
            try:
                comment_id = await self.github.post_comment(row.number, body)
            except GithubFailure as error:
                self._record_auth_failure(error)
                if error.outcome.outcome in {"invalid", "not_found"}:
                    continue
                raise
            self.auth_state = "ok"
            self._check_rate_limit()
            self.store.record_comment(
                row.issue_key,
                row.attempt,
                kind,
                row.fingerprint,
                comment_id,
                self._now(),
            )
            posted += 1
            summary.comments_posted += 1

    async def _maybe_retriage(self, row: IssueRow, issue: GithubIssue) -> None:
        if issue.state != "open" or GITHUB_LABEL not in issue.labels:
            return
        if row.attempt >= 99:
            return
        try:
            events = await self.github.list_label_events(issue.number)
        except GithubFailure as error:
            self._record_auth_failure(error)
            if self._should_abort(error):
                raise
            return
        self._check_rate_limit()

        timestamps = [
            value
            for value in (
                row.label_seen_at,
                row.finished_at,
                row.submitted_at,
                row.admitted_at,
            )
            if value is not None
        ]
        if not timestamps:
            return
        threshold = max(self._datetime(value) for value in timestamps)
        allow_self = self.settings.process_own_issues
        matching = [
            event
            for event in events
            if event.event == "labeled"
            and event.label_name.casefold() == GITHUB_LABEL.casefold()
            and event.actor_login is not None
            and (
                allow_self or event.actor_login.casefold() != self.self_login.casefold()
            )
            and event.created_at > threshold
        ]
        if not matching:
            return
        newest = max(matching, key=lambda event: event.created_at)
        self.store.upsert_issue(
            replace(
                row,
                attempt=row.attempt + 1,
                state="pending_start",
                reason=None,
                deferred_reason=None,
                fingerprint=fingerprint(issue),
                github_updated_at=self._iso(issue.updated_at),
                label_seen_at=self._iso(newest.created_at),
                start_attempts=0,
                admitted_at=None,
                submitted_at=None,
                finished_at=None,
                result_status=None,
            )
        )

    def _result_detail(self, result: IssueRunResult, at: str) -> dict[str, str]:
        detail = {"at": at, "status": result.status}
        if result.retry_at is not None:
            detail["retry_at"] = self._iso(result.retry_at).replace("+00:00", "Z")
        if result.retry_in is not None:
            seconds = result.retry_in
            detail["retry_in"] = (
                str(int(seconds)) if seconds.is_integer() else str(seconds)
            )
        if result.model is not None:
            detail["model"] = result.model
        if result.provider is not None:
            detail["provider"] = result.provider
        if result.triage is not None:
            detail["triage_kind"] = result.triage.kind
            detail["triage_priority"] = result.triage.priority
            detail["triage_summary"] = result.triage.summary
            detail["triage_area"] = result.area or "unknown"
        if result.triage_error is not None:
            detail["triage_error"] = result.triage_error
        return detail

    async def _refetch_result_detail(
        self, row: IssueRow, detail: dict[str, str]
    ) -> dict[str, str] | None:
        """Rebuild result details when the comment is posted in a later cycle."""
        try:
            status = await self.workflow.status(row.issue_key, attempt=row.attempt)
        except WorkflowUnavailable:
            return None
        if isinstance(status, IssueRunResult):
            return self._result_detail(status, detail["at"])
        return {
            "at": detail["at"],
            "status": detail["status"],
            "triage_missing": "true",
        }

    def _check_rate_limit(self) -> None:
        if self.github.rate_limit_floor_reached:
            raise _CycleAbort("rate_limit_floor")

    def _record_auth_failure(self, error: GithubFailure) -> None:
        outcome = error.outcome
        if outcome.outcome == "unauthorized":
            self.auth_state = "unauthorized"
        elif outcome.outcome in {"forbidden", "gone"}:
            self.auth_state = "forbidden"
        elif outcome.error_code == "repo_not_found":
            self.auth_state = "not_found"

    @staticmethod
    def _should_abort(error: GithubFailure) -> bool:
        outcome = error.outcome
        return (
            outcome.outcome
            in {
                "unauthorized",
                "forbidden",
                "rate_limited",
                "gone",
                "unavailable",
            }
            or outcome.error_code == "repo_not_found"
        )

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("poller clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    @classmethod
    def _iso(cls, value: datetime) -> str:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC).isoformat()

    @staticmethod
    def _datetime(value: str | None) -> datetime:
        if value is None:
            raise ValueError("required timestamp is missing")
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("stored timestamps must be timezone-aware")
        return parsed.astimezone(UTC)
