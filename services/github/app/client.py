import asyncio
import logging
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Literal

import httpx

from .schemas import (
    Clock,
    GithubComment,
    GithubIssue,
    GithubOutcome,
    LabelEvent,
    PollResult,
    RepoInfo,
    Sleep,
)
from .settings import (
    GITHUB_API_BASE_URL,
    GITHUB_CONNECT_TIMEOUT_S,
    GITHUB_LABEL,
    GITHUB_MAX_PAGES,
    GITHUB_PAGE_SIZE,
    GITHUB_RATE_LIMIT_FLOOR,
    GITHUB_READ_TIMEOUT_S,
    GITHUB_SECONDARY_LIMIT_WAIT_CAP_S,
)

logger = logging.getLogger(__name__)


class GithubFailure(Exception):
    def __init__(self, outcome: GithubOutcome) -> None:
        super().__init__(outcome.reason or outcome.outcome)
        self.outcome = outcome


class GithubClient:
    def __init__(
        self,
        *,
        token: str,
        repo: str,
        http: httpx.AsyncClient,
        clock: Clock,
        sleep: Sleep,
    ) -> None:
        self.token = token
        self.repo = repo
        self.http = http
        self.clock = clock
        self.sleep = sleep
        self._request_lock = asyncio.Lock()
        self.rate_limit_remaining: int | None = None
        self.rate_limit_reset: datetime | None = None
        self.poll_interval_s: int | None = None

    @property
    def rate_limit_floor_reached(self) -> bool:
        return (
            self.rate_limit_remaining is not None
            and self.rate_limit_remaining < GITHUB_RATE_LIMIT_FLOOR
        )

    async def get_self(self) -> str:
        response = await self._request("GET", "/user")
        return self._json(response).get("login", "")

    async def get_repo(self) -> RepoInfo:
        response = await self._request("GET", f"/repos/{self.repo}")
        item = self._json(response)
        owner = item.get("owner") or {}
        return RepoInfo(
            id=item["id"],
            full_name=item["full_name"],
            fork=item["fork"],
            owner_login=owner.get("login", ""),
            has_issues=item["has_issues"],
        )

    async def get_issue(self, number: int) -> GithubIssue:
        response = await self._request(
            "GET",
            f"/repos/{self.repo}/issues/{number}",
            missing="revalidation",
        )
        return GithubIssue.from_api(self._json(response))

    async def ensure_label(self, name: str) -> None:
        for page in range(1, GITHUB_MAX_PAGES + 1):
            response = await self._request(
                "GET",
                f"/repos/{self.repo}/labels",
                params={"per_page": GITHUB_PAGE_SIZE, "page": page},
            )
            labels = self._json_list(response)
            if any(label.get("name") == name for label in labels):
                return
            if len(labels) < GITHUB_PAGE_SIZE:
                break
        await self._request(
            "POST",
            f"/repos/{self.repo}/labels",
            json={"name": name, "color": "ededed"},
        )

    async def list_issues(self, *, since: datetime | None) -> PollResult:
        issues: list[GithubIssue] = []
        max_updated_at: datetime | None = None
        base_params: dict[str, Any] = {
            "labels": GITHUB_LABEL,
            "state": "open",
            "sort": "updated",
            "direction": "asc",
            "per_page": GITHUB_PAGE_SIZE,
        }
        if since is not None:
            base_params["since"] = since.isoformat()

        for page in range(1, GITHUB_MAX_PAGES + 1):
            response = await self._request(
                "GET",
                f"/repos/{self.repo}/issues",
                params={**base_params, "page": page},
            )
            items = self._json_list(response)
            for item in items:
                issue = GithubIssue.from_api(item)
                issues.append(issue)
                if max_updated_at is None or issue.updated_at > max_updated_at:
                    max_updated_at = issue.updated_at
            if len(items) < GITHUB_PAGE_SIZE:
                return PollResult(
                    issues=issues,
                    pages=page,
                    truncated=False,
                    max_updated_at=max_updated_at,
                )

        return PollResult(
            issues=issues,
            pages=GITHUB_MAX_PAGES,
            truncated=True,
            max_updated_at=max_updated_at,
        )

    async def list_label_events(self, number: int) -> list[LabelEvent]:
        events: list[LabelEvent] = []
        for page in range(1, GITHUB_MAX_PAGES + 1):
            response = await self._request(
                "GET",
                f"/repos/{self.repo}/issues/{number}/events",
                params={"per_page": GITHUB_PAGE_SIZE, "page": page},
                missing="issue",
            )
            items = self._json_list(response)
            events.extend(
                LabelEvent(
                    event=item["event"],
                    label_name=(item.get("label") or {}).get("name", ""),
                    created_at=item["created_at"],
                )
                for item in items
            )
            if len(items) < GITHUB_PAGE_SIZE:
                break
        return events

    async def list_comments(self, number: int) -> list[GithubComment]:
        comments: list[GithubComment] = []
        for page in range(1, GITHUB_MAX_PAGES + 1):
            response = await self._request(
                "GET",
                f"/repos/{self.repo}/issues/{number}/comments",
                params={"per_page": GITHUB_PAGE_SIZE, "page": page},
                missing="issue",
            )
            items = self._json_list(response)
            comments.extend(
                GithubComment(
                    id=item["id"],
                    body=item.get("body") or "",
                    author_login=(item.get("user") or {}).get("login", ""),
                    created_at=item["created_at"],
                )
                for item in items
            )
            if len(items) < GITHUB_PAGE_SIZE:
                break
        return comments

    async def post_comment(self, number: int, body: str) -> int:
        response = await self._request(
            "POST",
            f"/repos/{self.repo}/issues/{number}/comments",
            json={"body": body},
            missing="issue",
        )
        return self._json(response)["id"]

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        missing: Literal["repo", "issue", "revalidation"] = "repo",
    ) -> httpx.Response:
        server_retry = 0
        secondary_retried = False
        retry_delays = (1, 2, 4)
        url = f"{GITHUB_API_BASE_URL}{path}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        while True:
            try:
                async with self._request_lock:
                    response = await self.http.request(
                        method,
                        url,
                        params=params,
                        json=json,
                        headers=headers,
                        follow_redirects=False,
                        timeout=httpx.Timeout(
                            GITHUB_READ_TIMEOUT_S,
                            connect=GITHUB_CONNECT_TIMEOUT_S,
                        ),
                    )
            except httpx.HTTPError as error:
                if server_retry < len(retry_delays):
                    delay = retry_delays[server_retry]
                    server_retry += 1
                    await self.sleep(delay)
                    continue
                outcome = self._error_outcome(
                    "unavailable",
                    "github_unavailable",
                    "GitHub request failed",
                )
                raise GithubFailure(outcome) from error

            self._record_rate_limit(response)
            logger_method = (
                logger.warning if self._below_ten_percent(response) else logger.info
            )
            logger_method(
                "GitHub request method=%s path=%s status=%s",
                method,
                path,
                response.status_code,
            )
            status = response.status_code

            if 500 <= status < 600:
                if server_retry < len(retry_delays):
                    delay = retry_delays[server_retry]
                    server_retry += 1
                    await self.sleep(delay)
                    continue
                raise GithubFailure(
                    self._error_outcome(
                        "unavailable",
                        "github_unavailable",
                        "GitHub request failed",
                        http_status=status,
                    )
                )

            if status in (403, 429):
                if (
                    status == 403
                    and response.headers.get("x-ratelimit-remaining") == "0"
                ):
                    reset = self._reset_time(response)
                    raise GithubFailure(
                        GithubOutcome(
                            outcome="rate_limited",
                            decided_at=self.clock(),
                            http_status=status,
                            retry_at=reset,
                            retry_in=max(0, (reset - self.clock()).total_seconds()),
                            scope="primary",
                        )
                    )

                retry_in = self._retry_after(response.headers.get("retry-after"))
                if retry_in is not None:
                    if (
                        retry_in <= GITHUB_SECONDARY_LIMIT_WAIT_CAP_S
                        and not secondary_retried
                    ):
                        secondary_retried = True
                        await self.sleep(retry_in)
                        continue
                    raise GithubFailure(
                        GithubOutcome(
                            outcome="rate_limited",
                            decided_at=self.clock(),
                            http_status=status,
                            retry_at=self.clock() + timedelta(seconds=retry_in),
                            retry_in=retry_in,
                            scope="secondary",
                            error_code="secondary_rate_limit",
                            reason="GitHub secondary rate limit",
                        )
                    )
                if status == 403:
                    raise GithubFailure(
                        self._error_outcome(
                            "forbidden",
                            "github_forbidden",
                            "GitHub denied the requested permission",
                            http_status=status,
                        )
                    )
                raise GithubFailure(
                    GithubOutcome(
                        outcome="rate_limited",
                        decided_at=self.clock(),
                        http_status=status,
                        retry_in=0,
                        scope="secondary",
                        error_code="secondary_rate_limit",
                        reason="GitHub rate limit response had no Retry-After",
                    )
                )

            if status == 401:
                raise GithubFailure(
                    self._error_outcome(
                        "unauthorized",
                        "github_unauthorized",
                        "GitHub rejected the token",
                        http_status=status,
                    )
                )
            if status == 404:
                if missing == "repo":
                    code, reason = "repo_not_found", "GitHub repository was not found"
                elif missing == "revalidation":
                    code, reason = (
                        "issue_lookup_not_found",
                        "Stored issue lookup returned 404; state is unchanged",
                    )
                else:
                    code, reason = "issue_not_found", "GitHub issue was not found"
                raise GithubFailure(
                    self._error_outcome("not_found", code, reason, http_status=status)
                )
            if status == 410:
                raise GithubFailure(
                    self._error_outcome(
                        "gone",
                        "issues_disabled",
                        "GitHub Issues are disabled for this repository",
                        http_status=status,
                    )
                )
            if status == 422:
                raise GithubFailure(
                    self._error_outcome(
                        "invalid",
                        "comment_rejected",
                        "GitHub rejected the comment",
                        http_status=status,
                    )
                )
            if status >= 400:
                raise GithubFailure(
                    self._error_outcome(
                        "invalid",
                        "github_http_error",
                        f"GitHub returned HTTP {status}",
                        http_status=status,
                    )
                )
            return response

    def _record_rate_limit(self, response: httpx.Response) -> None:
        remaining = response.headers.get("x-ratelimit-remaining")
        if remaining is not None:
            try:
                self.rate_limit_remaining = int(remaining)
            except ValueError:
                self.rate_limit_remaining = None
        reset = response.headers.get("x-ratelimit-reset")
        if reset is not None:
            try:
                self.rate_limit_reset = datetime.fromtimestamp(float(reset), UTC)
            except (OverflowError, OSError, ValueError):
                self.rate_limit_reset = None
        poll_interval = response.headers.get("x-poll-interval")
        if poll_interval is not None:
            try:
                interval = int(poll_interval)
            except ValueError:
                return
            self.poll_interval_s = max(self.poll_interval_s or 0, interval)

    @staticmethod
    def _below_ten_percent(response: httpx.Response) -> bool:
        try:
            remaining = int(response.headers["x-ratelimit-remaining"])
            limit = int(response.headers["x-ratelimit-limit"])
        except (KeyError, ValueError):
            return False
        return limit > 0 and remaining / limit < 0.1

    def _reset_time(self, response: httpx.Response) -> datetime:
        try:
            return datetime.fromtimestamp(
                float(response.headers["x-ratelimit-reset"]), UTC
            )
        except (KeyError, OverflowError, OSError, ValueError):
            return self.clock()

    def _retry_after(self, value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return max(0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                return max(0, (retry_at - self.clock()).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    def _error_outcome(
        self,
        outcome: Literal[
            "unauthorized",
            "forbidden",
            "not_found",
            "gone",
            "invalid",
            "unavailable",
        ],
        error_code: str,
        reason: str,
        *,
        http_status: int | None = None,
    ) -> GithubOutcome:
        return GithubOutcome(
            outcome=outcome,
            decided_at=self.clock(),
            http_status=http_status,
            error_code=error_code,
            reason=reason,
        )

    def _json(self, response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as error:
            raise GithubFailure(
                GithubOutcome(
                    outcome="unavailable",
                    decided_at=self.clock(),
                    http_status=response.status_code,
                    error_code="invalid_github_response",
                    reason="GitHub returned invalid JSON",
                )
            ) from error
        if not isinstance(payload, dict):
            raise GithubFailure(
                GithubOutcome(
                    outcome="unavailable",
                    decided_at=self.clock(),
                    http_status=response.status_code,
                    error_code="invalid_github_response",
                    reason="GitHub returned an unexpected response",
                )
            )
        return payload

    def _json_list(self, response: httpx.Response) -> list[dict[str, Any]]:
        try:
            payload = response.json()
        except ValueError as error:
            raise GithubFailure(
                GithubOutcome(
                    outcome="unavailable",
                    decided_at=self.clock(),
                    http_status=response.status_code,
                    error_code="invalid_github_response",
                    reason="GitHub returned invalid JSON",
                )
            ) from error
        if not isinstance(payload, list) or not all(
            isinstance(item, dict) for item in payload
        ):
            raise GithubFailure(
                GithubOutcome(
                    outcome="unavailable",
                    decided_at=self.clock(),
                    http_status=response.status_code,
                    error_code="invalid_github_response",
                    reason="GitHub returned an unexpected response",
                )
            )
        return payload
