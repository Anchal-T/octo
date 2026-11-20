import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Literal

import httpx
from agentteam_contracts import PrFileChange

from .auth import GithubAppTokenSource, StaticTokenSource, TokenSource
from .git_objects import changed_tree_sha, object_sha
from .schemas import (
    Clock,
    GithubComment,
    GithubIssue,
    GithubOutcome,
    GithubPullRequest,
    GithubReview,
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
        token: str = "",
        repo: str,
        http: httpx.AsyncClient,
        clock: Clock,
        sleep: Sleep,
        token_source: TokenSource | None = None,
    ) -> None:
        self.token = token
        self.token_source: TokenSource = token_source or StaticTokenSource(token)
        if isinstance(self.token_source, GithubAppTokenSource):
            self.token_source.bind_requester(self._app_jwt_request)
        self.repo = repo
        self.http = http
        self.clock = clock
        self.sleep = sleep
        self._request_lock = asyncio.Lock()
        self.rate_limit_remaining: int | None = None
        self.rate_limit_reset: datetime | None = None
        self.poll_interval_s: int | None = None

    @property
    def auth_mode(self) -> Literal["app", "pat"]:
        return self.token_source.auth_mode

    @property
    def rate_limit_floor_reached(self) -> bool:
        return (
            self.rate_limit_remaining is not None
            and self.rate_limit_remaining < GITHUB_RATE_LIMIT_FLOOR
        )

    async def get_self(self) -> str:
        if self.auth_mode == "app":
            return await self.token_source.resolve_self_login()
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
            default_branch=item.get("default_branch") or "main",
        )

    async def _app_jwt_request(
        self,
        method: str,
        path: str,
        app_jwt: str,
        json_body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        response = await self._request(
            method,
            path,
            json=json_body,
            bearer_override=app_jwt,
        )
        return self._json(response)

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
                    actor_login=(item.get("actor") or {}).get("login"),
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
            retry_ambiguous=False,
        )
        return self._json(response)["id"]

    async def update_comment(self, comment_id: int, body: str) -> int:
        response = await self._request(
            "PATCH",
            f"/repos/{self.repo}/issues/comments/{comment_id}",
            json={"body": body},
            missing="issue",
            retry_ambiguous=False,
        )
        return self._json(response)["id"]

    async def get_branch_head(self, branch: str) -> str:
        response = await self._request(
            "GET",
            f"/repos/{self.repo}/git/ref/heads/{branch}",
            missing="issue",
        )
        obj = self._json(response).get("object") or {}
        sha = str(obj.get("sha") or "").strip()
        if not sha:
            raise GithubFailure(
                self._error_outcome(
                    "unavailable",
                    "invalid_github_response",
                    "GitHub ref response did not contain a commit SHA",
                    http_status=response.status_code,
                )
            )
        return sha

    async def create_tree(
        self, *, base_tree: str, files: Sequence[PrFileChange]
    ) -> str:
        entries: list[dict[str, Any]] = []
        for item in files:
            if item.content is None:
                entries.append(
                    {
                        "path": item.path,
                        "mode": "100644",
                        "type": "blob",
                        "sha": None,
                    }
                )
            else:
                entries.append(
                    {
                        "path": item.path,
                        "mode": "100644",
                        "type": "blob",
                        "content": item.content,
                    }
                )
        response = await self._request(
            "POST",
            f"/repos/{self.repo}/git/trees",
            json={"base_tree": await self._commit_tree(base_tree), "tree": entries},
            retry_ambiguous=False,
        )
        return self._json(response)["sha"]

    async def _commit_tree(self, sha: str) -> str:
        response = await self._request(
            "GET", f"/repos/{self.repo}/git/commits/{sha}", missing="issue"
        )
        return self._json(response)["tree"]["sha"]

    async def _tree_entries(self, sha: str) -> list[dict[str, Any]]:
        response = await self._request(
            "GET", f"/repos/{self.repo}/git/trees/{sha}", missing="issue"
        )
        data = self._json(response)
        if data.get("truncated"):
            raise GithubFailure(
                self._error_outcome(
                    "unavailable",
                    "truncated_tree",
                    "Cannot reconcile a truncated Git tree",
                )
            )
        return data["tree"]

    async def _git_object_exists(self, kind: str, sha: str) -> bool:
        try:
            await self._request(
                "GET", f"/repos/{self.repo}/git/{kind}/{sha}", missing="issue"
            )
        except GithubFailure as error:
            if error.outcome.outcome == "not_found":
                return False
            raise
        return True

    async def find_created_tree(
        self, *, base_tree: str, files: Sequence[PrFileChange]
    ) -> tuple[str] | None:
        base_sha = await self._commit_tree(base_tree)
        sha = await changed_tree_sha(base_sha, files, self._tree_entries)
        return (sha,) if await self._git_object_exists("trees", sha) else None

    async def create_commit(
        self,
        *,
        message: str,
        tree_sha: str,
        parents: Sequence[str],
        started_at: datetime,
        author_login: str,
    ) -> str:
        identity = self._commit_identity(started_at, author_login)
        response = await self._request(
            "POST",
            f"/repos/{self.repo}/git/commits",
            json={
                "message": message.rstrip("\n") + "\n",
                "tree": tree_sha,
                "parents": list(parents),
                "author": identity,
                "committer": identity,
            },
            retry_ambiguous=False,
        )
        return self._json(response)["sha"]

    @staticmethod
    def _commit_identity(started_at: datetime, author_login: str) -> dict[str, str]:
        return {
            "name": author_login,
            "email": f"{author_login}@users.noreply.github.com",
            "date": started_at.astimezone(UTC).isoformat(timespec="seconds"),
        }

    async def find_created_commit(
        self,
        *,
        message: str,
        tree_sha: str,
        parents: Sequence[str],
        started_at: datetime,
        author_login: str,
    ) -> tuple[str] | None:
        identity = self._commit_identity(started_at, author_login)
        epoch = int(datetime.fromisoformat(identity["date"]).timestamp())
        who = f"{identity['name']} <{identity['email']}> {epoch} +0000"
        headers = [f"tree {tree_sha}", *(f"parent {sha}" for sha in parents)]
        headers += [f"author {who}", f"committer {who}"]
        prefix = "\n".join(headers) + "\n\n"
        for ending in ("\n", ""):
            body = prefix + message.rstrip("\n") + ending
            sha = object_sha("commit", body.encode("utf-8"))
            if await self._git_object_exists("commits", sha):
                return (sha,)
        return None

    async def upsert_branch_ref(self, branch: str, sha: str) -> None:
        try:
            current_sha = await self.get_branch_head(branch)
        except GithubFailure as error:
            if error.outcome.outcome != "not_found":
                raise
            current_sha = None
        if current_sha == sha:
            return
        if current_sha is not None:
            await self._request(
                "PATCH",
                f"/repos/{self.repo}/git/refs/heads/{branch}",
                json={"sha": sha, "force": True},
                missing="issue",
                retry_ambiguous=False,
            )
        else:
            await self._request(
                "POST",
                f"/repos/{self.repo}/git/refs",
                json={"ref": f"refs/heads/{branch}", "sha": sha},
                retry_ambiguous=False,
            )

    async def find_open_pr(self, *, head_branch: str) -> GithubPullRequest | None:
        owner = self.repo.split("/", 1)[0]
        response = await self._request(
            "GET",
            f"/repos/{self.repo}/pulls",
            params={
                "state": "open",
                "head": f"{owner}:{head_branch}",
                "per_page": 10,
                "page": 1,
            },
        )
        items = self._json_list(response)
        return GithubPullRequest.from_api(items[0]) if items else None

    async def ensure_draft_pr(
        self, *, title: str, body: str, head_branch: str, base_branch: str
    ) -> GithubPullRequest:
        existing = await self.find_open_pr(head_branch=head_branch)
        if existing is not None:
            return existing
        return await self.create_draft_pr(
            title=title, body=body, head_branch=head_branch, base_branch=base_branch
        )

    async def find_created_draft_pr(
        self,
        *,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str,
        head_sha: str,
        author_login: str,
    ) -> tuple[GithubPullRequest] | None:
        owner = self.repo.split("/", 1)[0]
        for page in range(1, GITHUB_MAX_PAGES + 1):
            response = await self._request(
                "GET",
                f"/repos/{self.repo}/pulls",
                params={
                    "state": "all",
                    "head": f"{owner}:{head_branch}",
                    "per_page": GITHUB_PAGE_SIZE,
                    "page": page,
                },
            )
            items = self._json_list(response)
            for item in items:
                pr = GithubPullRequest.from_api(item)
                if (
                    pr.head_sha == head_sha
                    and pr.base_ref == base_branch
                    and pr.author_login.casefold() == author_login.casefold()
                    and item.get("title") == title
                    and (item.get("body") or "") == body
                ):
                    return (pr,)
            if len(items) < GITHUB_PAGE_SIZE:
                break
        return None

    async def create_draft_pr(
        self,
        *,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str,
    ) -> GithubPullRequest:
        response = await self._request(
            "POST",
            f"/repos/{self.repo}/pulls",
            json={
                "title": title,
                "body": body,
                "head": head_branch,
                "base": base_branch,
                "draft": True,
            },
            retry_ambiguous=False,
        )
        return GithubPullRequest.from_api(self._json(response))

    async def get_pull_request(self, pr_number: int) -> GithubPullRequest:
        response = await self._request(
            "GET",
            f"/repos/{self.repo}/pulls/{pr_number}",
            missing="issue",
        )
        return GithubPullRequest.from_api(self._json(response))

    async def request_reviewers(self, pr_number: int, reviewers: Sequence[str]) -> None:
        clean = [login for login in reviewers if login.strip()]
        if not clean:
            return
        await self._request(
            "POST",
            f"/repos/{self.repo}/pulls/{pr_number}/requested_reviewers",
            json={"reviewers": clean},
            missing="issue",
            retry_ambiguous=False,
        )

    async def find_review_request(
        self, *, pr_number: int, reviewers: Sequence[str], head_sha: str
    ) -> tuple[None] | None:
        response = await self._request(
            "GET",
            f"/repos/{self.repo}/pulls/{pr_number}/requested_reviewers",
            missing="issue",
        )
        requested = {user["login"].casefold() for user in self._json(response)["users"]}
        expected = {login.casefold() for login in reviewers if login.strip()}
        if not expected <= requested:
            requested.update(
                review.author_login.casefold()
                for review in await self.list_reviews(pr_number)
                if review.commit_id == head_sha and review.submitted_at is not None
            )
        return (None,) if expected <= requested else None

    async def list_reviews(self, pr_number: int) -> list[GithubReview]:
        reviews: list[GithubReview] = []
        for page in range(1, GITHUB_MAX_PAGES + 1):
            response = await self._request(
                "GET",
                f"/repos/{self.repo}/pulls/{pr_number}/reviews",
                params={"per_page": GITHUB_PAGE_SIZE, "page": page},
                missing="issue",
            )
            items = self._json_list(response)
            reviews.extend(GithubReview.from_api(item) for item in items)
            if len(items) < GITHUB_PAGE_SIZE:
                break
        return reviews

    async def create_commit_status(
        self,
        sha: str,
        *,
        state: Literal["pending", "success", "failure", "error"],
        description: str,
        context: str = "agent-team",
    ) -> None:
        await self._request(
            "POST",
            f"/repos/{self.repo}/statuses/{sha}",
            json={
                "state": state,
                "description": description[:140],
                "context": context,
            },
            missing="issue",
            retry_ambiguous=False,
        )

    async def find_commit_status(
        self, *, sha: str, state: str, description: str, context: str
    ) -> tuple[None] | None:
        for page in range(1, GITHUB_MAX_PAGES + 1):
            response = await self._request(
                "GET",
                f"/repos/{self.repo}/commits/{sha}/statuses",
                params={"per_page": GITHUB_PAGE_SIZE, "page": page},
                missing="issue",
            )
            statuses = self._json_list(response)
            for status in statuses:
                if status["context"] == context:
                    if (
                        status["state"] == state
                        and status.get("description") == description[:140]
                    ):
                        return (None,)
                    return None
            if len(statuses) < GITHUB_PAGE_SIZE:
                break
        return None

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        missing: Literal["repo", "issue", "revalidation"] = "repo",
        bearer_override: str | None = None,
        retry_ambiguous: bool = True,
    ) -> httpx.Response:
        server_retry = 0
        secondary_retried = False
        app_token_refreshed = False
        retry_delays = (1, 2, 4) if retry_ambiguous else ()
        url = f"{GITHUB_API_BASE_URL}{path}"
        bearer = (
            bearer_override
            if bearer_override is not None
            else await self.token_source.bearer_token()
        )

        while True:
            headers = {
                "Authorization": f"Bearer {bearer}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
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
                if (
                    bearer_override is None
                    and self.auth_mode == "app"
                    and not app_token_refreshed
                ):
                    app_token_refreshed = True
                    bearer = await self.token_source.bearer_token(force_refresh=True)
                    continue
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
