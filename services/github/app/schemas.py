from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from typing import Any, Literal, Self

from agentteam_contracts.models import Contract
from pydantic import Field, model_validator

Clock = Callable[[], datetime]
Sleep = Callable[[float], Awaitable[None]]


class GithubIssue(Contract):
    number: int = Field(ge=1)
    node_id: str
    title: str = Field(max_length=500)
    body: str = ""
    state: Literal["open", "closed"]
    labels: list[str]
    author_login: str
    locked: bool
    is_pull_request: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_api(cls, item: Mapping[str, Any]) -> "GithubIssue":
        author = item.get("user") or {}
        return cls(
            number=item["number"],
            node_id=item.get("node_id", ""),
            title=item.get("title") or "",
            body=item.get("body") or "",
            state=item["state"],
            labels=[label["name"] for label in item.get("labels", [])],
            author_login=author.get("login", ""),
            locked=item.get("locked", False),
            is_pull_request="pull_request" in item,
            created_at=item["created_at"],
            updated_at=item["updated_at"],
        )


class GithubOutcome(Contract):
    outcome: Literal[
        "ok",
        "unauthorized",
        "forbidden",
        "rate_limited",
        "not_found",
        "gone",
        "invalid",
        "unavailable",
    ]
    decided_at: datetime
    http_status: int | None = None
    retry_at: datetime | None = None
    retry_in: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    scope: Literal["primary", "secondary"] | None = None
    error_code: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_outcome_fields(self) -> Self:
        if self.outcome == "ok":
            if any(
                value is not None
                for value in (
                    self.retry_at,
                    self.retry_in,
                    self.scope,
                    self.error_code,
                    self.reason,
                )
            ):
                raise ValueError("ok outcomes cannot contain error or retry fields")
        elif self.outcome == "rate_limited":
            if self.retry_in is None or self.scope is None:
                raise ValueError("rate_limited outcomes require retry_in and scope")
        elif not self.error_code or not self.reason:
            raise ValueError("non-rate-limit errors require error_code and reason")
        return self


class PollResult(Contract):
    issues: list[GithubIssue]
    pages: int = Field(ge=0)
    truncated: bool
    max_updated_at: datetime | None = None


class IssueView(Contract):
    issue_key: str
    number: int
    state: Literal[
        "ignored",
        "skipped",
        "pending_start",
        "submitted",
        "finished",
        "failed",
        "cancelled",
    ]
    reason: str | None = None
    start_attempts: int = Field(ge=0)
    result_status: Literal["committed", "quota_exhausted", "throttled"] | None = None


class CycleSummary(Contract):
    ran_at: datetime
    started: int = Field(default=0, ge=0)
    deferred: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    finished: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    comments_posted: int = Field(default=0, ge=0)
    aborted_reason: str | None = None


class RepoInfo(Contract):
    id: int = Field(ge=1)
    full_name: str
    fork: bool
    owner_login: str
    has_issues: bool


class LabelEvent(Contract):
    event: str
    label_name: str
    created_at: datetime


class GithubComment(Contract):
    id: int = Field(ge=1)
    body: str
    author_login: str
    created_at: datetime
