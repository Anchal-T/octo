from typing import Literal
from urllib.parse import quote

import httpx
from agentteam_contracts import IssueRunRequest, IssueRunResult, RunStatus

from .settings import WORKFLOW_START_TIMEOUT_S

StartResult = Literal["submitted", "rejected", "conflict", "unavailable"]
RunState = Literal["running", "unknown"] | IssueRunResult


class WorkflowUnavailable(Exception):
    pass


class WorkflowClient:
    def __init__(self, *, base_url: str, http: httpx.AsyncClient) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = http

    async def start(self, request: IssueRunRequest) -> StartResult:
        issue_key = quote(request.issue_id, safe="")
        try:
            response = await self.http.post(
                f"{self.base_url}/v1/issues/{issue_key}/start",
                json=request.model_dump(mode="json"),
                timeout=WORKFLOW_START_TIMEOUT_S,
            )
        except httpx.HTTPError:
            return "unavailable"

        if response.status_code in (200, 202):
            return "submitted"
        if response.status_code == 422:
            return "rejected"
        if response.status_code == 409:
            try:
                state = await self.status(request.issue_id, attempt=request.attempt)
            except WorkflowUnavailable:
                return "unavailable"
            return "conflict" if state == "unknown" else "submitted"
        return "unavailable"

    async def status(self, issue_key: str, attempt: int = 1) -> RunState:
        key = quote(issue_key, safe="")
        try:
            response = await self.http.get(
                f"{self.base_url}/v1/issues/{key}/run",
                params={"attempt": attempt},
                timeout=WORKFLOW_START_TIMEOUT_S,
            )
        except httpx.HTTPError as error:
            raise WorkflowUnavailable("workflow status is unavailable") from error

        if response.status_code == 404:
            return "unknown"
        try:
            payload = response.json()
            if response.status_code == 202:
                return RunStatus.model_validate(payload).state
            if response.status_code == 200:
                return IssueRunResult.model_validate(payload)
        except (TypeError, ValueError) as error:
            raise WorkflowUnavailable("workflow returned an invalid status") from error
        raise WorkflowUnavailable("workflow status is unavailable")
