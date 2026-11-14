from __future__ import annotations

import httpx


class MemoryClient:
    def __init__(self, *, base_url: str, http: httpx.AsyncClient) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = http

    async def push(self, *, before: str | None, after: str) -> httpx.Response:
        return await self.http.post(
            f"{self.base_url}/v1/repo/push",
            json={"branch": "main", "before": before, "after": after},
            timeout=120.0,
        )

    async def close_by_issue(
        self,
        *,
        issue_ref: str,
        closing_commit: str,
        reason: str,
    ) -> list[str]:
        try:
            response = await self.http.post(
                f"{self.base_url}/v1/facts/close-by-issue",
                json={
                    "issue_ref": issue_ref,
                    "closing_commit": closing_commit,
                    "reason": reason,
                },
                timeout=10.0,
            )
            if response.status_code == 200:
                return response.json().get("closed_fact_ids", [])
        except httpx.HTTPError:
            pass
        return []
