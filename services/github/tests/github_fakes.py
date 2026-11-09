import json
from datetime import UTC, datetime
from urllib.parse import parse_qs

import httpx


class FakeGithub:
    def __init__(self, *, repo="bot/fork", self_login="agent-bot", repo_id=100):
        self.repo = repo
        self.self_login = self_login
        self.repo_id = repo_id
        self.issues = {}
        self.events = {}
        self.comments = {}
        self.labels = set()
        self.calls = []
        self.failures = []
        self.path_failures = {}

    @property
    def transport(self):
        return httpx.MockTransport(self.handle)

    def add_issue(
        self,
        number,
        *,
        title="Example issue",
        body="Example body",
        labels=("agent-team",),
        state="open",
        author="maintainer",
        locked=False,
        pull_request=False,
        created_at=None,
        updated_at=None,
    ):
        created_at = created_at or datetime.now(UTC)
        updated_at = updated_at or created_at
        item = {
            "number": number,
            "node_id": f"I_{number}",
            "title": title,
            "body": body,
            "state": state,
            "labels": [{"name": label} for label in labels],
            "user": {"login": author},
            "locked": locked,
            "created_at": created_at.isoformat(),
            "updated_at": updated_at.isoformat(),
        }
        if pull_request:
            item["pull_request"] = {
                "url": f"https://api.github.com/repos/{self.repo}/pulls/{number}"
            }
        self.issues[number] = item
        self.comments.setdefault(number, [])
        self.events.setdefault(number, [])
        return item

    def add_event(
        self, number, *, event="labeled", label="agent-team", created_at=None
    ):
        self.events.setdefault(number, []).append(
            {
                "event": event,
                "label": {"name": label},
                "created_at": (created_at or datetime.now(UTC)).isoformat(),
            }
        )

    def add_comment(self, number, *, body, author="agent-bot", comment_id=None):
        comments = self.comments.setdefault(number, [])
        if comment_id is None:
            comment_id = max((comment["id"] for comment in comments), default=0) + 1
        comments.append(
            {
                "id": comment_id,
                "body": body,
                "user": {"login": author},
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        return comment_id

    def fail_next(self, status, headers=None):
        self.failures.append((status, dict(headers or {})))

    def fail_path_next(self, path, status, headers=None):
        self.path_failures.setdefault(path, []).append((status, dict(headers or {})))

    def handle(self, request):
        query = parse_qs(request.url.query.decode())
        self.calls.append((request.method, request.url.path, query))
        response_headers = {
            "x-ratelimit-remaining": "5000",
            "x-ratelimit-reset": str(int(datetime.now(UTC).timestamp()) + 3600),
        }
        path_failures = self.path_failures.get(request.url.path, [])
        if path_failures:
            status, headers = path_failures.pop(0)
            return httpx.Response(
                status,
                headers={**response_headers, **headers},
                json={"message": "simulated response"},
            )
        if self.failures:
            status, headers = self.failures.pop(0)
            return httpx.Response(
                status,
                headers={**response_headers, **headers},
                json={"message": "simulated response"},
            )

        if request.url.host != "api.github.com":
            return httpx.Response(404, json={"message": "unexpected host"})
        if request.headers.get("authorization") != "Bearer test-token":
            return httpx.Response(
                401, headers=response_headers, json={"message": "bad auth"}
            )
        if request.headers.get("accept") != "application/vnd.github+json":
            return httpx.Response(
                400, headers=response_headers, json={"message": "bad accept"}
            )
        if request.headers.get("x-github-api-version") != "2022-11-28":
            return httpx.Response(
                400, headers=response_headers, json={"message": "bad version"}
            )

        path = request.url.path
        if request.method == "GET" and path == "/user":
            return httpx.Response(
                200, headers=response_headers, json={"login": self.self_login}
            )

        repo_prefix = f"/repos/{self.repo}"
        if request.method == "GET" and path == repo_prefix:
            owner = self.repo.split("/", 1)[0]
            return httpx.Response(
                200,
                headers=response_headers,
                json={
                    "id": self.repo_id,
                    "full_name": self.repo,
                    "fork": True,
                    "owner": {"login": owner},
                    "has_issues": True,
                },
            )

        if path == f"{repo_prefix}/labels":
            if request.method == "GET":
                labels = [{"name": label} for label in sorted(self.labels)]
                return self._page(labels, query, response_headers)
            if request.method == "POST":
                label = json.loads(request.content)["name"]
                self.labels.add(label)
                return httpx.Response(
                    201, headers=response_headers, json={"name": label}
                )

        if request.method == "GET" and path == f"{repo_prefix}/issues":
            items = list(self.issues.values())
            wanted_labels = set(query.get("labels", [""])[0].split(",")) - {""}
            wanted_state = query.get("state", [None])[0]
            since = query.get("since", [None])[0]
            if wanted_labels:
                items = [
                    item
                    for item in items
                    if wanted_labels <= {label["name"] for label in item["labels"]}
                ]
            if wanted_state:
                items = [item for item in items if item["state"] == wanted_state]
            if since:
                cutoff = datetime.fromisoformat(since)
                items = [
                    item
                    for item in items
                    if datetime.fromisoformat(item["updated_at"]) >= cutoff
                ]
            items.sort(key=lambda item: item["updated_at"])
            return self._page(items, query, response_headers)

        parts = path.removeprefix(f"{repo_prefix}/").split("/")
        if len(parts) == 2 and parts[0] == "issues" and request.method == "GET":
            number = int(parts[1])
            if number not in self.issues:
                return httpx.Response(
                    404, headers=response_headers, json={"message": "not found"}
                )
            return httpx.Response(
                200, headers=response_headers, json=self.issues[number]
            )
        if len(parts) == 3 and parts[0] == "issues":
            number = int(parts[1])
            if request.method == "GET" and parts[2] == "events":
                if number not in self.issues:
                    return httpx.Response(
                        404, headers=response_headers, json={"message": "not found"}
                    )
                return self._page(self.events.get(number, []), query, response_headers)
            if parts[2] == "comments":
                if number not in self.issues:
                    return httpx.Response(
                        404, headers=response_headers, json={"message": "not found"}
                    )
                if request.method == "GET":
                    return self._page(
                        self.comments.get(number, []), query, response_headers
                    )
                if request.method == "POST":
                    body = json.loads(request.content)["body"]
                    comment_id = self.add_comment(number, body=body)
                    return httpx.Response(
                        201,
                        headers=response_headers,
                        json={
                            "id": comment_id,
                            "body": body,
                            "user": {"login": self.self_login},
                        },
                    )

        return httpx.Response(
            404, headers=response_headers, json={"message": "unexpected endpoint"}
        )

    @staticmethod
    def _page(items, query, headers):
        page = int(query.get("page", ["1"])[0])
        per_page = int(query.get("per_page", ["100"])[0])
        start = (page - 1) * per_page
        return httpx.Response(
            200, headers=headers, json=items[start : start + per_page]
        )


class FakeWorkflow:
    def __init__(self):
        self.responses = {}
        self.calls = []
        self.queries = []
        self.accepted_starts = {}

    @property
    def transport(self):
        return httpx.MockTransport(self.handle)

    def set_response(self, method, path, status, payload=None):
        self.responses[(method, path)] = (status, payload)

    def handle(self, request):
        payload = json.loads(request.content) if request.content else None
        self.calls.append((request.method, request.url.path, payload))
        self.queries.append(parse_qs(request.url.query.decode()))
        path = request.url.path
        if (
            request.method == "POST"
            and path.endswith("/start")
            and path in self.accepted_starts
        ):
            return httpx.Response(200, json=self.accepted_starts[path])
        status, response_payload = self.responses.get(
            (request.method, path), (404, {"detail": "not found"})
        )
        if (
            request.method == "POST"
            and path.endswith("/start")
            and status in (200, 202)
        ):
            self.accepted_starts[path] = response_payload
        return httpx.Response(status, json=response_payload)
