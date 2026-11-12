import hashlib
import json
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs

import httpx
import jwt


class FakeGithub:
    def __init__(
        self,
        *,
        repo="bot/fork",
        self_login="agent-bot",
        repo_id=100,
        app_public_key_pem=None,
        app_id="4242",
        app_slug="octo-agent-team",
        installation_id=9001,
        clock=None,
        default_branch="main",
        default_head_sha="0000000000000000000000000000000000000001",
    ):
        self.repo = repo
        self.self_login = self_login
        self.repo_id = repo_id
        self.app_public_key_pem = app_public_key_pem
        self.app_id = app_id
        self.app_slug = app_slug
        self.installation_id = installation_id
        self._clock = clock or (lambda: datetime.now(UTC))
        self.default_branch = default_branch
        self.refs = {f"heads/{default_branch}": default_head_sha}
        self.trees = {}
        self.git_commits = {}
        self.pulls = {}
        self.reviews = {}
        self.review_requests = {}
        self.statuses = {}
        self.installation_tokens = {}
        self.minted_token_bodies = []
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
        self,
        number,
        *,
        event="labeled",
        label="agent-team",
        created_at=None,
        actor="maintainer",
    ):
        item = {
            "event": event,
            "label": {"name": label},
            "created_at": (created_at or datetime.now(UTC)).isoformat(),
        }
        if actor is not None:
            item["actor"] = {"login": actor}
        self.events.setdefault(number, []).append(item)

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
        if request.headers.get("accept") != "application/vnd.github+json":
            return httpx.Response(
                400, headers=response_headers, json={"message": "bad accept"}
            )
        if request.headers.get("x-github-api-version") != "2022-11-28":
            return httpx.Response(
                400, headers=response_headers, json={"message": "bad version"}
            )

        path = request.url.path
        auth_header = request.headers.get("authorization", "")
        if self.app_public_key_pem is not None:
            app_resp = self._handle_app_auth_routes(
                request, path, auth_header, response_headers
            )
            if app_resp is not None:
                return app_resp
            if not self._valid_install_token(auth_header):
                return httpx.Response(
                    401, headers=response_headers, json={"message": "bad install auth"}
                )
            if path == "/user":
                return httpx.Response(
                    403,
                    headers=response_headers,
                    json={"message": "Resource not accessible by integration"},
                )
        elif auth_header != "Bearer test-token":
            return httpx.Response(
                401, headers=response_headers, json={"message": "bad auth"}
            )

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
                    "default_branch": self.default_branch,
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
        if len(parts) == 3 and parts[0] == "issues" and parts[1].isdigit():
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
                    comment_id = self.add_comment(
                        number, body=body, author=self.self_login
                    )
                    return httpx.Response(
                        201,
                        headers=response_headers,
                        json={
                            "id": comment_id,
                            "body": body,
                            "user": {"login": self.self_login},
                        },
                    )

        pr_or_git_resp = self._handle_git_and_pr_routes(
            request, path, query, response_headers
        )
        if pr_or_git_resp is not None:
            return pr_or_git_resp

        return httpx.Response(
            404, headers=response_headers, json={"message": "unexpected endpoint"}
        )

    def add_review(
        self,
        pr_number,
        *,
        author,
        state,
        commit_id,
        body="",
        submitted_at=None,
        review_id=None,
    ):
        items = self.reviews.setdefault(pr_number, [])
        if review_id is None:
            review_id = max((r["id"] for r in items), default=0) + 1
        items.append(
            {
                "id": review_id,
                "user": {"login": author} if author is not None else None,
                "state": state,
                "commit_id": commit_id,
                "body": body,
                "submitted_at": (submitted_at or self._clock()).isoformat(),
            }
        )
        return review_id

    def push_pr_commit(self, pr_number, new_sha):
        pr = self.pulls[pr_number]
        branch = pr["head"]["ref"]
        self.refs[f"heads/{branch}"] = new_sha
        pr["head"]["sha"] = new_sha
        pr["updated_at"] = self._clock().isoformat()

    def _handle_git_and_pr_routes(self, request, path, query, response_headers):
        repo_prefix = f"/repos/{self.repo}"
        if not path.startswith(f"{repo_prefix}/"):
            return None
        subpath = path.removeprefix(f"{repo_prefix}/")

        if subpath.startswith("git/ref/") and request.method == "GET":
            ref_name = subpath.removeprefix("git/ref/")
            if ref_name not in self.refs:
                return httpx.Response(
                    404, headers=response_headers, json={"message": "ref not found"}
                )
            return httpx.Response(
                200,
                headers=response_headers,
                json={
                    "ref": f"refs/{ref_name}",
                    "object": {"sha": self.refs[ref_name]},
                },
            )

        if subpath == "git/refs" and request.method == "POST":
            body = json.loads(request.content)
            ref_name = body["ref"].removeprefix("refs/")
            self.refs[ref_name] = body["sha"]
            return httpx.Response(
                201,
                headers=response_headers,
                json={"ref": body["ref"], "object": {"sha": body["sha"]}},
            )

        if subpath.startswith("git/refs/") and request.method == "PATCH":
            ref_name = subpath.removeprefix("git/refs/")
            body = json.loads(request.content)
            self.refs[ref_name] = body["sha"]
            for pr in self.pulls.values():
                if f"heads/{pr['head']['ref']}" == ref_name:
                    pr["head"]["sha"] = body["sha"]
            return httpx.Response(
                200,
                headers=response_headers,
                json={"ref": f"refs/{ref_name}", "object": {"sha": body["sha"]}},
            )

        if subpath == "git/trees" and request.method == "POST":
            raw = request.content
            tree_sha = hashlib.sha1(raw).hexdigest()
            self.trees[tree_sha] = json.loads(raw)
            return httpx.Response(201, headers=response_headers, json={"sha": tree_sha})

        if subpath == "git/commits" and request.method == "POST":
            raw = request.content
            commit_sha = hashlib.sha1(raw).hexdigest()
            self.git_commits[commit_sha] = json.loads(raw)
            return httpx.Response(
                201, headers=response_headers, json={"sha": commit_sha}
            )

        if subpath.startswith("issues/comments/") and request.method == "PATCH":
            comment_id = int(subpath.removeprefix("issues/comments/"))
            new_body = json.loads(request.content)["body"]
            for comment_list in self.comments.values():
                for item in comment_list:
                    if item["id"] == comment_id:
                        item["body"] = new_body
                        return httpx.Response(200, headers=response_headers, json=item)
            return httpx.Response(
                404, headers=response_headers, json={"message": "comment not found"}
            )

        if subpath.startswith("statuses/") and request.method == "POST":
            sha = subpath.removeprefix("statuses/")
            body = json.loads(request.content)
            self.statuses.setdefault(sha, []).append(body)
            return httpx.Response(201, headers=response_headers, json=body)

        if subpath == "pulls":
            if request.method == "GET":
                wanted_state = query.get("state", ["open"])[0]
                wanted_head = query.get("head", [None])[0]
                owner = self.repo.split("/", 1)[0]
                items = list(self.pulls.values())
                if wanted_state != "all":
                    items = [pr for pr in items if pr["state"] == wanted_state]
                if wanted_head:
                    items = [
                        pr
                        for pr in items
                        if f"{owner}:{pr['head']['ref']}" == wanted_head
                    ]
                return self._page(items, query, response_headers)
            if request.method == "POST":
                body = json.loads(request.content)
                pr_number = (
                    max(
                        [*self.issues.keys(), *self.pulls.keys()],
                        default=0,
                    )
                    + 1
                )
                head_ref = body["head"]
                head_sha = self.refs.get(f"heads/{head_ref}", "")
                pr = {
                    "number": pr_number,
                    "html_url": f"https://github.com/{self.repo}/pull/{pr_number}",
                    "state": "open",
                    "merged": False,
                    "draft": bool(body.get("draft", False)),
                    "title": body["title"],
                    "body": body.get("body", ""),
                    "head": {"ref": head_ref, "sha": head_sha},
                    "base": {"ref": body["base"]},
                    "user": {"login": self.self_login},
                    "created_at": self._clock().isoformat(),
                    "updated_at": self._clock().isoformat(),
                }
                self.pulls[pr_number] = pr
                # On GitHub, every PR also has an issue-comments thread at /issues/{pr_number}/comments!
                self.issues.setdefault(
                    pr_number,
                    {
                        "number": pr_number,
                        "node_id": f"PR_{pr_number}",
                        "title": body["title"],
                        "body": body.get("body", ""),
                        "state": "open",
                        "labels": [],
                        "user": {"login": self.self_login},
                        "locked": False,
                        "pull_request": {
                            "url": f"https://api.github.com/repos/{self.repo}/pulls/{pr_number}"
                        },
                        "created_at": pr["created_at"],
                        "updated_at": pr["updated_at"],
                    },
                )
                self.comments.setdefault(pr_number, [])
                return httpx.Response(201, headers=response_headers, json=pr)

        pr_parts = subpath.split("/")
        if len(pr_parts) == 2 and pr_parts[0] == "pulls" and request.method == "GET":
            pr_number = int(pr_parts[1])
            if pr_number not in self.pulls:
                return httpx.Response(
                    404, headers=response_headers, json={"message": "PR not found"}
                )
            return httpx.Response(
                200, headers=response_headers, json=self.pulls[pr_number]
            )
        if len(pr_parts) == 3 and pr_parts[0] == "pulls":
            pr_number = int(pr_parts[1])
            if pr_number not in self.pulls:
                return httpx.Response(
                    404, headers=response_headers, json={"message": "PR not found"}
                )
            if pr_parts[2] == "requested_reviewers" and request.method == "POST":
                reviewers = list(json.loads(request.content).get("reviewers", []))
                self.review_requests.setdefault(pr_number, []).append(reviewers)
                return httpx.Response(
                    201, headers=response_headers, json=self.pulls[pr_number]
                )
            if pr_parts[2] == "reviews" and request.method == "GET":
                return self._page(
                    self.reviews.get(pr_number, []), query, response_headers
                )
        return None

    def _verify_app_jwt(self, auth_header: str) -> bool:
        if not auth_header.startswith("Bearer ") or self.app_public_key_pem is None:
            return False
        raw = auth_header.removeprefix("Bearer ")
        now_ts = int(self._clock().timestamp())
        try:
            claims = jwt.decode(
                raw,
                self.app_public_key_pem,
                algorithms=["RS256"],
                options={"verify_exp": False, "verify_iat": False},
            )
        except jwt.PyJWTError:
            return False
        return (
            claims.get("iss") == self.app_id
            and isinstance(claims.get("iat"), int)
            and isinstance(claims.get("exp"), int)
            and claims["iat"] <= now_ts <= claims["exp"]
        )

    def _valid_install_token(self, auth_header: str) -> bool:
        if not auth_header.startswith("Bearer "):
            return False
        token = auth_header.removeprefix("Bearer ")
        expires_at = self.installation_tokens.get(token)
        return expires_at is not None and self._clock() < expires_at

    def _handle_app_auth_routes(self, request, path, auth_header, response_headers):
        repo_prefix = f"/repos/{self.repo}"
        if path == "/app" and request.method == "GET":
            if not self._verify_app_jwt(auth_header):
                return httpx.Response(
                    401, headers=response_headers, json={"message": "bad app jwt"}
                )
            return httpx.Response(
                200,
                headers=response_headers,
                json={"id": int(self.app_id), "slug": self.app_slug},
            )
        if path == f"{repo_prefix}/installation" and request.method == "GET":
            if not self._verify_app_jwt(auth_header):
                return httpx.Response(
                    401, headers=response_headers, json={"message": "bad app jwt"}
                )
            return httpx.Response(
                200,
                headers=response_headers,
                json={"id": self.installation_id, "app_slug": self.app_slug},
            )
        if (
            path == f"/app/installations/{self.installation_id}/access_tokens"
            and request.method == "POST"
        ):
            if not self._verify_app_jwt(auth_header):
                return httpx.Response(
                    401, headers=response_headers, json={"message": "bad app jwt"}
                )
            body = json.loads(request.content) if request.content else {}
            self.minted_token_bodies.append(body)
            token = f"ghs_install_{len(self.installation_tokens) + 1}"
            expires_at = self._clock() + timedelta(hours=1)
            self.installation_tokens[token] = expires_at
            return httpx.Response(
                201,
                headers=response_headers,
                json={
                    "token": token,
                    "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
                },
            )
        return None

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
