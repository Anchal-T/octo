import asyncio
from datetime import UTC, datetime

import httpx
import pytest

from services.github.app import main as github_main
from services.github.app.client import GithubClient, GithubFailure
from services.github.app.main import build_runtime, close_runtime, create_app
from services.github.app.settings import GithubSettings
from services.github.tests.github_fakes import FakeGithub, FakeWorkflow

SECRET = "not-a-real-github-token"
NOW = datetime(2026, 11, 6, 12, tzinfo=UTC)


def test_create_app_does_not_read_environment(monkeypatch):
    calls = []

    def from_env(cls, env=None):
        calls.append(env)
        return cls(token="test-token", repo="agent-bot/fork")

    monkeypatch.setattr(GithubSettings, "from_env", classmethod(from_env))

    create_app()

    assert calls == []


@pytest.mark.anyio
async def test_lifespan_prepares_runtime_and_cancels_background_poller(
    monkeypatch, tmp_path
):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_workflow = FakeWorkflow()

    def dispatch(request):
        if request.url.host == "api.github.com":
            return fake_github.handle(request)
        return fake_workflow.handle(request)

    settings = GithubSettings(
        token="test-token",
        repo=fake_github.repo,
        sqlite_path=str(tmp_path / "github.sqlite"),
    )
    runtime = build_runtime(
        settings,
        transport=httpx.MockTransport(dispatch),
        clock=lambda: NOW,
    )
    started = asyncio.Event()
    cancelled = asyncio.Event()

    def from_env(cls, env=None):
        return settings

    async def run_forever(stop):
        started.set()
        try:
            await stop.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(GithubSettings, "from_env", classmethod(from_env))
    monkeypatch.setattr(github_main, "build_runtime", lambda _settings: runtime)
    runtime.poller.run_forever = run_forever
    app = github_main.create_app()

    async with app.router.lifespan_context(app):
        await started.wait()
        assert app.state.runtime is runtime
        assert runtime.poller.auth_state == "ok"

    assert cancelled.is_set()


@pytest.mark.anyio
async def test_missing_startup_settings_exit_with_code_two(capsys, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    app = create_app()

    with pytest.raises(SystemExit) as exited:
        async with app.router.lifespan_context(app):
            pytest.fail("invalid settings should stop startup")

    assert exited.value.code == 2
    assert "GITHUB_TOKEN" in capsys.readouterr().err


@pytest.mark.anyio
async def test_token_is_absent_from_settings_repr_api_errors_and_logs(caplog, tmp_path):
    fake_github = FakeGithub(repo="agent-bot/fork", self_login="agent-bot")
    fake_workflow = FakeWorkflow()

    def dispatch(request):
        if request.url.host == "api.github.com":
            return fake_github.handle(request)
        return fake_workflow.handle(request)

    settings = GithubSettings(
        token=SECRET,
        repo=fake_github.repo,
        sqlite_path=str(tmp_path / "github.sqlite"),
    )
    runtime = build_runtime(
        settings,
        transport=httpx.MockTransport(dispatch),
        clock=lambda: NOW,
    )
    app = create_app(runtime)
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )

    def unauthorized(request):
        return httpx.Response(401, json={"message": "bad credentials"})

    async def sleep(_seconds):
        return None

    github_http = httpx.AsyncClient(transport=httpx.MockTransport(unauthorized))
    github = GithubClient(
        token=SECRET,
        repo="agent-bot/fork",
        http=github_http,
        clock=lambda: NOW,
        sleep=sleep,
    )
    try:
        with caplog.at_level("INFO"):
            async with http:
                status = await http.get("/v1/status")
                issues = await http.get("/v1/issues")
            with pytest.raises(GithubFailure) as caught:
                await github.get_self()
    finally:
        await github_http.aclose()
        await close_runtime(runtime)

    assert SECRET not in repr(settings)
    assert SECRET not in status.text
    assert SECRET not in issues.text
    assert SECRET not in str(caught.value)
    assert SECRET not in caplog.text
    assert "method=GET path=/user status=401" in caplog.text
