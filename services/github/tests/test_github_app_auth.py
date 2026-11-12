from datetime import UTC, datetime, timedelta

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from services.github.app.auth import GithubAppTokenSource
from services.github.app.client import GithubClient
from services.github.app.main import build_runtime, close_runtime, create_app
from services.github.app.poller import Poller
from services.github.app.settings import GithubSettings, SettingsError
from services.github.app.store import Store
from services.github.app.workflow_client import WorkflowClient
from services.github.tests.github_fakes import FakeGithub, FakeWorkflow

NOW = datetime(2026, 11, 12, 12, tzinfo=UTC)


def _generate_rsa_keypair() -> tuple[str, str]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    return private_pem, public_pem


def test_app_settings_reject_both_pat_and_app_or_upstream_repo_and_hide_key(tmp_path):
    private_pem, _ = _generate_rsa_keypair()
    key_file = tmp_path / "app.pem"
    key_file.write_text(private_pem, encoding="utf-8")

    # Cannot configure personal PAT and App together.
    both = GithubSettings(
        token="ghp_personal_pat",
        app_id="4242",
        app_private_key=private_pem,
        repo="Anchal-T/sqlparse",
    )
    with pytest.raises(SettingsError, match="not both"):
        both.validate_for_start()

    # Upstream repos are always rejected.
    upstream = GithubSettings(
        app_id="4242",
        app_private_key_path=str(key_file),
        repo="andialbrecht/sqlparse",
    )
    with pytest.raises(SettingsError, match="upstream"):
        upstream.validate_for_start()

    # Invalid PEM fails cleanly without echoing secret text.
    bad_pem = GithubSettings(
        app_id="4242",
        app_private_key="not-a-valid-pem-secret-material",
        repo="Anchal-T/sqlparse",
    )
    with pytest.raises(SettingsError) as exc_info:
        bad_pem.validate_for_start()
    assert "not-a-valid-pem-secret-material" not in str(exc_info.value)

    # Escaped \n in env PEM loads cleanly, defaults workflow_version to v2, and hides key in repr.
    escaped = private_pem.replace("\n", "\\n")
    env_settings = GithubSettings.from_env(
        {
            "GITHUB_APP_ID": "4242",
            "GITHUB_APP_PRIVATE_KEY": escaped,
            "GITHUB_REPO": "Anchal-T/sqlparse",
        }
    )
    env_settings.validate_for_start()
    assert env_settings.auth_mode == "app"
    assert env_settings.workflow_version == "v2"
    assert "BEGIN PRIVATE KEY" not in repr(env_settings)


@pytest.mark.anyio
async def test_app_token_source_mints_rs256_jwt_caches_repo_scoped_token_and_refreshes():
    private_pem, public_pem = _generate_rsa_keypair()
    clock = [NOW]

    async def noop_sleep(_seconds: float) -> None:
        return None

    fake = FakeGithub(
        repo="Anchal-T/sqlparse",
        self_login="octo-agent-team[bot]",
        repo_id=1450220110,
        app_public_key_pem=public_pem,
        app_id="4242",
        app_slug="octo-agent-team",
        installation_id=9001,
        clock=lambda: clock[0],
    )
    async with httpx.AsyncClient(transport=fake.transport) as http:
        source = GithubAppTokenSource(
            app_id="4242",
            private_key_pem=private_pem,
            repo="Anchal-T/sqlparse",
            clock=lambda: clock[0],
        )
        # Verify signed JWT structure directly.
        minted_jwt = source.mint_jwt(clock[0])
        claims = jwt.decode(
            minted_jwt,
            public_pem,
            algorithms=["RS256"],
            options={"verify_exp": False, "verify_iat": False},
        )
        assert claims["iss"] == "4242"
        assert claims["iat"] == int(NOW.timestamp()) - 60
        assert claims["exp"] == int(NOW.timestamp()) + 540

        client = GithubClient(
            repo="Anchal-T/sqlparse",
            http=http,
            clock=lambda: clock[0],
            sleep=noop_sleep,
            token_source=source,
        )
        bot_login = await client.get_self()
        assert bot_login == "octo-agent-team[bot]"

        repo_info_1 = await client.get_repo()
        repo_info_2 = await client.get_repo()
        assert repo_info_1.full_name == "Anchal-T/sqlparse"
        assert repo_info_2.owner_login == "Anchal-T"

        # Installation token was minted once, scoped to ["sqlparse"], and reused on the second call.
        token_mints = [
            call
            for call in fake.calls
            if call[0] == "POST" and call[1] == "/app/installations/9001/access_tokens"
        ]
        assert len(token_mints) == 1
        assert fake.minted_token_bodies == [{"repositories": ["sqlparse"]}]

        # Advance clock within the 5-minute refresh buffer -> refreshes installation token.
        clock[0] = NOW + timedelta(minutes=56)
        await client.get_repo()
        token_mints_after = [
            call
            for call in fake.calls
            if call[0] == "POST" and call[1] == "/app/installations/9001/access_tokens"
        ]
        assert len(token_mints_after) == 2


@pytest.mark.anyio
async def test_poller_prepare_in_app_mode_defaults_approver_to_fork_owner_and_blocks_self_approver(
    caplog,
    tmp_path,
):
    private_pem, public_pem = _generate_rsa_keypair()
    fake_github = FakeGithub(
        repo="Anchal-T/sqlparse",
        self_login="octo-agent-team[bot]",
        repo_id=1450220110,
        app_public_key_pem=public_pem,
        app_id="4242",
        app_slug="octo-agent-team",
        installation_id=9001,
        clock=lambda: NOW,
    )
    fake_workflow = FakeWorkflow()

    def dispatch(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.github.com":
            return fake_github.handle(request)
        return fake_workflow.handle(request)

    settings = GithubSettings(
        app_id="4242",
        app_private_key=private_pem,
        repo="Anchal-T/sqlparse",
        sqlite_path=str(tmp_path / "github.sqlite"),
    )
    runtime = build_runtime(
        settings,
        transport=httpx.MockTransport(dispatch),
        clock=lambda: NOW,
    )
    app = create_app(runtime)
    try:
        with caplog.at_level("INFO"):
            await runtime.poller.prepare()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as http:
                status = await http.get("/v1/status")
        assert runtime.poller.self_login == "octo-agent-team[bot]"
        assert runtime.poller.approvers == ("Anchal-T",)
        assert "BEGIN PRIVATE KEY" not in caplog.text
        assert "ghs_install_" not in caplog.text
        assert "ghs_install_" not in status.text
    finally:
        await close_runtime(runtime)

    # Configuring the App bot itself as an approver is rejected at prepare().
    bad_settings = GithubSettings(
        app_id="4242",
        app_private_key=private_pem,
        repo="Anchal-T/sqlparse",
        approvers=("octo-agent-team[bot]",),
        sqlite_path=str(tmp_path / "github-bad.sqlite"),
    )
    bad_store = Store(bad_settings.sqlite_path)
    async with (
        httpx.AsyncClient(transport=fake_github.transport) as gh_http,
        httpx.AsyncClient(transport=fake_workflow.transport) as wf_http,
    ):
        source = GithubAppTokenSource(
            app_id="4242",
            private_key_pem=private_pem,
            repo="Anchal-T/sqlparse",
            clock=lambda: NOW,
        )
        gh_client = GithubClient(
            repo="Anchal-T/sqlparse",
            http=gh_http,
            clock=lambda: NOW,
            sleep=lambda _s: _noop(),
            token_source=source,
        )
        wf_client = WorkflowClient(base_url="http://workflow:8002", http=wf_http)
        poller = Poller(
            settings=bad_settings,
            github=gh_client,
            workflow=wf_client,
            store=bad_store,
            clock=lambda: NOW,
            sleep=lambda _s: _noop(),
            holder="test",
        )
        with pytest.raises(RuntimeError, match="approver"):
            await poller.prepare()


async def _noop() -> None:
    return None
