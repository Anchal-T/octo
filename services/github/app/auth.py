from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from .schemas import Clock
from .settings import (
    GITHUB_APP_JWT_BACKDATE_S,
    GITHUB_APP_JWT_TTL_S,
    GITHUB_APP_TOKEN_REFRESH_BUFFER_S,
    SettingsError,
)

AppRequester = Callable[
    [str, str, str, dict[str, Any] | None],
    Awaitable[dict[str, Any]],
]


def load_rsa_private_key(pem_text: str) -> rsa.RSAPrivateKey:
    normalized = pem_text.replace("\\n", "\n").strip()
    if not normalized:
        raise SettingsError("GitHub App private key must not be empty")
    try:
        loaded = serialization.load_pem_private_key(
            normalized.encode("utf-8"),
            password=None,
        )
    except (TypeError, ValueError) as error:
        raise SettingsError(
            "GITHUB_APP_PRIVATE_KEY is not a valid unencrypted PEM key"
        ) from error
    if not isinstance(loaded, rsa.RSAPrivateKey):
        raise SettingsError("GITHUB_APP_PRIVATE_KEY must be an RSA private key")
    return loaded


class TokenSource(Protocol):
    auth_mode: Literal["app", "pat"]

    async def bearer_token(self, *, force_refresh: bool = False) -> str: ...
    async def resolve_self_login(self) -> str: ...


class StaticTokenSource:
    auth_mode: Literal["app", "pat"] = "pat"

    def __init__(self, token: str) -> None:
        self._token = token

    def __repr__(self) -> str:
        return "StaticTokenSource(<redacted>)"

    async def bearer_token(self, *, force_refresh: bool = False) -> str:
        return self._token

    async def resolve_self_login(self) -> str:
        raise RuntimeError("StaticTokenSource resolves login via GET /user")


class GithubAppTokenSource:
    auth_mode: Literal["app", "pat"] = "app"

    def __init__(
        self,
        *,
        app_id: str,
        private_key_pem: str,
        repo: str,
        clock: Clock,
        installation_id: int | None = None,
    ) -> None:
        if not app_id.strip():
            raise SettingsError("GITHUB_APP_ID must be set")
        self.app_id = app_id.strip()
        self._private_key = load_rsa_private_key(private_key_pem)
        self.repo = repo.strip()
        self.clock = clock
        self.installation_id = installation_id
        self._requester: AppRequester | None = None
        self._lock = asyncio.Lock()
        self._cached_token = ""
        self._expires_at: datetime | None = None

    def __repr__(self) -> str:
        return (
            f"GithubAppTokenSource(app_id={self.app_id!r}, repo={self.repo!r}, "
            f"installation_id={self.installation_id!r})"
        )

    def bind_requester(self, requester: AppRequester) -> None:
        self._requester = requester

    def mint_jwt(self, now: datetime) -> str:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        ts = int(now.astimezone(UTC).timestamp())
        payload = {
            "iat": ts - GITHUB_APP_JWT_BACKDATE_S,
            "exp": ts + GITHUB_APP_JWT_TTL_S,
            "iss": self.app_id,
        }
        return jwt.encode(payload, self._private_key, algorithm="RS256")

    async def resolve_self_login(self) -> str:
        requester = self._require_requester()
        app_jwt = self.mint_jwt(self._now())
        app_info = await requester("GET", "/app", app_jwt, None)
        slug = str(app_info.get("slug") or "").strip()
        if not slug:
            raise RuntimeError("GitHub /app response did not include an app slug")
        return f"{slug}[bot]"

    async def bearer_token(self, *, force_refresh: bool = False) -> str:
        async with self._lock:
            now = self._now()
            if (
                not force_refresh
                and self._cached_token
                and self._expires_at is not None
                and self._expires_at - now
                > timedelta(seconds=GITHUB_APP_TOKEN_REFRESH_BUFFER_S)
            ):
                return self._cached_token

            requester = self._require_requester()
            app_jwt = self.mint_jwt(now)
            if self.installation_id is None:
                install = await requester(
                    "GET", f"/repos/{self.repo}/installation", app_jwt, None
                )
                raw_id = install.get("id")
                if not isinstance(raw_id, int) or raw_id < 1:
                    raise RuntimeError(
                        "GitHub installation response did not include a valid id"
                    )
                self.installation_id = raw_id

            repo_short_name = self.repo.split("/", 1)[1]
            minted = await requester(
                "POST",
                f"/app/installations/{self.installation_id}/access_tokens",
                app_jwt,
                {"repositories": [repo_short_name]},
            )
            token = str(minted.get("token") or "").strip()
            expires_raw = minted.get("expires_at")
            if not token or not isinstance(expires_raw, str):
                raise RuntimeError("GitHub installation token response was incomplete")
            expires_at = datetime.fromisoformat(expires_raw)
            if expires_at.tzinfo is None or expires_at.utcoffset() is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            self._cached_token = token
            self._expires_at = expires_at.astimezone(UTC)
            return self._cached_token

    def _require_requester(self) -> AppRequester:
        if self._requester is None:
            raise RuntimeError("GithubAppTokenSource is not bound to a GithubClient")
        return self._requester

    def _now(self) -> datetime:
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now.astimezone(UTC)
