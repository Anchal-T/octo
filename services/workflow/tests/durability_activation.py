"""Test-only activation control plane for the V1a durability proof.

This module is NOT part of any production service API. It is a minimal,
stdlib-only helper used by ``test_durability_integration.py``.

It keeps one long-lived HTTP endpoint listening (the *activation/control
plane*) and, on demand, spawns a separately versioned worker process that
serves the real Restate SDK probe. The worker is stopped after a bounded idle
grace once no request is in flight, so during a durable timer or an approval
wait there are genuinely zero worker processes. This is observable through
``evidence()`` (and the ``/__activation/evidence`` endpoint); the activator
itself intentionally stays up, so this does not claim the whole infrastructure
scales to zero.

Ownership: this file and ``test_durability_integration.py`` only.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Self

import httpx

DEFAULT_WORKER_MODULE = "services.workflow.tests.durability_probe"

# Headers that must never be copied verbatim through a proxy. ``content-length``
# and ``content-encoding`` are dropped because httpx re-frames the body.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
        "content-encoding",
        "date",
        "server",
    }
)


class ActivationError(RuntimeError):
    """The activator could not start a worker or reach it."""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _ForwardingHandler(BaseHTTPRequestHandler):
    """HTTP/1.1 proxy: control endpoint locally, everything else forwarded."""

    protocol_version = "HTTP/1.1"
    server_version = "AgentTeamActivator/1.0"

    @property
    def activator(self) -> ScaleToZeroActivator:  # type: ignore[name-defined]
        return self.server.activator  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()

    def do_PUT(self) -> None:
        self._dispatch()

    def do_PATCH(self) -> None:
        self._dispatch()

    def do_DELETE(self) -> None:
        self._dispatch()

    def do_HEAD(self) -> None:
        self._dispatch()

    def do_OPTIONS(self) -> None:
        self._dispatch()

    def log_message(self, fmt: str, *args: Any) -> None:
        # Restate polls /health frequently; keep it out of pytest output.
        return

    def _read_body(self) -> bytes:
        transfer = self.headers.get("Transfer-Encoding", "").lower()
        if "chunked" in transfer:
            chunks: list[bytes] = []
            while True:
                line = self.rfile.readline().strip()
                size = int(line.split(b";", 1)[0], 16)
                if size == 0:
                    while self.rfile.readline().strip():
                        pass
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.read(2)  # trailing CRLF
            return b"".join(chunks)
        length = int(self.headers.get("Content-Length", "0") or "0")
        return self.rfile.read(length) if length else b""

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _dispatch(self) -> None:
        if self.path.split("?", 1)[0] == "/__activation/evidence":
            self._send_json(200, self.activator.evidence())
            return

        body = self._read_body()
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _HOP_BY_HOP
        }
        try:
            status, response_headers, payload = self.activator.forward(
                self.command, self.path, body, headers
            )
        except ActivationError as error:
            self._send_json(503, {"error": "activation_failed", "detail": str(error)})
            return

        self.send_response(status)
        for key, value in response_headers.items():
            if key.lower() not in _HOP_BY_HOP:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)


class ScaleToZeroActivator:
    """On-demand activator for a single versioned worker process.

    The HTTP listener is long-lived. Only the child worker process is stopped
    when idle, which is what the durability proof observes.
    """

    def __init__(
        self,
        *,
        repo_root: str,
        state_dir: str,
        worker_module: str = DEFAULT_WORKER_MODULE,
        worker_port: int | None = None,
        idle_grace_s: float = 1.5,
        start_timeout_s: float = 15.0,
        request_timeout_s: float = 50.0,
        # Restate reaches the activator from inside Docker via
        # host.docker.internal -> host-gateway, so it cannot bind only to
        # 127.0.0.1. This is the one place a wider bind is required.
        bind_host: str = "0.0.0.0",
    ) -> None:
        self.repo_root = str(repo_root)
        self.state_dir = str(state_dir)
        self.worker_module = worker_module
        self.worker_port = worker_port if worker_port is not None else _free_port()
        self.idle_grace_s = idle_grace_s
        self.start_timeout_s = start_timeout_s
        self.bind_host = bind_host

        os.makedirs(self.state_dir, exist_ok=True)

        self._lock = threading.RLock()
        self._in_flight = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._start_count = 0
        self._last_start: float | None = None
        self._last_stop: float | None = None
        self._activity = threading.Event()
        self._shutdown = threading.Event()
        self._monitor: threading.Thread | None = None
        self._server: ThreadingHTTPServer | None = None

        self._client = httpx.Client(timeout=request_timeout_s, trust_env=False)
        self._ready_client = httpx.Client(timeout=0.5, trust_env=False)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> int:
        server = ThreadingHTTPServer((self.bind_host, 0), _ForwardingHandler)
        server.daemon_threads = True
        server.activator = self
        self._server = server
        self._monitor = threading.Thread(
            target=self._monitor_loop, name="durability-activation-monitor", daemon=True
        )
        self._monitor.start()
        threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.1},
            name="durability-activation-http",
            daemon=True,
        ).start()
        return int(server.server_address[1])

    def stop(self) -> None:
        self._shutdown.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        with self._lock:
            self._stop_worker_locked()
        if self._monitor is not None:
            self._monitor.join(timeout=2)
            self._monitor = None
        self._ready_client.close()
        self._client.close()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("activator is not started")
        return int(self._server.server_address[1])

    # -- observable evidence ----------------------------------------------

    def evidence(self) -> dict[str, Any]:
        with self._lock:
            proc = self._proc
            alive = proc is not None and proc.poll() is None
            return {
                "worker_alive": alive,
                "worker_pid": proc.pid if alive else None,
                "worker_start_count": self._start_count,
                "in_flight": self._in_flight,
                "last_start_unix": self._last_start,
                "last_stop_unix": self._last_stop,
                "worker_port": self.worker_port,
                "worker_module": self.worker_module,
                "idle_grace_s": self.idle_grace_s,
            }

    def wait_for_worker_stopped(self, timeout: float = 10.0) -> bool:
        return self._wait(lambda ev: not ev["worker_alive"], timeout)

    def wait_for_worker_starts(self, minimum: int, timeout: float = 10.0) -> bool:
        return self._wait(lambda ev: ev["worker_start_count"] >= minimum, timeout)

    def _wait(self, predicate: Any, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate(self.evidence()):
                return True
            time.sleep(0.1)
        return predicate(self.evidence())

    # -- forwarding --------------------------------------------------------

    def forward(
        self, method: str, path: str, body: bytes, headers: dict[str, str]
    ) -> tuple[int, dict[str, str], bytes]:
        with self._lock:
            self._in_flight += 1
            self._activity.set()
        try:
            with self._lock:
                self._ensure_worker_locked()
            url = f"http://127.0.0.1:{self.worker_port}{path}"
            response = self._client.request(method, url, content=body, headers=headers)
            return response.status_code, dict(response.headers), response.content
        except httpx.HTTPError as error:
            raise ActivationError(f"worker request failed: {error}") from error
        finally:
            with self._lock:
                self._in_flight -= 1

    # -- worker process management (all calls hold self._lock) ------------

    def _monitor_loop(self) -> None:
        while not self._shutdown.is_set():
            if self._activity.wait(self.idle_grace_s):
                self._activity.clear()
                continue
            with self._lock:
                if self._in_flight == 0 and self._proc is not None:
                    self._stop_worker_locked()

    def _ensure_worker_locked(self) -> subprocess.Popen[bytes]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        if self._proc is not None:
            self._stop_worker_locked()

        env = {"PATH": os.defpath, "PYTHONPATH": self.repo_root}
        env["AGENTTEAM_DURABILITY_ACTIVATION_ID"] = str(self._start_count + 1)
        command = [
            sys.executable,
            "-m",
            self.worker_module,
            "--port",
            str(self.worker_port),
            "--state-dir",
            self.state_dir,
        ]
        proc = subprocess.Popen(
            command,
            cwd=self.repo_root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._proc = proc
        self._start_count += 1
        self._last_start = time.time()
        self._activity.set()
        self._wait_ready_locked(proc)
        return proc

    def _wait_ready_locked(self, proc: subprocess.Popen[bytes]) -> None:
        deadline = time.monotonic() + self.start_timeout_s
        url = f"http://127.0.0.1:{self.worker_port}/healthz"
        last_error = "no response"
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise ActivationError(
                    f"worker exited during startup with code {proc.returncode}"
                )
            try:
                response = self._ready_client.get(url)
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as error:
                last_error = str(error)
            time.sleep(0.1)
        self._stop_worker_locked()
        raise ActivationError(f"worker did not become ready: {last_error}")

    def _stop_worker_locked(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        self._last_stop = time.time()
        if proc.poll() is not None:
            return
        for sig, wait_s in ((signal.SIGTERM, 3.0), (signal.SIGKILL, 3.0)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=wait_s)
                return
            except subprocess.TimeoutExpired:
                continue
