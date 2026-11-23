"""Real, opt-in V1a durability proof against an isolated Restate 1.8.1.

Run with::

    AGENTTEAM_TEST_DURABILITY=1 \
      .venv/bin/python -m pytest services/workflow/tests/test_durability_integration.py -s

Nothing here touches production services, ports 8080/9070/8002, P1 scheduled
resume files, real providers, credentials, or the Obsidian vault. A unique
Restate container uses a dedicated Docker volume retained across restarts. Both
are removed in the ``finally`` block.

Probe contract (owned by ``durability_probe.py``):

* ``python -m services.workflow.tests.durability_probe --port P --state-dir D``
* FastAPI ``/healthz``; Restate SDK mounted at ``/restate/v1``.
* ``DurabilityProbeV1.run`` arg ``{revision, timer_seconds}``: two journaled
  fake-provider turns separated by a durable ``ctx.sleep``, then an approval
  promise. Each provider call writes one row to ``D/provider.sqlite`` keyed by a
  stable request id derived from revision and turn number.
* ``DurabilityProbeV1.info`` (shared) -> ``{stage, revision}``.
* ``DurabilityProbeV1.approve`` arg ``{revision, approved, by}`` ->
  ``{status: stale_revision | recorded | already_decided}``.

Ownership: this file and ``durability_activation.py`` only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from services.workflow.tests.durability_activation import ScaleToZeroActivator
from services.workflow.tests.durability_probe import evidence as probe_evidence

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENTTEAM_TEST_DURABILITY") != "1",
    reason="opt-in real integration: set AGENTTEAM_TEST_DURABILITY=1",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKER_MODULE = os.environ.get(
    "AGENTTEAM_DURABILITY_PROBE_MODULE",
    "services.workflow.tests.durability_probe",
)
RESTATE_IMAGE = "docker.restate.dev/restatedev/restate:1.8.1"
PRODUCTION_PORTS = {8080, 9070, 8002}

SERVICE = "DurabilityProbeV1"
REVISION = "v1a-proof-0001"
RUN_KEY = f"durability-{REVISION}"
TIMER_SECONDS = 20
# The probe's stdlib SQLite counter records *every* fake body execution in
# ``call_events`` (one row per executed turn), separately from saved
# ``completions``. Counting distinct ids alone would hide a repeated call.
TURN_ONE_KEY = f"{RUN_KEY}:{REVISION}:turn:1"
TURN_TWO_KEY = f"{RUN_KEY}:{REVISION}:turn:2"


def _docker(*args: str) -> str:
    try:
        result = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        detail = getattr(error, "stderr", "") or ""
        raise AssertionError(f"docker {' '.join(args)} failed: {detail}") from error
    return result.stdout.strip()


def _poll_until(
    description: str,
    read: Callable[[], Any],
    ready: Callable[[Any], bool],
    timeout: float = 30.0,
    interval: float = 0.25,
) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        try:
            last = read()
            if ready(last):
                return last
        except (httpx.HTTPError, OSError, ValueError):
            pass
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {description}; last={last!r}")


class IsolatedRestate:
    """One uniquely named Restate container with a persisted data mount."""

    def __init__(self, data_dir: Path) -> None:
        self.name = f"agentteam-durability-{uuid.uuid4().hex[:12]}"
        self.data_dir = data_dir
        self.volume = f"{self.name}-data"
        self.ingress_port = 0
        self.admin_port = 0

    @property
    def ingress(self) -> str:
        return f"http://127.0.0.1:{self.ingress_port}"

    @property
    def admin(self) -> str:
        return f"http://127.0.0.1:{self.admin_port}"

    def start(self) -> None:
        _docker(
            "run",
            "-d",
            "--name",
            self.name,
            "-p",
            "127.0.0.1::8080",
            "-p",
            "127.0.0.1::9070",
            "-v",
            f"{self.volume}:/restate-data",
            "--add-host",
            "host.docker.internal:host-gateway",
            "-e",
            "RESTATE_LOG_FILTER=restate=info",
            "-e",
            "RESTATE_BOOTSTRAP_NUM_PARTITIONS=1",
            "-e",
            "RESTATE_DEFAULT_NUM_PARTITIONS=1",
            "-e",
            "RESTATE_ROCKSDB_TOTAL_MEMORY_SIZE=32 MB",
            "-e",
            "RESTATE_SHUTDOWN_TIMEOUT=10s",
            RESTATE_IMAGE,
        )
        self._refresh_ports()
        self._wait_health()

    def restart(self) -> None:
        _docker("stop", "-t", "10", self.name)
        _docker("start", self.name)
        self._refresh_ports()
        self._wait_health()

    def remove(self) -> None:
        try:
            _docker("rm", "-f", self.name)
        except AssertionError:
            # Last-ditch cleanup must not mask the original test failure.
            subprocess.run(
                ["docker", "rm", "-f", self.name],
                capture_output=True,
                timeout=30,
                check=False,
            )
        subprocess.run(
            ["docker", "volume", "rm", self.volume],
            capture_output=True,
            timeout=30,
            check=False,
        )

    def _refresh_ports(self) -> None:
        def host_port(container_port: str) -> int:
            output = _docker("port", self.name, container_port)
            match = re.search(r":(\d+)\s*$", output.splitlines()[-1])
            if match is None:
                raise AssertionError(
                    f"cannot read host port for {container_port}: {output!r}"
                )
            port = int(match.group(1))
            assert port not in PRODUCTION_PORTS, (
                "isolated port collided with production"
            )
            return port

        self.ingress_port = host_port("8080/tcp")
        self.admin_port = host_port("9070/tcp")
        assert self.ingress_port != self.admin_port

    def _wait_health(self) -> None:
        with httpx.Client(timeout=2.0, trust_env=False) as client:

            def ready() -> bool:
                try:
                    ingress = client.get(f"{self.ingress}/restate/health")
                    admin = client.get(f"{self.admin}/health")
                    return ingress.status_code == 200 and admin.status_code == 200
                except httpx.HTTPError:
                    return False

            _poll_until("isolated Restate health", ready, bool, timeout=45.0)


class _Env:
    def __init__(
        self,
        *,
        activator: ScaleToZeroActivator,
        restate: IsolatedRestate,
        state_dir: Path,
    ) -> None:
        self.activator = activator
        self.restate = restate
        self.state_dir = state_dir
        self.client = httpx.Client(timeout=5.0, trust_env=False)

    def close(self) -> None:
        self.client.close()

    # -- control plane -----------------------------------------------------

    def register(self) -> None:
        uri = f"http://host.docker.internal:{self.activator.port}/restate/v1"
        response = self.client.post(
            f"{self.restate.admin}/deployments",
            json={"uri": uri, "use_http_11": True, "force": True},
        )
        assert response.is_success, f"deployment registration failed: {response.text}"

    def services(self) -> list[dict[str, Any]]:
        response = self.client.get(f"{self.restate.admin}/services")
        response.raise_for_status()
        return response.json()["services"]

    def ensure_registered(self) -> None:
        if any(service.get("name") == SERVICE for service in self.services()):
            return
        self.register()

    def wait_service(self) -> None:
        _poll_until(
            f"{SERVICE} discovery",
            self.services,
            lambda services: any(s.get("name") == SERVICE for s in services),
            timeout=20.0,
        )

    # -- workflow ingress --------------------------------------------------

    # Use the ingress routes supported by the pinned server and Python SDK.

    def start_run(self) -> None:
        # No Idempotency-Key: the workflow key itself is the idempotency key
        # (AGENTS.md decision), and a workflow can only be submitted once.
        response = self.client.post(
            f"{self.restate.ingress}/{SERVICE}/{RUN_KEY}/run/send",
            json={"revision": REVISION, "timer_seconds": TIMER_SECONDS},
        )
        assert response.is_success, f"run start failed: {response.text}"

    def info(self) -> dict[str, Any] | None:
        response = self.client.get(
            f"{self.restate.ingress}/{SERVICE}/{RUN_KEY}/info",
        )
        if response.status_code != 200:
            return None
        value = response.json()
        return value if isinstance(value, dict) else None

    def approve(
        self, *, revision: str, approved: bool, by: str
    ) -> tuple[int, dict[str, Any]]:
        response = self.client.post(
            f"{self.restate.ingress}/{SERVICE}/{RUN_KEY}/approve",
            json={"revision": revision, "approved": approved, "by": by},
        )
        body = response.json()
        return response.status_code, body if isinstance(body, dict) else {}

    def result(self) -> tuple[int, Any]:
        # Match the workflow output API used by the production HTTP adapter.
        response = self.client.post(
            f"{self.restate.ingress}/restate/output",
            json={
                "target": "workflow",
                "workflowName": SERVICE,
                "workflowKey": RUN_KEY,
            },
        )
        try:
            body = response.json()
        except ValueError:
            body = response.text
        return response.status_code, body

    def wait_result(self, timeout: float = 30.0) -> dict[str, Any]:
        def ready(value: tuple[int, Any]) -> bool:
            status, body = value
            return status == 200 and isinstance(body, dict) and "revision" in body

        _status, body = _poll_until("workflow output", self.result, ready, timeout)
        assert isinstance(body, dict)
        return body

    # -- fake-provider evidence -------------------------------------------

    def evidence(self) -> dict[str, Any]:
        """Read the probe's stdlib SQLite counter directly (no worker started)."""
        return probe_evidence(self.state_dir)

    def provider_count(self) -> int:
        # Actual fake body executions, not distinct keys or saved completions.
        return int(self.evidence()["calls"])

    def call_counts(self) -> dict[str, int]:
        return dict(self.evidence()["call_counts"])

    def snapshot(self) -> dict[str, Any]:
        try:
            info = self.info()
        except (httpx.HTTPError, OSError):
            info = None
        return {
            "info": info,
            "provider": self.evidence(),
            "activation": self.activator.evidence(),
            "container": self.restate.name,
            "restate_image": RESTATE_IMAGE,
            "ingress_port": self.restate.ingress_port,
            "admin_port": self.restate.admin_port,
        }


@contextmanager
def _environment(tmp_path: Path):
    state_dir = tmp_path / "probe-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    activator = ScaleToZeroActivator(
        repo_root=str(REPO_ROOT),
        state_dir=str(state_dir),
        worker_module=WORKER_MODULE,
    )
    restate = IsolatedRestate(tmp_path / "restate-data")
    env: _Env | None = None
    try:
        activator.start()
        restate.start()
        env = _Env(activator=activator, restate=restate, state_dir=state_dir)
        yield env
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            try:
                restate.remove()
            finally:
                activator.stop()


def _write_receipt(receipt: dict[str, Any]) -> None:
    raw = os.environ.get("AGENTTEAM_DURABILITY_RECEIPT")
    if not raw:
        return
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def test_v1a_durability_end_to_end(tmp_path: Path) -> None:
    receipt: dict[str, Any] = {
        "scenario": "v1a-durability",
        "revision": REVISION,
        "timer_seconds": TIMER_SECONDS,
        "worker_module": WORKER_MODULE,
        "image": RESTATE_IMAGE,
        "passed": False,
        "observed": {},
    }
    env: _Env | None = None
    try:
        with _environment(tmp_path) as env:
            env.register()
            env.wait_service()
            assert env.activator.wait_for_worker_starts(1), (
                "discovery never started a worker"
            )
            assert env.activator.wait_for_worker_stopped(), (
                "worker stayed alive after discovery"
            )

            # Turn 1 is journaled, then the workflow suspends on its durable timer.
            env.start_run()
            sleeping = _poll_until(
                "turn-1 completion (sleeping stage)",
                env.info,
                lambda info: info is not None and info.get("stage") == "timer_wait",
                timeout=15.0,
            )
            assert sleeping["revision"] == REVISION
            assert env.provider_count() == 1, "expected exactly one fake body execution"
            assert env.activator.wait_for_worker_stopped(), (
                "worker alive during timer wait"
            )
            assert not env.activator.evidence()["worker_alive"]

            # The timer remains in Restate's journal while all workers are gone.
            timer_snapshot = env.evidence()
            timer_activation = env.activator.evidence()
            env.restate.restart()
            assert any(s["name"] == SERVICE for s in env.services())

            # The durable timer wakes the workflow on a brand-new worker.
            _poll_until(
                "turn-2 fake body execution",
                env.provider_count,
                lambda count: count >= 2,
                timeout=TIMER_SECONDS + 20.0,
            )
            waiting = _poll_until(
                "waiting_approval stage",
                env.info,
                lambda info: (
                    info is not None and info.get("stage") == "waiting_approval"
                ),
                timeout=15.0,
            )
            assert waiting["revision"] == REVISION
            assert env.provider_count() == 2, (
                "timer/approval replay repeated a provider call"
            )
            assert env.activator.wait_for_worker_stopped(), (
                "worker alive during approval wait"
            )
            assert not env.activator.evidence()["worker_alive"]
            approval_activation = env.activator.evidence()
            starts_before_restart = env.activator.evidence()["worker_start_count"]

            # Restart only the isolated Restate; its data mount is retained.
            env.restate.restart()
            env.ensure_registered()
            info = _poll_until(
                "workflow still waiting after restart",
                env.info,
                lambda value: (
                    value is not None and value.get("stage") == "waiting_approval"
                ),
                timeout=20.0,
            )
            assert info["revision"] == REVISION
            assert env.provider_count() == 2, "restart replay repeated a provider call"

            # A stale revision must not resolve the approval or replay calls.
            stale_status, stale_body = env.approve(
                revision="stale-revision", approved=True, by="test"
            )
            assert stale_body.get("status") == "stale_revision", (
                stale_status,
                stale_body,
            )
            assert env.provider_count() == 2
            info = env.info()
            assert info is not None and info.get("stage") == "waiting_approval"

            # A valid approval completes the workflow.
            assert env.activator.wait_for_worker_stopped()
            before_approval = env.activator.evidence()
            assert not before_approval["worker_alive"]
            valid_status, valid_body = env.approve(
                revision=REVISION, approved=True, by="test-reviewer"
            )
            assert valid_body.get("status") == "recorded", (valid_status, valid_body)
            result = env.wait_result()
            assert result["revision"] == REVISION
            assert result["decision"]["approved"] is True, result
            assert result["decision"]["by"] == "test-reviewer"
            assert (
                env.activator.evidence()["worker_start_count"]
                > before_approval["worker_start_count"]
            ), "approval did not cold-start a worker"

            # A second decision is harmless and replays no provider calls.
            dup_status, dup_body = env.approve(
                revision=REVISION, approved=False, by="duplicate-reviewer"
            )
            assert dup_body.get("status") == "already_decided", (dup_status, dup_body)
            assert env.provider_count() == 2

            calls = env.evidence()
            assert calls["calls"] == 2, calls
            elapsed = calls["events"][1]["at_epoch"] - calls["events"][0]["at_epoch"]
            assert TIMER_SECONDS - 0.1 <= elapsed < TIMER_SECONDS + 10, elapsed
            assert calls["call_counts"] == {TURN_ONE_KEY: 1, TURN_TWO_KEY: 1}, calls
            assert all(
                receipt_row["revision"] == REVISION for receipt_row in calls["receipts"]
            ), calls["receipts"]
            activation = env.activator.evidence()
            assert activation["worker_start_count"] > starts_before_restart, (
                "replays after the restart did not create a fresh worker"
            )

            receipt["observed"] = {
                "timer_wait_provider": timer_snapshot,
                "timer_wait_activation": timer_activation,
                "approval_wait_activation": approval_activation,
                "before_valid_approval_activation": before_approval,
                "timer_elapsed_seconds": elapsed,
                "stale_approval": {"status_code": stale_status, "body": stale_body},
                "valid_approval": {"status_code": valid_status, "body": valid_body},
                "duplicate_approval": {"status_code": dup_status, "body": dup_body},
                "result": result,
                "final_snapshot": env.snapshot(),
                "worker_starts_before_restart": starts_before_restart,
            }
        receipt["passed"] = True
    finally:
        if env is not None:
            receipt["cleanup_activation"] = env.activator.evidence()
        _write_receipt(receipt)
