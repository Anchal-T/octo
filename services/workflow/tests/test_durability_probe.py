"""Fast unit checks for the test-only durability probe (no Restate server)."""

import asyncio

import httpx

from services.workflow.tests.durability_probe import (
    build_app,
    evidence,
    fake_model_turn,
)


def test_repeated_body_is_counted_even_when_completion_is_saved(tmp_path):
    state_dir = str(tmp_path)
    request_id = "probe-key:turn:1"

    first = fake_model_turn(
        state_dir=state_dir, request_id=request_id, turn=1, revision="rev-a"
    )
    assert first["request_id"] == request_id

    snap = evidence(state_dir)
    assert snap["calls"] == 1
    assert snap["completions"] == 1
    assert snap["keys"] == [request_id]

    # A second execution of the *same* body: the saved completion is unchanged,
    # but the execution count must still expose the duplicate.
    fake_model_turn(
        state_dir=state_dir, request_id=request_id, turn=1, revision="rev-a"
    )
    snap = evidence(state_dir)
    assert snap["calls"] == 2
    assert snap["completions"] == 1
    assert snap["call_counts"][request_id] == 2
    assert snap["keys"] == [request_id]
    assert snap["receipts"][0]["output"]["text"] == "fake-model-turn-1"


def test_health_and_evidence_routes(tmp_path):
    app = build_app(tmp_path)

    async def check() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://probe"
        ) as client:
            assert (await client.get("/healthz")).json()["status"] == "ok"
            body = (await client.get("/evidence")).json()
            assert body["calls"] == 0
            assert body["keys"] == []

    asyncio.run(check())


def test_failed_worker_start_releases_activation_request(tmp_path):
    from pathlib import Path

    import pytest

    from services.workflow.tests.durability_activation import (
        ActivationError,
        ScaleToZeroActivator,
    )

    with ScaleToZeroActivator(
        repo_root=str(Path(__file__).resolve().parents[3]),
        state_dir=str(tmp_path),
        worker_module="octo_missing_test_worker",
        start_timeout_s=2,
    ) as activator:
        with pytest.raises(ActivationError):
            activator.forward("GET", "/healthz", b"", {})
        assert activator.evidence()["in_flight"] == 0
