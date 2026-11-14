"""Push route checks use synthetic repositories and SCIP indexes."""

import subprocess
from dataclasses import replace

import httpx
import pytest

from services.memory.app import main
from services.memory.tests.test_git_repo import _commit, _repo
from services.memory.tests.test_repo import FakeIndexer, FakeStore, _index


@pytest.mark.anyio
async def test_push_is_ordered_idempotent_and_closes_anchored_fact(
    tmp_path, monkeypatch
):
    path = tmp_path / "source"
    _repo(path)
    (path / "module.py").write_text("def f():\n    return 1\ndef g():\n    return 1\n")
    root = _commit(path, "root")
    (path / "module.py").write_text("def f():\n    return 2\ndef g():\n    return 1\n")
    tip = _commit(path, "change method")
    store = FakeStore(
        [
            {
                "_id": "fact-example",
                "valid_from_commit": root,
                "anchors": [{"symbol": "module.f().", "sha": root}],
                "evidence_test": None,
            }
        ]
    )
    store.settings = replace(main.store.settings, scip_repo_path=str(path))
    store.scip_indexer = FakeIndexer(
        {sha: _index(sha, "module.f().", "module.g().") for sha in (root, tip)}
    )
    monkeypatch.setattr(main, "store", store)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        first = await client.post(
            "/v1/repo/push", json={"branch": "main", "before": None, "after": tip}
        )
        second = await client.post(
            "/v1/repo/push", json={"branch": "main", "before": None, "after": tip}
        )
    assert first.status_code == second.status_code == 200
    assert first.json() == {
        "processed": [root, tip],
        "already_recorded": [],
        "closed_fact_ids": ["fact-example"],
        "main_head": tip,
    }
    assert second.json() == {
        "processed": [],
        "already_recorded": [root, tip],
        "closed_fact_ids": [],
        "main_head": tip,
    }
    assert list(store.commits) == [root, tip]
    assert len(store.closures) == 1
    assert store.scip_indexer.calls == [root, tip]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case",
    ["known", "fetched", "missing", "fetch_failed", "non_fast_forward", "unconfigured"],
)
async def test_push_fetch_and_error_contracts(monkeypatch, case):
    before, after = "a" * 40, "b" * 40  # synthetic SHAs
    calls = []
    known = (
        {before, after}
        if case != "fetched" and case != "missing" and case != "fetch_failed"
        else {before}
    )

    class FakeGit:
        def __init__(self, path):
            pass

        def has_commit(self, sha):
            return sha in known

        def fetch(self):
            calls.append("fetch")
            if case == "fetch_failed":
                raise subprocess.CalledProcessError(1, "git fetch")
            if case == "fetched":
                known.add(after)

        def is_ancestor(self, a, b):
            return case != "non_fast_forward"

        def rev_list(self, after, *, before):
            return [after]

    class FakeRepo:
        def __init__(self, *args):
            pass

        def on_push(self, shas, branch):
            calls.append((shas, branch))

    monkeypatch.setattr(main, "GitRepo", FakeGit)
    monkeypatch.setattr(main, "Repo", FakeRepo)
    monkeypatch.setattr(
        main.store, "scip_indexer", None if case == "unconfigured" else object()
    )
    monkeypatch.setattr(
        main.store, "settings", replace(main.store.settings, scip_repo_path="/clone")
    )
    monkeypatch.setattr(main.store, "get_commits", list)
    monkeypatch.setattr(main.store, "all_fact_closures", list)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/repo/push", json={"branch": "main", "before": before, "after": after}
        )
    if case in {"known", "fetched"}:
        assert response.status_code == 200
        assert response.json()["processed"] == [after]
        assert calls == (["fetch"] if case == "fetched" else []) + [([after], "main")]
    else:
        expected = {
            "missing": (409, {"detail": "unknown_sha", "sha": after}),
            "fetch_failed": (503, {"detail": "fetch_failed"}),
            "non_fast_forward": (
                409,
                {"detail": "non_fast_forward", "before": before, "after": after},
            ),
            "unconfigured": (503, {"detail": "scip_not_configured"}),
        }[case]
        assert (response.status_code, response.json()) == expected
        assert calls == (["fetch"] if case in {"missing", "fetch_failed"} else [])
