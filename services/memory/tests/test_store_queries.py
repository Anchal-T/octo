from datetime import UTC, datetime

import pytest

from services.memory.app.store import MemoryStore


class FakeCursor:
    def __init__(self, rows=()):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def execute(self, query, params):
        params = tuple(params)
        self.calls.append((query, params))
        return next(self.responses)


def _store_with_connection(monkeypatch, connection):
    store = MemoryStore()
    monkeypatch.setattr(store, "_connect", lambda: connection)
    return store


def test_facts_for_symbols_chunks_and_deduplicates_results(monkeypatch):
    symbols = [f"symbol-{index}" for index in range(60_001)]
    fact = {"_id": "fact-1", "valid_from_commit": "commit-1"}
    connection = FakeConnection([FakeCursor([fact]), FakeCursor([fact])])
    store = _store_with_connection(monkeypatch, connection)

    result = store.facts_for_symbols(symbols)

    assert [len(params) for _, params in connection.calls] == [60_000, 1]
    assert result == [fact]


def test_facts_by_ids_chunks_large_input(monkeypatch):
    fact_ids = [f"fact-{index}" for index in range(60_001)]
    connection = FakeConnection([FakeCursor(), FakeCursor()])
    store = _store_with_connection(monkeypatch, connection)

    assert store.facts_by_ids(fact_ids) == []

    assert [len(params) for _, params in connection.calls] == [60_000, 1]


def test_closures_for_facts_chunks_large_input(monkeypatch):
    fact_ids = [f"fact-{index}" for index in range(60_001)]
    connection = FakeConnection([FakeCursor(), FakeCursor()])
    store = _store_with_connection(monkeypatch, connection)

    assert store.closures_for_facts(fact_ids) == []

    assert [len(params) for _, params in connection.calls] == [60_000, 1]


def test_end_fact_valid_time_chunks_select_and_delete_parameters(monkeypatch):
    fact_ids = [f"fact-{index}" for index in range(60_001)]
    valid_from = datetime(2026, 1, 1, tzinfo=UTC)
    select_responses = [
        FakeCursor([{"_id": fact_id, "_valid_from": valid_from} for fact_id in chunk])
        for chunk in (fact_ids[:60_000], fact_ids[60_000:])
    ]
    connection = FakeConnection([*select_responses, FakeCursor(), FakeCursor()])
    store = _store_with_connection(monkeypatch, connection)

    store.end_fact_valid_time(fact_ids, valid_from)

    assert [len(params) for _, params in connection.calls] == [
        60_000,
        1,
        60_000,
        3,
    ]
    assert all(len(params) <= 60_000 for _, params in connection.calls)


def test_facts_for_symbols_rejects_nul_before_querying(monkeypatch):
    connection = FakeConnection([FakeCursor()])
    store = _store_with_connection(monkeypatch, connection)

    with pytest.raises(ValueError, match="NUL"):
        store.facts_for_symbols(["valid-symbol", "bad\x00symbol"])

    assert connection.calls == []
