import json

from services.memory.app.scip import ScipDefinition, ScipIndex
from services.memory.app.sync import resolve_anchors


def _index_file(tmp_path, symbol):
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "relative_path": "src/module.py",
                        "occurrences": [
                            {"range": [1, 4, 11], "symbol": symbol, "symbol_roles": 1}
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return index_path


def test_loads_definition_occurrences_from_scip_json(tmp_path):
    symbol = "scip-python python demo 1.0 `demo.module`/example()."
    index = ScipIndex.load(_index_file(tmp_path, symbol))

    assert index.definitions_for(symbol) == (
        ScipDefinition(
            symbol=symbol,
            relative_path="src/module.py",
            range=(1, 4, 11),
            enclosing_range=None,
        ),
    )
    assert index.definitions_for("missing") == ()


def test_anchor_names_resolve_only_against_the_matching_base_commit(tmp_path):
    symbol = "scip-python python demo 1.0 `demo.module`/example()."
    index = ScipIndex.load(_index_file(tmp_path, symbol), sha="base-1")

    assert index.resolve(symbol) == symbol
    assert index.has_definition(symbol)
    assert not index.has_definition("missing")
    assert resolve_anchors([symbol], index, "base-1") == [
        {"symbol": symbol, "sha": "base-1"}
    ]
    assert resolve_anchors([symbol], index, "other-base") is None
    assert resolve_anchors(["missing"], index, "base-1") is None
    assert resolve_anchors([], index, "base-1") is None
