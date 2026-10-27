import json

from services.memory.app.scip import ScipDefinition, ScipIndex


def test_loads_definition_occurrences_from_scip_json(tmp_path):
    symbol = "scip-python python demo 1.0 `demo.module`/example()."
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "relative_path": "src/module.py",
                        "occurrences": [
                            {"range": [8, 0, 7], "symbol": symbol, "symbol_roles": 8},
                            {
                                "range": [1, 4, 11],
                                "symbol": symbol,
                                "symbol_roles": 1,
                                "enclosing_range": [1, 0, 2, 16],
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    index = ScipIndex.load(index_path)

    assert index.definitions_for(symbol) == (
        ScipDefinition(
            symbol=symbol,
            relative_path="src/module.py",
            range=(1, 4, 11),
            enclosing_range=(1, 0, 2, 16),
        ),
    )
    assert index.definitions_for("missing") == ()
