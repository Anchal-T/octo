from services.memory.app.git_repo import Hunk
from services.memory.app.scip import ScipDefinition, ScipIndex
from services.memory.app.symbol_differ import SymbolDiffer


def _index(definitions):
    return ScipIndex({symbol: tuple(items) for symbol, items in definitions.items()})


def _definition(symbol, start, end):
    return ScipDefinition(
        symbol=symbol,
        relative_path="module.py",
        range=(start, 0, start, 5),
        enclosing_range=(start, 0, end, 0),
    )


def test_diff_includes_every_definition_overlapping_each_hunk():
    class_symbol = "module.Class#"
    method_symbol = "module.Class#method()."
    parent = _index(
        {
            class_symbol: [_definition(class_symbol, 0, 10)],
            method_symbol: [_definition(method_symbol, 3, 5)],
        }
    )
    child = _index(
        {
            class_symbol: [_definition(class_symbol, 0, 10)],
            method_symbol: [_definition(method_symbol, 3, 5)],
        }
    )

    method_hunk = Hunk("module.py", old_start=4, old_len=1, new_start=4, new_len=1)
    class_body_hunk = Hunk("module.py", old_start=8, old_len=1, new_start=8, new_len=1)

    assert SymbolDiffer.diff(parent, child, [method_hunk]) == {
        class_symbol,
        method_symbol,
    }
    assert SymbolDiffer.diff(parent, child, [class_body_hunk]) == {class_symbol}


def test_hunk_crossing_symbol_boundaries_closes_both_symbols():
    first = "module.first()."
    second = "module.second()."
    parent = _index(
        {
            first: [_definition(first, 0, 3)],
            second: [_definition(second, 4, 7)],
        }
    )
    child = _index(
        {
            first: [_definition(first, 0, 3)],
            second: [_definition(second, 4, 7)],
        }
    )
    hunk = Hunk("module.py", old_start=3, old_len=3, new_start=3, new_len=3)

    assert SymbolDiffer.diff(parent, child, [hunk]) == {first, second}


def test_diff_includes_symbols_added_and_removed_without_hunks():
    removed = "module.old()."
    added = "module.new()."
    parent = _index({removed: [_definition(removed, 0, 1)]})
    child = _index({added: [_definition(added, 0, 1)]})

    assert SymbolDiffer.diff(parent, child, []) == {removed, added}


def test_added_or_deleted_line_uses_the_index_that_contains_it():
    symbol = "module.function()."
    parent = _index({})
    child = _index({symbol: [_definition(symbol, 0, 3)]})
    hunk = Hunk("module.py", old_start=0, old_len=0, new_start=1, new_len=1)

    assert SymbolDiffer.diff(parent, child, [hunk]) == {symbol}
