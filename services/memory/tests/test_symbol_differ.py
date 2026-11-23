from services.memory.app.git_repo import Hunk
from services.memory.app.scip import ScipDefinition, ScipIndex
from services.memory.app.symbol_differ import SymbolDiffer


def _index(definitions):
    return ScipIndex({symbol: tuple(items) for symbol, items in definitions.items()})


def _definition(symbol, start, end, relative_path="module.py"):
    return ScipDefinition(
        symbol=symbol,
        relative_path=relative_path,
        range=(start, 0, start, 5),
        enclosing_range=(start, 0, end, 0),
    )


def test_diff_includes_innermost_definition_per_hunk():
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

    assert SymbolDiffer.diff(parent, child, [method_hunk]) == {method_symbol}
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


def test_3_1_request_encoding_mixin_case():
    path = "requests/models.py"
    class_sym = "requests.models/RequestEncodingMixin#"
    method_encode_files = "requests.models/RequestEncodingMixin#_encode_files()."
    method_encode_params = "requests.models/RequestEncodingMixin#_encode_params()."

    definitions = {
        class_sym: [_definition(class_sym, 10, 100, relative_path=path)],
        method_encode_files: [
            _definition(method_encode_files, 20, 50, relative_path=path)
        ],
        method_encode_params: [
            _definition(method_encode_params, 60, 90, relative_path=path)
        ],
    }
    parent = _index(definitions)
    child = _index(definitions)

    # 3.1: A one-method change inside _encode_files yields _encode_files and NOT RequestEncodingMixin#
    one_method_hunk = Hunk(path, old_start=25, old_len=4, new_start=25, new_len=4)
    result = SymbolDiffer.diff(parent, child, [one_method_hunk])
    assert result == {method_encode_files}
    assert class_sym not in result


def test_3_2_edit_in_class_body_outside_methods_yields_class_symbol():
    path = "requests/models.py"
    class_sym = "requests.models/RequestEncodingMixin#"
    method_sym = "requests.models/RequestEncodingMixin#_encode_files()."

    definitions = {
        class_sym: [_definition(class_sym, 10, 100, relative_path=path)],
        method_sym: [_definition(method_sym, 20, 50, relative_path=path)],
    }
    parent = _index(definitions)
    child = _index(definitions)

    # Edit at lines 12..14 (class body outside method)
    class_body_hunk = Hunk(path, old_start=12, old_len=3, new_start=12, new_len=3)
    result = SymbolDiffer.diff(parent, child, [class_body_hunk])
    assert result == {class_sym}


def test_3_3_hunk_spanning_two_methods_yields_both_methods():
    path = "requests/models.py"
    class_sym = "requests.models/RequestEncodingMixin#"
    m1 = "requests.models/RequestEncodingMixin#_encode_files()."
    m2 = "requests.models/RequestEncodingMixin#_encode_params()."

    definitions = {
        class_sym: [_definition(class_sym, 10, 100, relative_path=path)],
        m1: [_definition(m1, 20, 30, relative_path=path)],
        m2: [_definition(m2, 30, 40, relative_path=path)],
    }
    parent = _index(definitions)
    child = _index(definitions)

    # Hunk spans lines 25..35 (covering lines in m1 and m2)
    spanning_hunk = Hunk(path, old_start=25, old_len=10, new_start=25, new_len=10)
    result = SymbolDiffer.diff(parent, child, [spanning_hunk])
    assert m1 in result
    assert m2 in result
