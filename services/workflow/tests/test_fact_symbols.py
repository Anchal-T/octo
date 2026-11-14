from services.workflow.app.fact_symbols import symbol_candidates


def test_candidates_cover_dotted_names_stack_traces_and_old_diff_locations():
    result = symbol_candidates(
        "a" * 40,
        "sqlparse.lexer.Lexer.get_tokens sqlparse.lexer.Lexer.get_tokens\n"
        'File "/venv/lib/sqlparse/lexer.py", line 42\n'
        "sqlparse/sql.py:10\n../escape.py:2\n",
        "--- a/tests/test_lexer.py\n+++ b/tests/test_lexer.py\n@@ -4,2 +4,3 @@\n"
        "+ sqlparse.lexer.Lexer\n",
    )
    assert result.names == ["sqlparse.lexer.Lexer.get_tokens", "sqlparse.lexer.Lexer"]
    assert [location.model_dump() for location in result.locations] == [
        {"file": "sqlparse/lexer.py", "line": 42},
        {"file": "sqlparse/sql.py", "line": 10},
        {"file": "tests/test_lexer.py", "line": 4},
        {"file": "tests/test_lexer.py", "line": 5},
    ]
