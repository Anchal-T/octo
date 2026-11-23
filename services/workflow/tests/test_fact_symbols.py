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


def test_issue_references_are_explicit_scoped_and_deduplicated():
    from services.workflow.app.fact_symbols import issue_references

    assert issue_references(
        "Anchal-T/sqlparse",
        "Related: #7, ANCHAL-T/sqlparse#7 and other/repo#8. "
        "https://github.com/Anchal-T/sqlparse/issues/9#issuecomment-42 "
        "Do not follow https://example.com/other/repo/issues/10 or commit deadbeef.",
        issue_id="gh-100-6",
    ) == [
        "anchal-t/sqlparse#6",
        "anchal-t/sqlparse#7",
        "other/repo#8",
        "anchal-t/sqlparse#9",
    ]
