import pytest

from services.workflow.app.triage import (
    TRIAGER_SYSTEM_PROMPT,
    parse_triage,
    triage_messages,
)

VALID = '{"kind": "bug", "priority": "medium", "summary": "Example summary."}'
AREAS = ("fields", "schema", "unknown")


@pytest.mark.parametrize(
    "text",
    [
        VALID,
        f"```json\n{VALID}\n```",
        f"Here is the triage:\n{VALID}\nThanks.",
        f"{VALID}\nNote: braces {{like this}} after the object.",
        '{"kind": "BUG", "priority": " Medium ", "summary": "Example summary."}',
        (
            '{"kind": "bug", "priority": "medium", "summary": "Example summary.",'
            ' "labels": ["agent-team"]}'
        ),
    ],
)
def test_parse_triage_accepts_usable_replies(text):
    result = parse_triage(text)

    assert result is not None
    assert (result.kind, result.priority, result.summary) == (
        "bug",
        "medium",
        "Example summary.",
    )
    assert result.area == "unknown"


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "   ",
        "not json at all",
        "[1, 2, 3]",
        '{"kind": "crash", "priority": "medium", "summary": "Example"}',
        '{"kind": "bug", "priority": "urgent", "summary": "Example"}',
        '{"kind": "bug", "summary": "Example"}',
        '{"kind": "bug", "priority": "medium", "summary": "   "}',
        '{"kind": "bug", "priority": "medium", "summary": 3}',
        '{"kind": "bug", "priority": "medium", "summary": "Example"',
    ],
)
def test_parse_triage_rejects_unusable_replies(text):
    assert parse_triage(text) is None


def test_parse_triage_collapses_whitespace_and_cuts_summary():
    long_summary = "word\n\n" * 200
    text = (
        f'{{"kind": "docs", "priority": "low", "summary": "{long_summary}"}}'.replace(
            "\n", "\\n"
        )
    )

    result = parse_triage(text)

    assert result is not None
    assert len(result.summary) <= 300
    assert "\n" not in result.summary


def test_triage_messages_state_facts_and_ask_for_json_only():
    messages = triage_messages("Example title", "Example body")

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == TRIAGER_SYSTEM_PROMPT.replace(
        "{area_values}", "unknown"
    )
    assert "already added the agent-team label" in TRIAGER_SYSTEM_PROMPT
    assert "cannot take any action" in TRIAGER_SYSTEM_PROMPT
    assert "JSON" in TRIAGER_SYSTEM_PROMPT
    assert messages[1]["content"] == "GitHub issue title: Example title\n\nExample body"
    assert len(TRIAGER_SYSTEM_PROMPT) / 4 < 150


def test_parse_triage_keeps_braces_inside_the_summary():
    result = parse_triage(
        '{"kind": "bug", "priority": "low", "summary": "Fails on {a} and {b}."}'
    )

    assert result is not None
    assert result.summary == "Fails on {a} and {b}."


def test_parse_triage_normalises_area_before_checking_it():
    result = parse_triage(
        '{"kind":"bug","priority":"low","summary":"Example","area":"Parser-Tree "}',
        allowed_areas=("parser-tree", "unknown"),
    )

    assert result is not None
    assert result.area == "parser-tree"


@pytest.mark.parametrize(
    ("area_field", "expected_area"),
    [
        ('"area": "fields",', "fields"),
        ('"area": "not-configured",', "unknown"),
        ("", "unknown"),
    ],
)
def test_parse_triage_maps_missing_or_unlisted_area_to_unknown(
    area_field, expected_area
):
    text = f'{{"kind":"bug","priority":"low",{area_field}"summary":"Example"}}'

    result = parse_triage(text, allowed_areas=AREAS)

    assert result is not None
    assert result.area == expected_area


def test_triage_prompt_lists_areas_from_repo_config():
    system = triage_messages("Example title", "Example body", allowed_areas=AREAS)[0][
        "content"
    ]

    assert '"area": "fields|schema|unknown"' in system


@pytest.mark.parametrize(
    ("raw_summary", "expected_summary"),
    [
        (
            r"Regex anchor \Z rejects trailing newlines.",
            r"Regex anchor \Z rejects trailing newlines.",
        ),
        (
            r"Pattern \d+ fails on unicode digits.",
            r"Pattern \d+ fails on unicode digits.",
        ),
    ],
)
def test_parse_triage_keeps_invalid_escapes_as_literal_backslashes(
    raw_summary, expected_summary
):
    text = (
        '{"kind": "bug", "priority": "medium", '
        f'"summary": "{raw_summary}", "area": "validate"}}'
    )

    result = parse_triage(text, allowed_areas=("fields", "validate", "unknown"))

    assert result is not None
    assert result.area == "validate"
    assert result.summary == expected_summary


def test_parse_triage_repair_leaves_valid_escapes_alone():
    text = (
        r'{"kind": "bug", "priority": "low", '
        r'"summary": "Say \"hi\" \u0041\\B\/C\tD\nE \Z", "area": "fields"}'
    )

    result = parse_triage(text, allowed_areas=AREAS)

    assert result is not None
    assert result.area == "fields"
    assert result.summary == 'Say "hi" A\\B/C D E \\Z'


@pytest.mark.parametrize(
    "text",
    [
        r'{"kind": "bug", "priority": "medium", "summary": "Bad \Z',
        r'{kind: "bug", priority: "medium", summary: "Bad \Z"}',
        r'{"kind": "bug", "priority": "urgent", "summary": "Bad \Z"}',
    ],
)
def test_parse_triage_repair_does_not_rescue_broken_replies(text):
    assert parse_triage(text) is None
