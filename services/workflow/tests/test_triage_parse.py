import pytest

from services.workflow.app.triage import (
    TRIAGER_SYSTEM_PROMPT,
    parse_triage,
    triage_messages,
)

VALID = '{"kind": "bug", "priority": "medium", "summary": "Example summary."}'


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
    assert messages[0]["content"] == TRIAGER_SYSTEM_PROMPT
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
