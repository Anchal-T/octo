"""Text actions for repository agents. The first newline separates metadata from bytes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from agentteam_common import is_safe_relative_path, is_test_path

from .sandbox import validate_test_node_id

RAW_ACTION_PROTOCOL = "p1-repro-raw-v2"
RAW_ACTION_FORMAT = (
    "Reply with a single JSON header on line 1, without Markdown or prose. "
    'For a shell command, reply {"action": "bash"} on line 1, then put the '
    "raw command on the following lines. "
    'For a file edit, reply {"action": "write_file", "path": "relative/path", "append": true} '
    "on line 1, then put the exact file contents to append on the following lines. "
    'Omit "append" to create or replace the file. '
    "The raw body runs until the end of your reply. Keep its quotes, backslashes, "
    "indentation and newlines exactly as needed."
)
REPRO_ACTION_FORMAT = (
    RAW_ACTION_FORMAT + '\nFor a test, reply {"action": "run_test", '
    '"test_node_id": "tests/test_x.py::test_y"}. '
    'To submit, reply {"action": "submit", "test_node_id": "tests/test_x.py::test_y", '
    '"summary": "...", "fact_claim": "optional", "anchor_names": ["a.b.C"]}. '
    "The run_test and submit actions have no body."
)
FIX_ACTION_FORMAT = (
    RAW_ACTION_FORMAT + '\nTo submit, reply {"action": "submit", "summary": "..."} '
    "with no body."
)


@dataclass(frozen=True)
class BashAction:
    command: str
    kind: Literal["bash"] = "bash"


@dataclass(frozen=True)
class WriteFileAction:
    path: str
    content: str
    append: bool = False
    kind: Literal["write_file"] = "write_file"


@dataclass(frozen=True)
class RunTestAction:
    test_node_id: str
    kind: Literal["run_test"] = "run_test"


@dataclass(frozen=True)
class SubmitReproAction:
    test_node_id: str
    summary: str
    fact_claim: str | None = None
    anchor_names: tuple[str, ...] = ()
    kind: Literal["submit"] = "submit"


@dataclass(frozen=True)
class SubmitFixAction:
    summary: str
    kind: Literal["submit"] = "submit"


@dataclass(frozen=True)
class ActionParseError:
    reason: str
    line: int = 1
    column: int = 1

    def observation(self) -> str:
        return f"Invalid action at line {self.line}, column {self.column}: {self.reason}."


def repro_submission(node: str, data: dict) -> SubmitReproAction:
    summary = data.get("summary", "Reproduced reported issue with failing test.")
    claim = data.get("fact_claim")
    anchors = data.get("anchor_names", [])
    if (
        not isinstance(claim, str)
        or not claim.strip()
        or not isinstance(anchors, list)
        or any(not isinstance(name, str) for name in anchors)
    ):
        claim, anchors = None, []
    return SubmitReproAction(
        test_node_id=node,
        summary=" ".join(summary.split())[:300].strip() or "Reproduced issue.",
        fact_claim=claim.strip()[:400] if claim else None,
        anchor_names=tuple(anchors),
    )


def fix_submission(summary: str) -> SubmitFixAction:
    return SubmitFixAction(summary=" ".join(summary.split())[:300].strip() or "Applied fix.")


def repro_node(node: str) -> str:
    validated = validate_test_node_id(node)
    path, separator, test = validated.partition("::")
    if not is_test_path(path) or not path.endswith(".py") or not separator or not test:
        raise ValueError("test_node_id must name a pytest test in a test file (path::test)")
    return validated


def parse_raw_action(
    text: str | None, *, reproducer: bool
) -> BashAction | WriteFileAction | RunTestAction | SubmitReproAction | SubmitFixAction | ActionParseError:
    header, newline, body = (text or "").partition("\n")

    def field_error(field: str, reason: str) -> ActionParseError:
        return ActionParseError(reason, column=max(1, header.find(json.dumps(field)) + 1))

    try:
        data = json.loads(header)
    except json.JSONDecodeError as error:
        return ActionParseError(error.msg, error.lineno, error.colno)
    except (RecursionError, ValueError):
        return ActionParseError("JSON header is too complex")
    if not isinstance(data, dict):
        return ActionParseError("header must be a JSON object")

    fields = {"bash": set(), "write_file": {"path", "append"}, "submit": {"summary"}}
    if reproducer:
        fields["run_test"] = {"test_node_id"}
        fields["submit"].update({"test_node_id", "fact_claim", "anchor_names"})
    action = data.get("action")
    if not isinstance(action, str) or action not in fields:
        return field_error("action", f"action must be one of {', '.join(fields)}")
    extra = data.keys() - fields[action] - {"action"}
    if extra:
        field = min(extra)
        return field_error(field, "unexpected field in the JSON header" + (
            "; put the command in the raw body" if action == "bash" else ""
        ))

    if action in {"bash", "write_file"}:
        if not newline or (action == "bash" and not body.strip()):
            return ActionParseError(f"{action} requires a raw body after the header", line=2)
        if action == "bash":
            return BashAction(body)
        path = data.get("path")
        if not isinstance(path, str) or not is_safe_relative_path(path):
            return field_error("path", "write_file requires a safe relative path")
        append = data.get("append", False)
        if not isinstance(append, bool):
            return field_error("append", "append must be a JSON boolean")
        return WriteFileAction(path, body, append)

    if body.strip():
        before_content = body[:len(body) - len(body.lstrip())]
        return ActionParseError(
            f"{action} accepts only the JSON header",
            line=2 + before_content.count("\n"),
            column=1 + len(before_content.rsplit("\n", 1)[-1]),
        )
    if action == "submit":
        if not isinstance(data.get("summary"), str) or not data["summary"].strip():
            return field_error("summary", "submit requires a nonempty summary")
        if not reproducer:
            return fix_submission(data["summary"])
    try:
        node = repro_node(data.get("test_node_id", ""))
    except ValueError as error:
        return field_error("test_node_id", str(error))
    if action == "run_test":
        return RunTestAction(node)
    if "fact_claim" in data and not isinstance(data["fact_claim"], str):
        return field_error("fact_claim", "fact_claim must be a string")
    anchors = data.get("anchor_names", [])
    if not isinstance(anchors, list) or any(not isinstance(name, str) for name in anchors):
        return field_error("anchor_names", "anchor_names must be a JSON array of strings")
    return repro_submission(node, data)
