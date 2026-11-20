from collections.abc import Awaitable, Callable, Sequence
from hashlib import sha1
from typing import Any

from agentteam_contracts import PrFileChange


def object_sha(kind: str, body: bytes) -> str:
    return sha1(f"{kind} {len(body)}\0".encode() + body).hexdigest()


async def changed_tree_sha(
    base_sha: str | None,
    files: Sequence[PrFileChange],
    read_tree: Callable[[str], Awaitable[list[dict[str, Any]]]],
) -> str:
    entries = (
        {item["path"]: item for item in await read_tree(base_sha)} if base_sha else {}
    )
    children: dict[str, list[PrFileChange]] = {}
    for change in files:
        name, separator, tail = change.path.partition("/")
        if separator:
            children.setdefault(name, []).append(
                change.model_copy(update={"path": tail})
            )
        elif change.content is None:
            entries.pop(name, None)
        else:
            entries[name] = {
                "mode": "100644",
                "sha": object_sha("blob", change.content.encode("utf-8")),
            }

    for name, changes in children.items():
        existing = entries.get(name)
        child_sha = await changed_tree_sha(
            existing["sha"] if existing and existing["mode"] == "040000" else None,
            changes,
            read_tree,
        )
        if child_sha == object_sha("tree", b""):
            entries.pop(name, None)
        else:
            entries[name] = {"mode": "040000", "sha": child_sha}

    # Git sorts directory entries as if their names ended in "/".
    sorted_entries = sorted(
        entries.items(),
        key=lambda pair: (
            pair[0] + ("/" if pair[1]["mode"] == "040000" else "")
        ).encode(),
    )
    body = b"".join(
        f"{entry['mode'].lstrip('0')} {name}\0".encode() + bytes.fromhex(entry["sha"])
        for name, entry in sorted_entries
    )
    return object_sha("tree", body)
