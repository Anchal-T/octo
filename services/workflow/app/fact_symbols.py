"""Extract symbol candidates; Memory resolves them at the run's base commit."""

import re

from agentteam_contracts import SymbolLocation, SymbolsResolve

_DOTTED = re.compile(r"\bsqlparse(?:\.[A-Za-z_]\w*)+")
_LOCATION = re.compile(
    r'(?:File ["\']([^"\'\n]+\.py)["\'], line (\d+)|([\w./-]+\.py):(\d+))'
)
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+")


def symbol_candidates(base_commit: str, text: str, diff: str = "") -> SymbolsResolve:
    names = list(dict.fromkeys(_DOTTED.findall(text + "\n" + diff)))[:200]
    locations: list[SymbolLocation] = []

    def add_location(file: str, line: int) -> None:
        # Stack traces commonly include absolute site-packages prefixes.
        if "/sqlparse/" in file:
            file = "sqlparse/" + file.rsplit("/sqlparse/", 1)[1]
        if len(locations) >= 200 or line < 1:
            return
        try:
            location = SymbolLocation(file=file, line=line)
        except ValueError:
            return
        if location not in locations:
            locations.append(location)

    for match in _LOCATION.finditer(text):
        file, line, short_file, short_line = match.groups()
        add_location(file or short_file, int(line or short_line))

    file = ""
    for line in diff.splitlines():
        if line.startswith("--- a/"):
            file = line[6:]
        elif line.startswith("--- "):
            file = ""
        elif file and (match := _HUNK.match(line)):
            start = int(match[1])
            length = int(match[2]) if match[2] is not None else 1
            for offset in range(min(length, 200)):
                add_location(file, start + offset)
    return SymbolsResolve(base_commit=base_commit, names=names, locations=locations)
