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


_ISSUE_REF = re.compile(
    r"https?://github\.com/([\w.-]+/[\w.-]+)/issues/([1-9]\d*)\b"
    r"|(?<![\w/])([\w.-]+/[\w.-]+)#([1-9]\d*)\b"
    r"|(?<![\w/#])#([1-9]\d*)\b"
)


def issue_references(repo: str, text: str, *, issue_id: str = "") -> list[str]:
    references = []
    if match := re.fullmatch(r"gh-\d+-(\d+)", issue_id):
        references.append(f"{repo.lower()}#{int(match[1])}")
    for match in _ISSUE_REF.finditer(text):
        url_repo, url_number, full_repo, full_number, number = match.groups()
        reference = f"{(url_repo or full_repo or repo).lower()}#{int(url_number or full_number or number)}"
        if reference not in references:
            references.append(reference)
        if len(references) >= 100:
            break
    return references
