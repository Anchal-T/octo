import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFINITION_ROLE = 1


@dataclass(frozen=True)
class ScipDefinition:
    symbol: str
    relative_path: str
    range: tuple[int, ...]
    enclosing_range: tuple[int, ...] | None


@dataclass(frozen=True)
class ScipIndex:
    _definitions: dict[str, tuple[ScipDefinition, ...]]
    sha: str | None = None

    @classmethod
    def load(cls, path: str | Path, sha: str | None = None) -> "ScipIndex":
        index: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(index, dict):
            raise TypeError("SCIP JSON root must be an object")
        documents = index.get("documents")
        if not isinstance(documents, list):
            raise TypeError("SCIP JSON must contain a documents list")

        definitions: dict[str, list[ScipDefinition]] = {}
        for document in documents:
            if not isinstance(document, dict):
                raise TypeError("SCIP document must be an object")
            relative_path = document.get("relative_path")
            occurrences = document.get("occurrences", [])
            if not isinstance(relative_path, str) or not isinstance(occurrences, list):
                raise TypeError(
                    "SCIP document must contain a relative_path and occurrences"
                )

            for occurrence in occurrences:
                if not isinstance(occurrence, dict):
                    raise TypeError("SCIP occurrence must be an object")
                roles = occurrence.get("symbol_roles")
                if type(roles) is not int or not roles & DEFINITION_ROLE:
                    continue

                symbol = occurrence.get("symbol")
                range_ = cls._range(occurrence.get("range"))
                enclosing_range = occurrence.get("enclosing_range")
                enclosing_range_ = (
                    cls._range(enclosing_range) if enclosing_range is not None else None
                )
                if not isinstance(symbol, str):
                    raise TypeError("SCIP definition occurrence must contain a symbol")
                if range_ is None or (
                    enclosing_range is not None and enclosing_range_ is None
                ):
                    raise ValueError("invalid SCIP definition range")

                definition = ScipDefinition(
                    symbol=symbol,
                    relative_path=relative_path,
                    range=range_,
                    enclosing_range=enclosing_range_,
                )
                definitions.setdefault(symbol, []).append(definition)

        return cls({symbol: tuple(items) for symbol, items in definitions.items()}, sha)

    def definitions_for(self, symbol: str) -> tuple[ScipDefinition, ...]:
        return self._definitions.get(symbol, ())

    def resolve(self, name: str) -> str | None:
        # ponytail: exact SCIP symbols only; add aliases if proposals need short names.
        return name if self.has_definition(name) else None

    def has_definition(self, symbol: str) -> bool:
        return bool(self.definitions_for(symbol))

    @staticmethod
    def _range(value: Any) -> tuple[int, ...] | None:
        if (
            not isinstance(value, list)
            or len(value) not in {3, 4}
            or any(type(part) is not int or part < 0 for part in value)
        ):
            return None
        return tuple(value)
