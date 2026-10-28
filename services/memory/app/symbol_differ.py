from collections.abc import Iterable

from .git_repo import Hunk
from .scip import ScipIndex


class SymbolDiffer:
    @staticmethod
    def diff(parent: ScipIndex, child: ScipIndex, hunks: Iterable[Hunk]) -> set[str]:
        changed = set(parent.symbols ^ child.symbols)
        for hunk in hunks:
            if hunk.old_len:
                start = hunk.old_start - 1
                symbol = parent.innermost(hunk.file, start, start + hunk.old_len)
                if symbol is not None:
                    changed.add(symbol)
            if hunk.new_len:
                start = hunk.new_start - 1
                symbol = child.innermost(hunk.file, start, start + hunk.new_len)
                if symbol is not None:
                    changed.add(symbol)
        return changed
