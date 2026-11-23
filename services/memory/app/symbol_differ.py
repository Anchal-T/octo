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
                for line in range(start, start + hunk.old_len):
                    sym = parent.innermost(hunk.file, line, line + 1)
                    if sym is not None:
                        changed.add(sym)
            if hunk.new_len:
                start = hunk.new_start - 1
                for line in range(start, start + hunk.new_len):
                    sym = child.innermost(hunk.file, line, line + 1)
                    if sym is not None:
                        changed.add(sym)
        return changed
