"""Paste the local kernel headers into the source string before it compiles.

The kernel sources compile away from this directory, and both IRON's compile
cache and each ExternalFunction's object name are keyed on the source string.
A header has to be part of that string, or editing it leaves a stale object
behind. Each header is pasted once per source; the recursive includes between
them are resolved here too.
"""
from __future__ import annotations

import re
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_INCLUDE = re.compile(r'^[ \t]*#include[ \t]+"(xdna-[a-z0-9-]+\.h)"[ \t]*$', re.M)
_PRAGMA = re.compile(r'^[ \t]*#pragma[ \t]+once[ \t]*\n', re.M)


def inline(src: str) -> str:
    """Return `src` with every local `#include "xdna-*.h"` replaced by the
    header's text."""
    seen: set[str] = set()

    def expand(text: str) -> str:
        out, pos = [], 0
        for m in _INCLUDE.finditer(text):
            out.append(text[pos:m.start()])
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                # Each header is pasted once, so its include guard is noise -
                # and lands in the middle of a translation unit, where the
                # compiler warns about it.
                out.append(_PRAGMA.sub("", expand((_DIR / name).read_text())))
            pos = m.end()
        out.append(text[pos:])
        return "".join(out)

    return expand(src)


def load(path: Path) -> str:
    """Read a kernel source and inline the headers it includes."""
    return inline(path.read_text())
