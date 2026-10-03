"""Update only the campaign marker section, preserving every historical byte outside it."""

from __future__ import annotations

import argparse
from pathlib import Path

from make_table import analyze, report

START = "<!-- dflash2-gpu-next -->"
END = "<!-- /dflash2-gpu-next -->"


def fill(text: str, generated: str) -> str:
    block = START + "\n" + generated.rstrip("\n") + "\n" + END
    starts, ends = text.count(START), text.count(END)
    if starts == ends == 0:
        separator = "" if not text or text.endswith("\n\n") else "\n" if text.endswith("\n") else "\n\n"
        return text + separator + block + "\n"
    if starts != 1 or ends != 1 or text.index(START) > text.index(END):
        raise ValueError("report requires exactly one ordered campaign marker pair, or none")
    start = text.index(START)
    end = text.index(END) + len(END)
    return text[:start] + block + text[end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument("--require-complete", action="store_true", help="write partial evidence, but exit 1 if aggregate is withheld")
    args = parser.parse_args()
    try:
        result = analyze(args.index)
        # Explicit newline handling preserves historical CRLF sections too.
        with args.report.open("r", newline="") as source:
            old = source.read()
        updated = fill(old, report(result))
        with args.report.open("w", newline="") as output:
            output.write(updated)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(str(args.report))
    if args.require_complete and result["aggregate_error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
