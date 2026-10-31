"""Write the held-out quality-gate windows listed in windows.json.

usage: build_windows.py <external dir> <output dir> [--repo <llama.cpp checkout>]

Repository windows are read with `git show <repo_commit>:<path>` from the checkout (default: this repository).
External windows are read from <external dir>, which must hold the files named in windows.json; their sha256 is checked.
Every window is checked against its sha256 and written to <output dir>/w<tokens>/<name>.txt.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("external", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo", type=Path, default=HERE.parents[2])
    args = parser.parse_args()

    manifest = json.loads((HERE / "windows.json").read_text())
    sources: dict[str, str] = {}
    for name, info in manifest["external"].items():
        text = (args.external / name).read_text(encoding="utf-8", errors="replace")
        if sha256(text) != info["sha256"]:
            print(f"{name}: sha256 differs from the file used for the gate (download from {info['url']})", file=sys.stderr)
            return 1
        sources[name] = text

    failed = 0
    for window in manifest["windows"]:
        source = window["source"]
        if window["kind"] == "repo" and source not in sources:
            sources[source] = subprocess.run(["git", "-C", str(args.repo), "show", f"{manifest['repo_commit']}:{source}"],
                                             check=True, capture_output=True).stdout.decode("utf-8", errors="replace")
        text = sources[source][window["offset"]:window["offset"] + window["chars"]]
        if sha256(text) != window["sha256"]:
            print(f"{window['name']}: sha256 mismatch", file=sys.stderr)
            failed += 1
            continue
        target = args.output / f"w{window['tokens']}" / f"{window['name']}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(f"{len(manifest['windows']) - failed} of {len(manifest['windows'])} windows written to {args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
