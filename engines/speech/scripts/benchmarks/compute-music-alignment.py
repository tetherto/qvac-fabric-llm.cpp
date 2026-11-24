#!/usr/bin/env python3
"""Offline caption-only music scoring; prints exactly one result object."""
import argparse
import json
from pathlib import Path
from music_alignment import score_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav', '--audio', required=True)
    parser.add_argument('--caption', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--model-manifest', '--manifest', required=True)
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args()
    result = score_file(args.wav, args.caption, args.model_dir, args.model_manifest)
    output = json.dumps(result, allow_nan=False, sort_keys=True) + '\n'
    if args.json_out:
        args.json_out.write_text(output)
    print(output, end='')
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
