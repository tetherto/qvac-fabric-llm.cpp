#!/usr/bin/env python3
"""Prepare checksum-verified CLAP artifacts before generation/performance timing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import urllib.request


DEFAULT_MANIFEST = Path(__file__).with_name('music-alignment-model.json')


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def validate(spec: dict) -> None:
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', spec.get('model_id', '')):
        raise ValueError('invalid CLAP model repository')
    if not re.fullmatch(r'[0-9a-f]{40}', spec.get('revision', '')):
        raise ValueError('CLAP revision must be an immutable commit')
    files = spec.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('CLAP manifest must enumerate artifact checksums')
    for name, checksum in files.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or '..' in path.parts or str(path) != name
                or '\\' in name or not path.parts):
            raise ValueError('invalid CLAP artifact path')
        if not isinstance(checksum, str) or not re.fullmatch(r'[0-9a-f]{64}', checksum):
            raise ValueError('invalid CLAP artifact SHA-256')


def prepare(spec: dict, root: Path, offline: bool = False, opener=None) -> Path:
    validate(spec)
    opener = opener or urllib.request.urlopen
    target = root.resolve() / 'reference-clap' / spec['revision']
    target.mkdir(parents=True, exist_ok=True)
    for name, checksum in spec['files'].items():
        destination = target / name
        # Reject symlink escapes even when a cache already exists.
        if not destination.resolve().is_relative_to(target.resolve()):
            raise ValueError('CLAP artifact escapes the model cache')
        if destination.is_file() and digest(destination) == checksum:
            continue
        if offline:
            raise FileNotFoundError(f'CLAP cache missing or checksum mismatch: {name}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/{spec["model_id"]}/resolve/{spec["revision"]}/{name}'
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with opener(url, timeout=120) as response, temporary.open('wb') as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    stream.write(chunk)
            if digest(temporary) != checksum:
                raise ValueError(f'CLAP artifact checksum mismatch: {name}')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--models-root', type=Path, required=True)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    try:
        spec = json.loads(args.manifest.read_text())
        target = prepare(spec, args.models_root, args.offline)
        result = {'status': 'ok', 'model_dir': str(target), 'reason': None,
                  'model_id': spec['model_id'], 'revision': spec['revision']}
    except Exception as error:
        result = {'status': 'model-missing', 'model_dir': None, 'reason': str(error)}
    print(json.dumps(result, allow_nan=False))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
