#!/usr/bin/env python3
"""Fetch and verify the pinned reference ASR model before benchmark timing."""
import argparse
import hashlib
import json
import pathlib
import re
import tempfile
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(spec, root):
    repo, revision, name, sha = (spec[k] for k in ('repo', 'revision', 'name', 'sha256'))
    if not re.fullmatch(r'[\w-]+/[\w.-]+', repo) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('reference ASR requires a repository and pinned commit')
    if pathlib.Path(name).name != name or not re.fullmatch(r'[0-9a-f]{64}', sha):
        raise ValueError('reference ASR requires a filename and SHA-256')
    target = root / 'reference-asr' / sha / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and digest(target) == sha:
        return target
    url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
    # Publish only a complete, verified download; interrupted transfers never
    # poison the model cache. Concurrent processes use separate temporary files.
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = pathlib.Path(stream.name)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                stream.write(chunk)
        if digest(temporary) != sha:
            raise ValueError('reference ASR model checksum mismatch')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=pathlib.Path, required=True)
    parser.add_argument('--family', required=True)
    parser.add_argument('--models-root', type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        spec = json.loads(args.spec.read_text())[args.family]['correctness']['asr_model']
        path = prepare(spec, args.models_root.resolve())
        result = {'status': 'ok', 'model': str(path), 'sha256': spec['sha256']}
    except Exception as error:
        result = {'status': 'unavailable', 'model': None, 'reason': str(error)}
    print(json.dumps(result))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
