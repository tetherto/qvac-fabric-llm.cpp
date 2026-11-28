#!/usr/bin/env python3
"""Hash the generator binary and provisioned GGUF files outside generation timing."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def identity(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return {'path': str(path.resolve()), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def build_identity(binary):
    """Retain relevant build options without dumping arbitrary cache variables."""
    for directory in binary.resolve().parents:
        cache = directory / 'CMakeCache.txt'
        if not cache.is_file():
            continue
        options = {}
        for line in cache.read_text().splitlines():
            if line.startswith(('#', '//')) or '=' not in line or ':' not in line:
                continue
            key_type, value = line.split('=', 1)
            key, kind = key_type.split(':', 1)
            if key in {'CMAKE_BUILD_TYPE', 'CMAKE_C_COMPILER', 'CMAKE_CXX_COMPILER',
                       'CMAKE_C_COMPILER_ID', 'CMAKE_CXX_COMPILER_ID', 'CMAKE_GENERATOR'} or (
                    kind == 'BOOL' and key.startswith(('GGML_', 'AUDIOGEN_', 'SPEECH_'))):
                options[key] = value
        return {**identity(cache), 'options': options}
    return None


def source_identity():
    source = Path(__file__).resolve().parent.parent.parent
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=source,
                                             text=True, stderr=subprocess.DEVNULL).strip())
        return {'commit': commit, 'dirty': dirty,
                'scope': 'benchmark source checkout; binary identity is recorded separately'}
    except (OSError, subprocess.CalledProcessError):
        return None


def model_paths(root):
    return sorted(path for directory in (root, root / 'mm3') if directory.is_dir()
                  for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() == '.gguf')


def file_snapshot(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'bytes': stat.st_size, 'device': stat.st_dev,
            'inode': stat.st_ino, 'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}


def write_fingerprint_cache(path, record):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(record, stream, allow_nan=False)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def model_identities(root, cache_path=None):
    paths = model_paths(root)
    if cache_path is None:
        return [identity(path) for path in paths]
    snapshot = {'root': str(root.resolve()), 'files': [file_snapshot(path) for path in paths]}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached['snapshot'] != snapshot:
            raise ValueError('model files changed during the pilot; start a new pilot')
        return cached['model_files']
    identities = [identity(path) for path in paths]
    if snapshot['files'] != [file_snapshot(path) for path in model_paths(root)]:
        raise ValueError('model files changed while fingerprinting')
    write_fingerprint_cache(cache_path, {'snapshot': snapshot, 'model_files': identities})
    return identities


def provenance(root, binary, device, cache_path=None):
    preparation_file = root / 'provenance.json'
    preparation = None
    if preparation_file.is_file():
        preparation = {**identity(preparation_file),
                       'metadata': json.loads(preparation_file.read_text())}
    return {'binary': identity(binary), 'model_files': model_identities(root, cache_path),
            'requested_device': device, 'preparation': preparation,
            'build_cache': build_identity(binary), 'source_checkout': source_identity(),
            'environment': {name: os.environ.get(name) for name in (
                'MM3_DIT_NO_FLASH', 'MM3_LM_FLASH', 'MM3_LM_NO_FLASH')},
            'scope': 'all provisioned GGUFs in the generator model directory; native CLI selects stages by filename'}


if __name__ == '__main__':
    cache = os.environ.get('MUSIC_MODEL_FINGERPRINT_CACHE')
    print(json.dumps(provenance(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3],
                                Path(cache) if cache else None),
                     allow_nan=False))
