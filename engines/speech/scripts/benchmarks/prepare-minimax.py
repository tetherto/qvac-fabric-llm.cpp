#!/usr/bin/env python3
"""Provision pinned MiniMax sources and verified GGUF pairs outside benchmark timing."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CONVERTER = REPO / 'audiogen/scripts/convert-minimax-music3-to-gguf.py'
DEFAULT_MANIFEST = HERE / 'minimax-model.json'
CHUNK = 8 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 120
MODEL_ROLES = ('lm', 'synth')
SOURCE_ROLES = ('lm', 'dit', 'vocoder')
IMPORTED_IDENTITY_FIELDS = ('manifest', 'quant', 'converter_sha256', 'preparer_sha256',
                            'requirements_sha256')


class PreparationError(RuntimeError):
    def __init__(self, stage: str, reason: str):
        super().__init__(reason)
        self.stage = stage


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b''):
            value.update(chunk)
    return value.hexdigest()


def json_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate(spec: dict) -> None:
    if spec.get('schema_version') != 1:
        raise ValueError('unsupported MiniMax manifest schema')
    if not isinstance(spec.get('license'), dict) or not spec['license'].get('name'):
        raise ValueError('manifest must identify the model license')
    files = spec.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('manifest must enumerate source artifacts')
    names, roles = set(), set()
    for item in files:
        name = item.get('path', '')
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or '..' in path.parts or '\\' in name
                or str(path) != name or name in names):
            raise ValueError('invalid or duplicate artifact path')
        names.add(name)
        roles.add(item.get('role'))
        if not re.fullmatch(r'[\w.-]+/[\w.-]+', item.get('model_id', spec.get('model_id', ''))):
            raise ValueError('invalid Hugging Face repository')
        if not re.fullmatch(r'[0-9a-f]{40}', item.get('revision', spec.get('revision', ''))):
            raise ValueError('source revision must be an immutable commit')
        if not re.fullmatch(r'[0-9a-f]{64}', item.get('sha256', '')):
            raise ValueError('invalid source SHA-256')
        if type(item.get('size')) is not int or item['size'] <= 0:
            raise ValueError('source size must be positive')
    if not {'lm', 'dit', 'vocoder', 'license'}.issubset(roles):
        raise ValueError('manifest must include LM, DiT, vocoder and license')
    for key in ('min_memory_bytes', 'conversion_free_disk_bytes'):
        if type(spec.get('resource_requirements', {}).get(key)) is not int or spec['resource_requirements'][key] <= 0:
            raise ValueError(f'invalid resource requirement: {key}')


def safe_path(root: Path, relative: str) -> Path:
    """Reject symlinks in cache paths, including symlinks to another cache entry."""
    path = root / relative
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f'symlink in model cache: {parent}')
        if parent == root:
            break
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('artifact escapes model cache')
    return path


def matches(path: Path, item: dict) -> bool:
    return path.is_file() and path.stat().st_size == item['size'] and digest(path) == item['sha256']


def download(item: dict, spec: dict, target: Path, offline: bool, opener=None,
             verified: bool | None = None) -> None:
    destination = safe_path(target, item['path'])
    source_verified = matches(destination, item) if verified is None else verified
    if source_verified:
        return
    if offline:
        raise PreparationError('download', f'offline source missing or corrupt: {item["path"]}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    repo = item.get('model_id', spec['model_id'])
    revision = item.get('revision', spec['revision'])
    url = f'https://huggingface.co/{repo}/resolve/{revision}/{item["path"]}'
    opener = opener or urllib.request.urlopen
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.download-', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        value, size = hashlib.sha256(), 0
        with opener(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, temporary.open('wb') as stream:
            for chunk in iter(lambda: response.read(CHUNK), b''):
                size += len(chunk)
                if size > item['size']:
                    raise PreparationError('download', f'source exceeds pinned size: {item["path"]}')
                value.update(chunk)
                stream.write(chunk)
        if size != item['size'] or value.hexdigest() != item['sha256']:
            raise PreparationError('download', f'source size/checksum mismatch: {item["path"]}')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def dependency_identity() -> dict:
    """Hash installed GGUF Python support; never silently use a different writer."""
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'gguf', 'PyYAML', 'tqdm')}
    if versions != locked_versions():
        raise ValueError('converter dependencies differ from requirements-minimax.txt')
    module = importlib.util.find_spec('gguf')
    if module is None or module.origin is None:
        raise ValueError('gguf is not installed; install the pinned MiniMax requirements')
    source_root = Path(module.origin).parent
    sources = {str(path.relative_to(source_root)): digest(path) for path in sorted(source_root.rglob('*.py'))}
    return {'versions': versions, 'gguf_sources': sources,
            'python': platform.python_version(), 'platform': sys.platform,
            'machine': platform.machine(), 'byteorder': sys.byteorder}


def locked_versions() -> dict:
    lock = (HERE / 'requirements-minimax.txt').read_text()
    return {name: re.search(rf'(?mi)^{name}==([^\s\\]+)', lock).group(1)
            for name in ('numpy', 'gguf', 'PyYAML', 'tqdm')}


def build_metadata(quantizer: Path | None) -> dict:
    helper_spec = importlib.util.spec_from_file_location('music_provenance', HERE / 'music-generation-provenance.py')
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    return {'source_checkout': helper.source_identity(),
            'quantizer_build': helper.build_identity(quantizer) if quantizer else None}


def library_identity(paths: list[Path]) -> dict:
    result = {}
    for path in sorted({path.resolve() for path in paths}):
        if not path.is_file() or not path.name.startswith(('libggml', 'libqvac-speech-ggml')):
            raise ValueError(f'expected a ggml shared library: {path}')
        if path.name in result:
            raise ValueError(f'duplicate ggml library name: {path.name}')
        result[path.name] = {'sha256': digest(path), 'size': path.stat().st_size}
    return result


def cache_identity(spec: dict, quant: str, dependencies: dict, quantizer: Path | None,
                   libraries: dict | None = None, native_build: dict | None = None) -> dict:
    return {'manifest': spec, 'quant': quant, 'converter_sha256': digest(CONVERTER),
            'preparer_sha256': digest(Path(__file__)),
            'requirements_sha256': digest(HERE / 'requirements-minimax.txt'),
            'dependencies': dependencies,
            'quantizer_sha256': digest(quantizer) if quantizer else None,
            'quantizer_libraries': libraries or {}, 'quantizer_build': native_build}


def memory_bytes() -> int:
    if sys.platform == 'linux':
        info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(info['MemAvailable'].strip().split()[0]) * 1024
        for limit_file, used_file in [('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                                      ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
            try:
                limit = Path(limit_file).read_text().strip()
                if limit != 'max':
                    available = min(available, max(0, int(limit) - int(Path(used_file).read_text())))
            except (OSError, ValueError):
                pass
        return available
    if sys.platform == 'darwin':
        return int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
    raise ValueError('resource preflight supports Linux and macOS')


def preflight(spec: dict, root: Path, missing_bytes: int) -> None:
    requirements = spec['resource_requirements']
    required = requirements['conversion_free_disk_bytes'] + missing_bytes
    if shutil.disk_usage(root).free < required:
        raise PreparationError('preflight', f'insufficient free disk: need {required} bytes for sources and conversion')
    required = requirements['min_memory_bytes']
    if memory_bytes() < required:
        raise PreparationError('preflight', f'insufficient memory: need {required} bytes for conversion')


def extra_model_candidates(target: Path, quant: str) -> list[str]:
    expected = {target / f'mm3-{role}-{quant}.gguf' for role in ('lm', 'synth')}
    extras = []
    for directory in (target, safe_path(target, 'mm3')):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if (path.is_file() and re.fullmatch(r'mm3-(?:lm|synth)-.+\.gguf', path.name, re.IGNORECASE)
                    and path not in expected):
                extras.append(str(path.relative_to(target)))
    return sorted(extras)


def verify_pair(target: Path, identity: dict) -> dict | None:
    provenance = safe_path(target, 'provenance.json')
    if not provenance.is_file():
        return None
    try:
        result = json.loads(provenance.read_text())
        if result.get('identity') != identity or result.get('cache_key') != json_digest(identity):
            return None
        outputs = result.get('outputs', {})
        expected = {f'mm3-{role}-{identity["quant"]}.gguf' for role in ('lm', 'synth')}
        if set(outputs) != expected:
            return None
        if extra_model_candidates(target, identity['quant']):
            return None
        for name, item in outputs.items():
            path = safe_path(target, name)
            if not matches(path, item):
                return None
            with path.open('rb') as stream:
                if stream.read(4) != b'GGUF':
                    return None
        license_item = next(item for item in identity['manifest']['files'] if item['role'] == 'license')
        if not matches(safe_path(target, 'LICENSE'), license_item):
            return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


@contextlib.contextmanager
def cache_lock(root: Path):
    import fcntl
    with safe_path(root, '.preparation.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PreparationError('cache', 'another MiniMax preparation is using this models root') from error
        yield


@contextlib.contextmanager
def preparation_stage(name: str):
    try:
        yield
    except PreparationError:
        raise
    except Exception as error:
        raise PreparationError(name, str(error)) from error


def verified_result(target: Path, key: str, quant: str, cached: bool) -> dict:
    return {'status': 'ok', 'stage': 'verified', 'reason': None, 'cached': cached,
            'model_dir': str(target), 'cache_key': key, 'quant': quant,
            'provenance': str(target / 'provenance.json')}


def validate_imported_dependencies(identity: dict) -> None:
    dependencies = identity.get('dependencies', {})
    if dependencies.get('versions') != locked_versions():
        raise ValueError('prepared pair has incompatible converter dependency versions')
    support = dependencies.get('gguf_sources')
    if (not isinstance(support, dict) or not support or
            any(not re.fullmatch(r'[0-9a-f]{64}', value) for value in support.values())):
        raise ValueError('prepared pair lacks GGUF support source hashes')


def validate_imported_quantizer(identity: dict) -> None:
    if not re.fullmatch(r'[0-9a-f]{64}', identity.get('quantizer_sha256') or ''):
        raise ValueError('prepared q4 pair lacks quantizer provenance')
    libraries = identity.get('quantizer_libraries')
    if not isinstance(libraries, dict) or not libraries:
        raise ValueError('prepared q4 pair lacks ggml library provenance')
    for item in libraries.values():
        if (not isinstance(item, dict) or not re.fullmatch(r'[0-9a-f]{64}', item.get('sha256', ''))
                or type(item.get('size')) is not int or item['size'] <= 0):
            raise ValueError('prepared q4 pair lacks ggml library provenance')


def verify_imported_pair(spec: dict, quant: str, target: Path) -> dict:
    with preparation_stage('cache'):
        target = target.resolve()
        record = json.loads(safe_path(target, 'provenance.json').read_text())
        identity = record['identity']
        expected = cache_identity(spec, quant, {}, None)
        for name in IMPORTED_IDENTITY_FIELDS:
            if identity.get(name) != expected[name]:
                raise ValueError(f'prepared pair has incompatible {name}')
        validate_imported_dependencies(identity)
        if quant == 'q4_k_m':
            validate_imported_quantizer(identity)
        if not verify_pair(target, identity):
            raise ValueError('prepared GGUF pair missing, incomplete, corrupt or incompatible')
        return verified_result(target, record['cache_key'], quant, True)


def select_quantizer(quant: str, quantizer: Path | None, libraries: list[Path]) -> Path | None:
    if quant != 'q4_k_m':
        return None
    if quantizer is None or not quantizer.is_file() or not os.access(quantizer, os.X_OK):
        raise ValueError('q4_k_m requires --quantizer pointing to an executable acestep-quantize')
    if not libraries:
        raise ValueError('q4_k_m requires --quantizer-library for every ggml shared library used by acestep-quantize')
    return quantizer.resolve()


def prepare_sources(spec: dict, root: Path, source: Path, offline: bool) -> None:
    with preparation_stage('preflight'):
        verified = {item['path']: matches(safe_path(source, item['path']), item)
                    for item in spec['files']}
        missing = sum(item['size'] for item in spec['files'] if not verified[item['path']])
        if offline and missing:
            raise PreparationError('download', 'offline source cache missing or corrupt')
        preflight(spec, root, missing)
        source.mkdir(parents=True, exist_ok=True)
    with preparation_stage('download'):
        download_sources(spec, source, offline, verified)


def download_sources(spec: dict, source: Path, offline: bool, verified: dict) -> None:
    for item in spec['files']:
        print(f'[prepare-minimax] verifying source {item["path"]}', file=sys.stderr, flush=True)
        download(item, spec, source, offline, verified=verified[item['path']])


def convert_sources(spec: dict, source: Path, staging: Path, quant: str) -> list[str]:
    conversion_quant = 'f16' if quant == 'q4_k_m' else quant
    command = [sys.executable, str(CONVERTER), '--out', str(staging), '--quant', conversion_quant]
    for item in spec['files']:
        if item['role'] in SOURCE_ROLES:
            command.extend(['--src', str(source / item['path'])])
    subprocess.run(command, check=True, stdout=sys.stderr, stderr=sys.stderr)
    return command


def quantize_pair(staging: Path, quant: str, quantizer: Path) -> list[list[str]]:
    commands = []
    for role in MODEL_ROLES:
        original = staging / f'mm3-{role}-f16.gguf'
        command = [str(quantizer), str(original), str(staging / f'mm3-{role}-{quant}.gguf'), 'Q4_K_M']
        subprocess.run(command, check=True, stdout=sys.stderr, stderr=sys.stderr)
        original.unlink()
        commands.append(command)
    return commands


def verify_outputs(staging: Path, quant: str) -> dict:
    outputs = {}
    for role in MODEL_ROLES:
        name = f'mm3-{role}-{quant}.gguf'
        path = staging / name
        with path.open('rb') as stream:
            if stream.read(4) != b'GGUF' or path.stat().st_size <= 24:
                raise ValueError(f'converter did not produce a GGUF: {name}')
        outputs[name] = {'sha256': digest(path), 'size': path.stat().st_size}
    return outputs


def write_preparation_record(spec: dict, source: Path, staging: Path, identity: dict,
                             build: dict, commands: list[list[str]], outputs: dict) -> None:
    license_item = next(item for item in spec['files'] if item['role'] == 'license')
    shutil.copyfile(source / license_item['path'], staging / 'LICENSE')
    record = {'schema_version': 1, 'cache_key': json_digest(identity), 'identity': identity,
              'outputs': outputs, 'source_dir': str(source),
              'conversion_command': commands[0], 'commands': commands,
              'source_checkout': build['source_checkout'], 'license': spec['license']}
    (staging / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')


def publish_pair_provenance_last(staging: Path, target: Path, outputs: dict) -> None:
    target.mkdir(parents=True, exist_ok=True)
    safe_path(target, 'provenance.json').unlink(missing_ok=True)
    for name in (*outputs, 'LICENSE', 'provenance.json'):
        (staging / name).replace(safe_path(target, name))


def build_pair(spec: dict, source: Path, target: Path, identity: dict, build: dict,
               quantizer: Path | None) -> None:
    quant = identity['quant']
    with preparation_stage('conversion'):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=target.parent, prefix=f'.{target.name}-')
    with temporary as directory:
        staging = Path(directory)
        with preparation_stage('conversion'):
            commands = [convert_sources(spec, source, staging, quant)]
        if quantizer:
            with preparation_stage('quantization'):
                commands.extend(quantize_pair(staging, quant, quantizer))
        with preparation_stage('output-verification'):
            outputs = verify_outputs(staging, quant)
            write_preparation_record(spec, source, staging, identity, build, commands, outputs)
        with preparation_stage('publication'):
            publish_pair_provenance_last(staging, target, outputs)


def prepare_cached_pair(spec: dict, root: Path, identity: dict, build: dict,
                        quantizer: Path | None, offline: bool, verify_only: bool) -> dict:
    key, quant = json_digest(identity), identity['quant']
    with preparation_stage('cache'):
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        target = safe_path(root, f'minimax/{key}')
        source = safe_path(root, f'minimax-sources/{json_digest(spec)}')
        with cache_lock(root):
            extras = extra_model_candidates(target, quant)
            if extras:
                raise PreparationError('cache', f'unverified MiniMax candidates in model cache: {extras}')
            if verify_pair(target, identity):
                return verified_result(target, key, quant, True)
            if verify_only:
                raise PreparationError('cache', 'verified GGUF pair missing, incomplete, corrupt or incompatible')
            prepare_sources(spec, root, source, offline)
            build_pair(spec, source, target, identity, build, quantizer)
        return verified_result(target, key, quant, False)


def prepare(spec: dict, root: Path, quant: str = 'q8_0', *, offline: bool = False,
            verify_only: bool = False, quantizer: Path | None = None,
            prepared_dir: Path | None = None, quantizer_libraries: list[Path] | None = None,
            cache_key_only: bool = False) -> dict:
    with preparation_stage('manifest'):
        validate(spec)
        if quant not in ('f16', 'q8_0', 'q4_k_m'):
            raise ValueError('unsupported quantization')
    if prepared_dir is not None:
        return verify_imported_pair(spec, quant, prepared_dir)
    with preparation_stage('manifest'):
        quantizer = select_quantizer(quant, quantizer, quantizer_libraries or [])
    with preparation_stage('dependencies'):
        build = build_metadata(quantizer)
        libraries = library_identity(quantizer_libraries or []) if quantizer else {}
        identity = cache_identity(spec, quant, dependency_identity(), quantizer, libraries, build['quantizer_build'])
    if cache_key_only:
        return {'status': 'ok', 'stage': 'identity', 'reason': None,
                'cache_key': json_digest(identity), 'quant': quant, 'model_dir': None,
                'provenance': None, 'cached': False}
    return prepare_cached_pair(spec, root, identity, build, quantizer, offline, verify_only)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-root', required=True, type=Path)
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument('--quant', choices=('f16', 'q8_0', 'q4_k_m'), default='q8_0')
    parser.add_argument('--quantizer', type=Path)
    parser.add_argument('--quantizer-library', action='append', type=Path, default=[],
                        help='each ggml shared library used by the quantizer (repeat for all libraries)')
    parser.add_argument('--offline', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--cache-key-only', action='store_true',
                      help='emit the exact preparation cache key without accessing models')
    mode.add_argument('--prepared-dir', type=Path,
                        help='verify a prepared pair from another builder; never download or convert')
    args = parser.parse_args()
    try:
        try:
            spec = json.loads(args.manifest.read_text())
        except (OSError, ValueError) as error:
            raise PreparationError('manifest', str(error)) from error
        result = prepare(spec, args.models_root, args.quant, offline=args.offline,
                         verify_only=args.verify_only, quantizer=args.quantizer,
                         prepared_dir=args.prepared_dir, quantizer_libraries=args.quantizer_library,
                         cache_key_only=args.cache_key_only)
    except PreparationError as error:
        result = {'status': 'preparation-failed', 'stage': error.stage, 'reason': str(error),
                  'model_dir': None, 'cache_key': None, 'provenance': None,
                  'quant': args.quant, 'cached': False}
    print(json.dumps(result, allow_nan=False))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
