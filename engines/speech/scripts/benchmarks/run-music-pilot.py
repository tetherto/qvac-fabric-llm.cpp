#!/usr/bin/env python3
"""Run the frozen music pilot using the benchmark family's generation driver.

A limited run retains the full expected cohort and labels unattempted outputs.
Configuration JSON requires name, checkpoint_identity, backend, build_options.
Checkpoint identity names the original checkpoint, shared by quantized variants.
"""
import argparse
import hashlib
import importlib.util
from itertools import product
import json
import math
import os
import signal
import re
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def normalize_score(score):
    if not isinstance(score, dict):
        return {'status': 'scorer-error', 'score': None, 'reason': 'malformed score object'}
    score = dict(score)
    value = score.get('score')
    if score.get('status') == 'ok':
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1:
            return {'status': 'scorer-error', 'score': None, 'reason': 'invalid cosine score'}
    else:
        score['score'] = None
    return score


def validate_manifest(manifest):
    seeds = manifest['seeds']
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('manifest seeds must be unique nonnegative integers')
    ids = set()
    for prompt in manifest['prompts']:
        identifier = prompt['id']
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', identifier) or identifier in ids:
            raise ValueError('manifest prompt IDs must be unique safe path components')
        ids.add(identifier)
        if prompt['subset'] not in ('primary', 'stress') or not prompt['caption'].strip():
            raise ValueError('manifest prompt requires a subset and nonempty caption')
    if not ids:
        raise ValueError('manifest has no prompts')
    if manifest['bootstrap']['replicates'] < 100:
        raise ValueError('bootstrap requires at least 100 replicates')


def extract_score(path):
    if not path.exists():
        return {'status': 'scorer-error', 'score': None, 'reason': 'scorer did not write JSON'}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError) as error:
        return {'status': 'scorer-error', 'score': None, 'reason': str(error)}


def run_bounded(command, *, timeout, **kwargs):
    """Kill the whole driver process group on timeout, including generators."""
    with subprocess.Popen(command, start_new_session=True, **kwargs) as process:
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise


def score_control(wav, caption, destination, args):
    command = [os.environ.get('MUSIC_ALIGNMENT_PYTHON', sys.executable),
               str(HERE / 'compute-music-alignment.py'), '--wav', str(wav),
               '--caption', caption, '--model-dir', str(args.scorer_model_dir),
               '--model-manifest', str(args.scorer_manifest), '--json-out', str(destination)]
    try:
        with destination.with_suffix('.log').open('w') as log:
            run_bounded(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.scorer_timeout)
        return normalize_score(extract_score(destination))
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'score': None, 'reason': 'control scorer timeout'}


def parse_inputs():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['acestep', 'minimax'], required=True)
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=HERE / 'fixtures/music-pilot-v1.json')
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--models-root', type=Path, default=Path('bench-models'))
    parser.add_argument('--model-dir', type=Path, help='Provisioned generation models; bypass registry')
    parser.add_argument('--scorer-model-dir', type=Path, required=True)
    parser.add_argument('--scorer-manifest', type=Path, default=HERE / 'music-alignment-model.json')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Explicit partial smoke cohort, including all missing IDs in report')
    parser.add_argument('--generation-timeout', type=float, default=3600)
    parser.add_argument('--scorer-timeout', type=float, default=300)
    parser.add_argument('--controls', action='store_true', help='Same-WAV fresh-process and different-category caption controls; one per category')
    parser.add_argument('--repeat-generation', action='store_true', help='Repeat first successful prompt/seed generation once')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    manifest = json.loads(args.manifest.read_text())
    try:
        validate_manifest(manifest)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    config = json.loads(args.configuration.read_text())
    for key in ('name', 'checkpoint_identity', 'backend', 'build_options'):
        if key not in config or not config[key]:
            parser.error(f'configuration requires {key}')
    device = str(config['backend']).lower()
    if device not in ('cpu', 'gpu'):
        parser.error('configuration backend must be cpu or gpu; requested device is applied through MUSIC_DEVICE')
    if args.out_dir.exists():
        parser.error('--out-dir must be new to prevent overwriting retained evidence')
    return args, manifest, config


def record_configuration(config, settings):
    config['generation_settings'] = settings
    config['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    config['source_diff_sha256'] = hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=HERE)).hexdigest()
    config['source_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=HERE, text=True).strip())
    config['source_pinned'] = not config['source_dirty']
    config['benchmark_source_hashes'] = {str(p.relative_to(HERE)): digest(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.suffix in ('.py', '.sh', '.json', '.txt')}


def prepare_pilot(args, manifest, config):
    args.out_dir.mkdir(parents=True)
    output = args.out_dir.resolve()
    record_configuration(config, manifest['generation'][args.family])
    pilot = {'schema_version': 1, 'manifest': manifest, 'manifest_sha256': digest(args.manifest),
             'family': args.family, 'configuration': config, 'records': [], 'controls': [],
             'listening_review': 'pending human observations',
             'validity_controls': 'see model-free scorer tests; empirical corruption controls not run'}
    write(output / 'manifest.json', manifest)
    write(output / 'configuration.json', config)
    return pilot


def build_environment(args, config):
    env = os.environ.copy()
    env.update(MUSIC_ALIGNMENT='1', MUSIC_DEVICE=str(config['backend']).lower(), MUSIC_ALIGNMENT_MODEL=str(args.scorer_model_dir.resolve()),
               MUSIC_ALIGNMENT_MANIFEST=str(args.scorer_manifest.resolve()),
               MUSIC_ALIGNMENT_TIMEOUT=str(args.scorer_timeout),
               MUSIC_MODEL_FINGERPRINT_CACHE=str(args.out_dir.resolve() / 'model-fingerprints.json'))
    if args.model_dir:
        env['MUSIC_ALIGNMENT_MODEL_DIR'] = str(args.model_dir.resolve())
    return env


def load_benchmark_module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initialize_model_fingerprints(args, env):
    root = Path(env.get('MUSIC_ALIGNMENT_MODEL_DIR') or args.models_root / args.family)
    report = env.get('MINIMAX_PREPARATION_REPORT') if args.family == 'minimax' else None
    if report:
        preparation = extract_score(Path(report))
        if not isinstance(preparation, dict) or preparation.get('status') != 'ok' or not preparation.get('model_dir'):
            return
        root = Path(preparation['model_dir'])
    validator = load_benchmark_module('validate-music-models')
    if validator.validate(root, args.family):
        generation = load_benchmark_module('music-generation-provenance')
        generation.model_identities(root, Path(env['MUSIC_MODEL_FINGERPRINT_CACHE']))


def build_record_environment(env, prompt, seed, settings):
    env = env.copy()
    env.update(MUSIC_CAPTION=prompt['caption'], MUSIC_LYRICS=prompt['lyrics'], MUSIC_SEED=str(seed),
               MUSIC_DURATION=str(settings.get('duration_seconds') or 20),
               MUSIC_MAX_FRAMES=str(settings.get('max_frames') or 300))
    return env


def generation_command(args, result_path):
    return ['bash', str(HERE / 'run-family.sh'), '--family', args.family, '--runs', '1', '--warmup', '0',
            '--build-dir', str(args.build_dir.resolve()), '--models-root', str(args.models_root.resolve()),
            '--out', str(result_path)]


def read_generation_result(row, result_path, output):
    result = extract_score(result_path) if result_path.exists() else {
        'status': 'run-failed', 'notes': 'generation driver did not write result.json; see driver.log'}
    scored_runs = result.get('music_alignment', {}).get('runs', [])
    score = normalize_score(scored_runs[0] if scored_runs else result.get('music_alignment', {}))
    row.update(status=score.get('status', result.get('status', 'run-failed')),
               score=score.get('score'), reason=score.get('reason') or result.get('notes'),
               scorer_provenance=({'policy_version': score.get('policy_version'), **score['provenance']}
                                  if score.get('provenance') else None),
               result=str(result_path.relative_to(output)))
    row['performance_result'] = result
    artifact_dir = score.get('artifact_dir')
    generation_file = Path(artifact_dir) / 'generation.json' if artifact_dir else None
    row['generation'] = json.loads(generation_file.read_text()) if generation_file and generation_file.is_file() else None
    row['score_artifact'] = score


def run_record(row, run_dir, args, env):
    run_dir.mkdir()
    result_path = run_dir / 'result.json'
    started = time.monotonic()
    try:
        with (run_dir / 'driver.log').open('w') as log:
            run_bounded(generation_command(args, result_path), cwd=HERE.parent.parent, env=env,
                        stdout=log, stderr=subprocess.STDOUT, timeout=args.generation_timeout)
        read_generation_result(row, result_path, run_dir.parent)
    except subprocess.TimeoutExpired:
        row.update(status='timeout', reason='generation driver timeout')
    row['elapsed_seconds'] = time.monotonic() - started


def run_caption_controls(prompt, row, wav, run_dir, args, manifest):
    mismatch = next(p for p in manifest['prompts']
                    if p['subset'] == 'primary' and p['category'] != prompt['category'])
    return {'prompt_id': prompt['id'], 'seed': row['seed'], 'category': prompt['category'],
            'matched': row['score_artifact'], 'wav_sha256': digest(wav),
            'repeat': score_control(wav, prompt['caption'], run_dir / 'repeat-score.json', args),
            'mismatch_prompt_id': mismatch['id'],
            'mismatch': score_control(wav, mismatch['caption'], run_dir / 'mismatch-score.json', args)}


def repeat_generation(row, run_dir, args, env):
    control = {'kind': 'repeat-generation', 'prompt_id': row['prompt_id'], 'seed': row['seed'],
               'original_result': row.get('performance_result'),
               'original_score': normalize_score(row.get('score_artifact')),
               'original_wav_sha256': None, 'repeat_wav_sha256': None,
               'wav_bytes_equal': None, 'score_delta': None,
               'score_comparison_status': 'unavailable'}
    repeat_path = run_dir / 'repeat-result.json'
    # Hash the original before starting another generation, preserving evidence
    # even if a faulty driver reuses its output path.
    original_wavs = sorted(run_dir.glob('result.music.*/run-*/audio.wav'))
    if len(original_wavs) == 1:
        control['original_wav_sha256'] = digest(original_wavs[0])
    try:
        with (run_dir / 'repeat-driver.log').open('w') as log:
            returncode = run_bounded(generation_command(args, repeat_path), cwd=HERE.parent.parent, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, timeout=args.generation_timeout)
        control.update(status='ok' if returncode == 0 else 'run-failed', returncode=returncode)
    except subprocess.TimeoutExpired:
        control.update(status='timeout', reason='repeat generation driver timeout')
    except OSError as error:
        control.update(status='run-failed', reason=str(error))

    control['result'] = extract_score(repeat_path)
    result = control['result'] if isinstance(control['result'], dict) else {}
    alignment = result.get('music_alignment')
    alignment = alignment if isinstance(alignment, dict) else {}
    runs = alignment.get('runs')
    repeated = normalize_score(runs[0] if isinstance(runs, list) and runs else alignment)
    control['repeat_score'] = repeated
    if control['status'] == 'ok':
        if not repeat_path.is_file():
            control.update(status='missing-result', reason='repeat generation driver did not write result JSON')
        elif result.get('status') != 'ok':
            control.update(status=result.get('status') or 'run-failed',
                           reason=result.get('reason') or result.get('notes') or 'repeat generation failed')
        elif repeated.get('status') != 'ok':
            control.update(status=repeated.get('status') or 'scorer-error',
                           reason=repeated.get('reason') or 'repeat generation was not scored')

    repeat_wavs = sorted(run_dir.glob('repeat-result.music.*/run-*/audio.wav'))
    if len(repeat_wavs) == 1:
        control['repeat_wav_sha256'] = digest(repeat_wavs[0])
    if control['original_wav_sha256'] and control['repeat_wav_sha256']:
        control['wav_bytes_equal'] = control['original_wav_sha256'] == control['repeat_wav_sha256']
    elif control['status'] == 'ok':
        control.update(status='missing-audio', reason='expected one retained original WAV and one repeat WAV')

    original = control['original_score']
    if control['status'] == 'ok' and original.get('status') == repeated.get('status') == 'ok':
        same_policy = (original.get('policy_version') and original.get('provenance')
                       and original['policy_version'] == repeated.get('policy_version')
                       and original['provenance'] == repeated.get('provenance'))
        if same_policy:
            control.update(score_delta=repeated['score'] - original['score'], score_comparison_status='ok')
        else:
            control['score_comparison_status'] = 'incompatible-policy'
    return control


def run_controls(prompt, row, run_dir, args, env, pilot, state):
    wav_paths = sorted(run_dir.glob('result.music.*/run-*/audio.wav'))
    if row['status'] != 'ok' or not wav_paths:
        return
    if args.controls and prompt['category'] not in state['categories'] and prompt['subset'] == 'primary':
        state['categories'].add(prompt['category'])
        pilot['controls'].append(run_caption_controls(prompt, row, wav_paths[0], run_dir, args, pilot['manifest']))
    if args.repeat_generation and not state['repeated']:
        state['repeated'] = True
        pilot['controls'].append(repeat_generation(row, run_dir, args, env))


def run_cohort(args, pilot, env):
    output = args.out_dir.resolve()
    manifest = pilot['manifest']
    settings = pilot['configuration']['generation_settings']
    control_state = {'categories': set(), 'repeated': False}
    for index, (prompt, seed) in enumerate(product(manifest['prompts'], manifest['seeds'])):
        row = {'prompt_id': prompt['id'], 'seed': seed, 'status': 'not-run', 'score': None,
               'reason': 'outside explicitly limited cohort' if args.limit is not None else 'not attempted'}
        pilot['records'].append(row)
        if args.limit is not None and index >= args.limit:
            continue
        run_dir = output / f"{prompt['id']}-{seed}"
        record_env = build_record_environment(env, prompt, seed, settings)
        run_record(row, run_dir, args, record_env)
        run_controls(prompt, row, run_dir, args, record_env, pilot, control_state)
        write(output / 'pilot.json', pilot)
    write(output / 'pilot.json', pilot)


def main():
    args, manifest, config = parse_inputs()
    pilot = prepare_pilot(args, manifest, config)
    env = build_environment(args, config)
    initialize_model_fingerprints(args, env)
    run_cohort(args, pilot, env)
    print(args.out_dir.resolve() / 'pilot.json')


if __name__ == '__main__':
    main()
