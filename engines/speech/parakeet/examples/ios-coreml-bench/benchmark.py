#!/usr/bin/env python3
"""Reproduce full-EU serial benchmarks on desktop and a provisioned iPhone."""
import json, os, re, shutil, subprocess, time, unicodedata, wave
from pathlib import Path
ROOT = OUT = BINARY = DEVICE = BUNDLE = None
DEVICES = []
ROWS = []
BASE_ENV = {k: v for (k, v) in os.environ.items() if not k.startswith('PARAKEET_') and (not k.startswith('DEVICECTL_CHILD_'))}
REFERENCES = {'jfk.wav': 'And so my fellow Americans ask not what your country can do for you ask what you can do for your country', 'fr-multipiece-16k.wav': 'Ses analyses linguistiques et esthétiques restent politiques', 'silence800.wav': '', 'silence20.wav': ''}
for n in [1, 4, 8, 16, 20]:
    REFERENCES[f'en-{n * 15}s.wav'] = ' '.join([REFERENCES['jfk.wav']] * n)
REFERENCES['fr-long.wav'] = ' '.join([REFERENCES['fr-multipiece-16k.wav']] * 12)

def normalized(s):
    return ' '.join(re.findall('\\w+', unicodedata.normalize('NFKC', s).casefold()))

def distance(a, b):
    row = list(range(len(b) + 1))
    for (i, x) in enumerate(a, 1):
        nxt = [i]
        for (j, y) in enumerate(b, 1):
            nxt.append(min(row[j] + 1, nxt[-1] + 1, row[j - 1] + (x != y)))
        row = nxt
    return row[-1]

def run(device, name, wav, backend, language='auto', settings=None, models='models', quant='f16', runs=5, warmup=2, extra=None):
    label = f'{device}-{name}'
    log = OUT / (label + '.log')
    target = OUT / (label + '.json')
    for previous in ROWS:
        if previous['device'] == device and previous['case'] == name:
            print('REUSE', label, flush=True)
            return previous
    settings = dict(settings or {})
    if backend != 'coreml':
        settings['PARAKEET_COREML_DISABLE'] = '1'
    args = ['--model', str(ROOT / models / f'transcribepsy-250m-eu.{quant}.gguf') if device == 'mac' else f'{models}/transcribepsy-250m-eu.{quant}.gguf', '--wav', str(ROOT / 'inputs' / wav) if device == 'mac' else wav, '--language', language, '--backend', 'cpu' if backend == 'coreml' else backend, '--threads', '4', '--bench', '--bench-runs', str(runs), '--bench-warmup', str(warmup), '--bench-json', str(target) if device == 'mac' else target.name, '--verbose']
    if backend == 'coreml':
        args += ['--require-coreml']
    args += list(extra or [])
    if device == 'iphone':
        wait_for_phone_cooldown()
    started = time.time()
    print('START', label, flush=True)
    command, environment = build_command(device, args, settings)
    content = execute_benchmark(device, command, environment, log, target, label)
    data = json.loads(target.read_text())
    if settings.get('PARAKEET_COREML_ONLY') == '1' and not data.get('encoder_owned_by_coreml'):
        raise RuntimeError('Core ML-only benchmark did not omit native encoder weights')
    data.update({'device': device, 'case': name, 'language': language, 'settings': settings, 'model_format': quant, 'sidecars': models, 'wall_seconds': time.time() - started, 'wav_name': wav, 'command': command})
    record_runtime_metrics(data, content, device)
    score_transcript(data, wav)
    if device == 'iphone' and data.get('thermal_state', 0) != 0:
        receipt = OUT / f'{label}.thermal-{time.time_ns()}'
        shutil.copyfile(log, Path(str(receipt) + '.log'))
        Path(str(receipt) + '.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print('RETRY after thermal cooldown', label, flush=True)
        return run(device, name, wav, backend, language, settings, models, quant, runs, warmup, extra)
    ROWS.append(data)
    (OUT / 'results.json').write_text(json.dumps(ROWS, ensure_ascii=False, indent=2))
    print('DONE', label, 'median', data['inference_ms']['median'], 'WER', data.get('wer'), 'load', data['load_ms'], 'RSS', data.get('peak_rss_bytes'), flush=True)
    return data

def wait_for_phone_cooldown():
    maximum_attempts = 40
    interval_seconds = 15
    command = ['xcrun', 'devicectl', 'device', 'process', 'launch', '--device', DEVICE,
               '--terminate-existing', '--console', BUNDLE, '--status']
    for attempt in range(maximum_attempts):
        probe = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        match = re.search(r'thermal=(\d+)', probe.stdout + probe.stderr)
        if match and int(match.group(1)) == 0:
            return
        if not match:
            raise RuntimeError('Phone status probe did not report thermal state')
        print('COOLDOWN', attempt + 1, flush=True)
        time.sleep(interval_seconds)
    raise RuntimeError('Phone did not return to nominal thermal state')


def build_command(device, args, settings):
    if device == 'mac':
        return ['/usr/bin/time', '-l', str(BINARY)] + args, {**BASE_ENV, **settings}
    command = ['xcrun', 'devicectl', 'device', 'process', 'launch', '--device', DEVICE,
               '--terminate-existing', '--console']
    if settings:
        command += ['--environment-variables', json.dumps(settings)]
    return command + ['--', BUNDLE] + args, BASE_ENV


def execute_benchmark(device, command, environment, log, target, label):
    maximum_launch_attempts = 3
    for attempt in range(maximum_launch_attempts):
        with log.open('w') as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                    env=environment, timeout=600)
        content = log.read_text(errors='replace')
        if result.returncode == 0 and (device != 'iphone' or 'PHONE_METRICS rc=0' in content):
            break
        no_app_output = not any(marker in content for marker in ('ggml_', 'PHONE_', 'parakeet:', 'error:'))
        if device != 'iphone' or not no_app_output or attempt + 1 == maximum_launch_attempts:
            raise RuntimeError(f'{label} failed: {content[-2000:]}')
        shutil.copyfile(log, log.with_name(log.stem + f'.launch-{attempt}.log'))
        print('RETRY empty device launch output', label, flush=True)
        time.sleep(1)
    if device == 'iphone':
        target.unlink(missing_ok=True)
        subprocess.run(['xcrun', 'devicectl', 'device', 'copy', 'from', '--device', DEVICE,
                        '--domain-type', 'appDataContainer', '--domain-identifier', BUNDLE,
                        '--source', 'Documents/' + target.name, '--destination', str(target)],
                       capture_output=True, text=True, timeout=60, check=True)
    return content


def record_runtime_metrics(data, content, device):
    gates = re.findall('p\\(non-English\\)=([0-9.]+).*?route=(en|other)', content)
    if gates:
        data['gate_probability'] = float(gates[-1][0])
        data['gate_route'] = gates[-1][1]
    dispatches = [int(v) for v in re.findall('EU decoder: coreml, dispatches=(\\d+)', content)]
    if dispatches:
        data['decoder_dispatches'] = dispatches
    peak = re.search('peak_rss_bytes=(\\d+)', content) if device == 'iphone' else re.search('(\\d+)\\s+maximum resident set size', content)
    if peak:
        data['peak_rss_bytes'] = int(peak.group(1))
    footprint = re.search(r'peak_footprint_bytes=(\d+)', content) if device == 'iphone' else re.search(r'(\d+)\s+peak memory footprint', content)
    if footprint:
        data['peak_footprint_bytes'] = int(footprint.group(1))
    thermal = re.search('thermal=(\\d+)', content)
    if thermal:
        data['thermal_state'] = int(thermal.group(1))


def score_transcript(data, wav):
    if wav in REFERENCES:
        ref = normalized(REFERENCES[wav])
        got = normalized(data['transcript'])
        edits = distance(ref.split(), got.split())
        chars = distance(ref, got)
        data['reference'] = REFERENCES[wav]
        data['word_edits'] = edits
        data['reference_words'] = len(ref.split())
        data['wer'] = edits / max(1, len(ref.split()))
        data['cer'] = chars / max(1, len(ref))
        data['normalized_exact'] = ref == got


def main():
    for device in DEVICES:
        for (wav, key) in [('jfk.wav', 'en'), ('fr-multipiece-16k.wav', 'fr')]:
            for backend in ['cpu', 'metal', 'coreml']:
                run(device, f'{backend}-{key}-auto', wav, backend)
            for language in ['en', 'other']:
                run(device, f'coreml-{key}-{language}', wav, 'coreml', language)
        for wav, key in [('jfk.wav', 'en'), ('fr-multipiece-16k.wav', 'fr')]:
            run(device, f'hybrid-metal-{key}', wav, 'coreml', runs=3, warmup=1,
                extra=['--backend', 'metal'])
            run(device, f'owned-hybrid-metal-{key}', wav, 'coreml', runs=3, warmup=1,
                settings={'PARAKEET_COREML_ONLY':'1'}, extra=['--backend', 'metal'])
        run(device, 'hybrid-unwindowed-en', 'en-60s.wav', 'coreml', runs=3, warmup=1,
            settings={'PARAKEET_COREML_ONLY':'1'}, extra=['--backend', 'metal', '--window-sec', '0'])
        for (wav, key) in [('en-300s.wav', 'long-en'), ('fr-long.wav', 'long-fr')]:
            for backend in ['cpu', 'metal', 'coreml']:
                run(device, f'{backend}-{key}', wav, backend, runs=3, warmup=1, extra=['--window-sec', '60', '--overlap', '0.25'])
        for (wav, key) in [('en-300s.wav', 'en'), ('fr-long.wav', 'fr')]:
            run(device, f'native-decoder-long-{key}', wav, 'coreml', settings={'PARAKEET_ANE_BATCH_DECODE': '0'}, runs=3, warmup=1, extra=['--window-sec', '60', '--overlap', '0.25'])
        for (wav, key) in [('en-60s.wav', 'en'), ('fr-long.wav', 'fr')]:
            for backend in ['cpu', 'metal', 'coreml']:
                run(device, f'unwindowed-{backend}-{key}', wav, backend, runs=3, warmup=1, extra=['--window-sec', '0'])
            run(device, f'unwindowed-decoder-{key}', wav, 'coreml', settings={'PARAKEET_ANE_BATCH_DECODE': '1'}, runs=3, warmup=1, extra=['--window-sec', '0'])
        for (wav, key) in [('en-300s.wav', 'en'), ('fr-long.wav', 'fr')]:
            run(device, f'palette6-long-{key}', wav, 'coreml', models='models6', runs=3, warmup=1, extra=['--window-sec', '60', '--overlap', '0.25'])
        for units in ['all', 'cpu_and_gpu', 'cpu_only']:
            for (wav, key) in [('jfk.wav', 'en'), ('fr-multipiece-16k.wav', 'fr')]:
                run(device, f'units-{units}-{key}', wav, 'coreml', settings={'PARAKEET_COREML_COMPUTE_UNITS': units}, runs=3, warmup=1)
        for units in ['all', 'cpu_and_gpu', 'cpu_only', 'cpu_and_ane']:
            run(device, f'decoder-units-{units}', 'jfk.wav', 'coreml', language='en', settings={'PARAKEET_COREML_COMPUTE_UNITS': units, 'PARAKEET_ANE_BATCH_DECODE': '1'}, runs=3, warmup=1)
        for (wav, key) in [('jfk.wav', 'en'), ('fr-multipiece-16k.wav', 'fr')]:
            run(device, f'only-{key}', wav, 'coreml', settings={'PARAKEET_COREML_ONLY': '1'}, runs=3, warmup=1)
            run(device, f'q8-{key}', wav, 'coreml', quant='q8_0', runs=3, warmup=1)
            run(device, f'palette6-{key}', wav, 'coreml', models='models6', runs=3, warmup=1)
        for (wav, key) in [('silence800.wav', 'silence'), ('en-partial.wav', 'partial'), ('hi-16k.wav', 'stability')]:
            run(device, f'coreml-{key}', wav, 'coreml', runs=1, warmup=0)
        for seconds in [15, 60, 120, 240]:
            for mode in ['coreml', 'host', 'cpu']:
                settings = {'PARAKEET_ANE_BATCH_DECODE': '1', 'PARAKEET_COREML_DECODER_COMPUTE_UNITS': 'cpu_only' if mode == 'cpu' else 'cpu_and_ane'}
                if mode == 'host':
                    settings['PARAKEET_ANE_DECODE_HOST'] = '1'
                run(device, f'crossover-{seconds}-{mode}', f'en-{seconds}s.wav', 'coreml', language='en', settings=settings, runs=3, warmup=1)

def configure(streaming=False):
    import argparse
    global ROOT, OUT, BINARY, DEVICE, BUNDLE, DEVICES, ROWS
    parser = argparse.ArgumentParser(description='Serial full-EU correctness/performance matrix; see README.md')
    parser.add_argument('--assets', type=Path, required=True, help='contains inputs/, models/, models6/')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, help='desktop Parakeet CLI')
    parser.add_argument('--device', help='paired CoreDevice identifier')
    parser.add_argument('--bundle', help='installed benchmark app identifier')
    parser.add_argument('--devices', nargs='+', choices=['mac', 'iphone'], default=['mac', 'iphone'])
    if streaming:
        parser.add_argument('--backends', nargs='+', choices=['cpu', 'metal', 'coreml'],
                            default=['cpu', 'metal', 'coreml'])
        parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if 'mac' in args.devices and args.binary is None:
        parser.error('--binary is required for mac')
    if 'iphone' in args.devices and (not (args.device and args.bundle)):
        parser.error('--device and --bundle are required for iphone')
    ROOT = args.assets.resolve()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    BINARY = args.binary.resolve() if args.binary else None
    DEVICE = args.device
    BUNDLE = args.bundle
    DEVICES = args.devices
    if (OUT / 'results.json').is_file():
        ROWS = json.loads((OUT / 'results.json').read_text())
    return args
if __name__ == '__main__':
    configure()
    main()
