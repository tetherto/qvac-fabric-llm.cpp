#!/usr/bin/env python3
"""Validate explicit context, forced decoder dispatch, and precise CPU load policy."""
import json
import re
import benchmark as batch


def check_explicit_context(device):
    expected = None
    for mode in ('auto', '0', '1'):
        name = 'explicit-context-' + mode
        settings = {} if mode == 'auto' else {'PARAKEET_ANE_BATCH_DECODE': mode}
        data = batch.run(device, name, 'en-120s.wav', 'coreml', language='en',
            settings=settings, runs=1, warmup=0,
            extra=['--window-sec', '60', '--overlap', '0.0666667'])
        content = (batch.OUT / f'{device}-{name}.log').read_text()
        plans = re.findall(r'window_frames=(\d+) context_frames=(\d+)', content)
        if not plans or any(int(context) != 25 for _, context in plans):
            raise RuntimeError('Explicit encoder context was not preserved')
        if data['decoder_coreml_all_runs'] != (mode == '1'):
            raise RuntimeError('Explicit-context decoder backend did not honor its setting')
        if mode == '1' and ('context_preserved=1' not in content or not data.get('decoder_dispatches')):
            raise RuntimeError('Forced context-preserving Core ML decoder did not execute')
        normalized = batch.normalized(data['transcript'])
        if expected is None:
            expected = normalized
        if expected != normalized:
            raise RuntimeError('Explicit-context decoder changed normalized English words')
        print(device, mode, 'explicit context and decoder execution verified', flush=True)


def check_precise_loads(device):
    fixed_auxiliary = r"full-EU Core ML (other|gate) encoder loaded from '[^']*-encoder-(?:other|gate)\.mlmodelc'"
    for policy in ('cpu_only', 'cpu_and_gpu'):
        for language, wav in [('en', 'jfk.wav'), ('other', 'fr-multipiece-16k.wav'),
                              ('auto', 'fr-multipiece-16k.wav')]:
            name = f'precise-load-{policy}-{language}'
            data = batch.run(device, name, wav, 'coreml', language,
                settings={'PARAKEET_COREML_COMPUTE_UNITS': policy}, runs=1, warmup=0)
            content = (batch.OUT / f'{device}-{name}.log').read_text()
            if re.search(fixed_auxiliary, content):
                raise RuntimeError('Precise policy eagerly loaded an unusable fixed auxiliary')
            if not data['encoder_coreml_all_runs']:
                raise RuntimeError('Precise policy encoder or gate fell back')
            if language == 'auto' and 'TranscribePsy EU gate encoder: coreml' not in content:
                raise RuntimeError('Automatic precise routing did not execute its Core ML gate')
            print(device, policy, language, 'precise path without unused fixed auxiliaries', flush=True)


def main():
    batch.configure()
    for device in batch.DEVICES:
        check_explicit_context(device)
        check_precise_loads(device)


if __name__ == '__main__':
    main()
