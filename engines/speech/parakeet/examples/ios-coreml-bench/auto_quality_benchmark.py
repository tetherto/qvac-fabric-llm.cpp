#!/usr/bin/env python3
"""Check automatic encoder quality independently of decoder crossover/presence."""
import benchmark as batch


def require_french_words(data, repetitions):
    words = batch.normalized(data['transcript']).split()
    canonical = ['ses' if word == 'ces' else word for word in words]
    expected = batch.normalized(batch.REFERENCES['fr-multipiece-16k.wav']).split() * repetitions
    if canonical != expected:
        raise RuntimeError('French speech was dropped, duplicated, or changed beyond homophone spelling')


def check_default_quality(device, models, missing_decoder):
    prefix = 'no-decoder-' if missing_decoder else 'auto-quality-'
    for name, wav, mode in [('short', 'fr-multipiece-16k.wav', 'auto'),
                            ('long', 'fr-long.wav', 'auto'),
                            ('above-crossover', 'en-120s.wav', 'auto'),
                            ('continuous', 'fr-long.wav', '0'),
                            ('forced', 'fr-long.wav', '1')]:
        settings = {} if mode == 'auto' else {'PARAKEET_ANE_BATCH_DECODE': mode}
        data = batch.run(device, prefix + name, wav, 'coreml', settings=settings,
            models=models, runs=1, warmup=0)
        expected_decoder = not missing_decoder and (mode == '1' or
            name == 'above-crossover' or name == 'long' and device == 'mac')
        if data['decoder_coreml_all_runs'] != expected_decoder:
            raise RuntimeError('Automatic/forced decoder dispatch differs from the measured policy')
        if wav.startswith('fr'):
            require_french_words(data, 12 if wav == 'fr-long.wav' else 1)
        elif not data['normalized_exact']:
            raise RuntimeError('English automatic segmentation changed the reference words')
        if mode == '0' and data['encoder_backend'] not in ('coreml-cpu', 'coreml-cpu-gpu'):
            raise RuntimeError('Continuous automatic encoding did not use the precise Core ML path')
        print(device, prefix + name, 'quality/dispatch verified', flush=True)
    if missing_decoder:
        data = batch.run(device, prefix + 'owned-long', 'fr-long.wav', 'coreml', models=models,
            settings={'PARAKEET_COREML_ONLY': '1'}, runs=1, warmup=0)
        require_french_words(data, 12)
        if not data['encoder_owned_by_coreml'] or data['decoder_coreml_all_runs']:
            raise RuntimeError('Encoder-only ownership or native decoder flag differs')


def main():
    batch.configure()
    for device in batch.DEVICES:
        check_default_quality(device, 'models', False)
        check_default_quality(device, 'models-nodecoder', True)


if __name__ == '__main__':
    main()
