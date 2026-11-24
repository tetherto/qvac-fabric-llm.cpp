"""Frozen, offline CLAP diagnostic policy shared by benchmark entry points.

Imports of optional numerical/model dependencies are deferred so missing preparation
can be reported as structured JSON. Never downloads weights during measurement.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import time
from pathlib import Path

POLICY_VERSION = 'clap-music-v2'
SAMPLE_RATE = 48000
WINDOW_SAMPLES = 480000


class AlignmentError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(model_dir, manifest_path):
    root = Path(model_dir).resolve()
    try:
        manifest = json.loads(Path(manifest_path).read_text())
        if not re.fullmatch(r'[a-f0-9]{40}', manifest['revision']):
            raise ValueError('revision must be an immutable 40-character commit')
        if not manifest.get('model_id'):
            raise ValueError('model_id is required')
        files = manifest['files']
        if not isinstance(files, dict) or not files:
            raise ValueError('files must contain artifact SHA256 hashes')
        required = {'config.json', 'preprocessor_config.json', 'tokenizer_config.json'}
        if not required.issubset(files) or not any(p.endswith(('.safetensors', '.bin')) for p in files):
            raise ValueError('manifest must include model, processor and tokenizer artifacts')
        if not ('tokenizer.json' in files or {'vocab.json', 'merges.txt'}.issubset(files)):
            raise ValueError('manifest must include tokenizer vocabulary')
        for name, expected in files.items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or not re.fullmatch(r'[a-f0-9]{64}', expected):
                raise ValueError(f'invalid artifact declaration: {name}')
            # Hugging Face snapshots may use symlinks into their blob cache.
            if sha256_file(root / relative) != expected:
                raise ValueError(f'artifact checksum mismatch: {name}')
        # Additional loader-visible model/tokenizer files must not escape the hash contract.
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.bin', '.safetensors', '.txt', '.model'):
                if str(path.relative_to(root)) not in files and path.resolve() != Path(manifest_path).resolve():
                    raise ValueError(f'unverified model artifact: {path.name}')
        if manifest.get('sampling_rate', SAMPLE_RATE) != SAMPLE_RATE or manifest.get('window_samples', WINDOW_SAMPLES) != WINDOW_SAMPLES:
            raise ValueError('model manifest disagrees with frozen audio policy')
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AlignmentError('model-missing', f'local model verification failed: {exc}') from exc


def load_clap_local(model_dir, manifest_path):
    manifest = verify_model(model_dir, manifest_path)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from transformers import ClapModel, ClapProcessor
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    processor = ClapProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = ClapModel.from_pretrained(str(model_dir), local_files_only=True)
    model.to(device='cpu', dtype=torch.float32).eval()
    extractor = processor.feature_extractor
    if int(extractor.sampling_rate) != SAMPLE_RATE or int(extractor.nb_max_samples) != WINDOW_SAMPLES:
        raise AlignmentError('scorer-error', 'processor sample rate/window differs from frozen policy')
    return model, processor, manifest


def validate_caption(processor, caption):
    if not isinstance(caption, str) or not caption.strip():
        raise AlignmentError('scorer-error', 'caption must be a nonempty string; lyrics are not scoring text')
    tokenizer = processor.tokenizer
    limit = int(tokenizer.model_max_length)
    if limit <= 0 or limit > 512:
        raise AlignmentError('scorer-error', 'tokenizer does not declare the expected bounded text limit')
    tokens = tokenizer(caption, add_special_tokens=True, truncation=False)['input_ids']
    if len(tokens) > limit:
        raise AlignmentError('scorer-error', f'caption has {len(tokens)} tokens; maximum is {limit}; truncation forbidden')
    return len(tokens)


def read_wav(path):
    """Decode PCM/IEEE-float WAV only, returning float64 (frames, channels).

    scipy returns 24-bit PCM left-aligned in int32, so scaling by the dtype
    width also preserves its original [-1, 1) amplitude. Float WAV amplitudes
    are retained unchanged, including nonfinite values for validity rejection.
    Other containers and compressed WAV codecs are intentionally unsupported.
    """
    import numpy as np
    import struct
    import warnings
    from scipy.io import wavfile
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='Reached EOF prematurely.*', category=wavfile.WavFileWarning)
            sample_rate, audio = wavfile.read(path, mmap=False)
        if audio.dtype.kind == 'u' and audio.dtype.itemsize == 1:
            audio = (audio.astype(np.float64) - 128.0) / 128.0
        elif audio.dtype.kind == 'i':
            audio = audio.astype(np.float64) / float(2 ** (8 * audio.dtype.itemsize - 1))
        elif audio.dtype.kind == 'f':
            audio = audio.astype(np.float64)
        else:
            raise ValueError(f'unsupported WAV sample type: {audio.dtype}')
        if audio.ndim == 1:
            audio = audio[:, None]
        if audio.ndim != 2 or sample_rate <= 0:
            raise ValueError('invalid WAV shape or sample rate')
        return audio, int(sample_rate)
    except (OSError, ValueError, EOFError, struct.error, wavfile.WavFileWarning) as exc:
        raise AlignmentError('invalid-audio', f'WAV decode failed: {exc}') from exc


def validate_audio(waveform, sample_rate):
    import numpy as np
    audio = np.asarray(waveform, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or not audio.shape[0] or not audio.shape[1] or sample_rate <= 0:
        raise AlignmentError('invalid-audio', 'empty PCM or invalid audio shape/sample rate')
    if not np.isfinite(audio).all():
        raise AlignmentError('invalid-audio', 'PCM contains NaN or infinity')
    if not np.any(audio):
        raise AlignmentError('invalid-audio', 'all-zero PCM')
    mono = audio.mean(axis=1).astype(np.float32)
    if not np.isfinite(mono).all() or not np.any(mono):
        raise AlignmentError('invalid-audio', 'mono signal is nonfinite or all-zero (possible stereo cancellation)')
    diagnostics = {
        'sample_rate': int(sample_rate), 'channels': int(audio.shape[1]), 'frames': int(audio.shape[0]),
        'duration_seconds': audio.shape[0] / sample_rate,
        'rms': float(np.sqrt(np.mean(audio ** 2))), 'peak': float(np.max(np.abs(audio))),
        'clipping_fraction': float(np.mean(np.abs(audio) >= 1.0)),
        'per_channel_energy': [float(x) for x in np.mean(audio ** 2, axis=0)],
        'mono_rms': float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))),
    }
    numeric = [diagnostics[key] for key in ('rms', 'peak', 'mono_rms')] + diagnostics['per_channel_energy']
    if not all(math.isfinite(value) for value in numeric):
        raise AlignmentError('invalid-audio', 'PCM diagnostics overflowed')
    return mono, diagnostics


def resample_to(waveform, orig_rate, target_rate=SAMPLE_RATE):
    import numpy as np
    from scipy.signal import resample_poly
    if orig_rate <= 0 or target_rate <= 0:
        raise AlignmentError('invalid-audio', 'sample rates must be positive')
    if orig_rate == target_rate:
        return np.asarray(waveform, dtype=np.float32)
    divisor = math.gcd(int(orig_rate), int(target_rate))
    result = resample_poly(waveform, target_rate // divisor, orig_rate // divisor,
                           window=('kaiser', 5.0), padtype='constant').astype(np.float32)
    if not np.isfinite(result).all() or not np.any(result):
        raise AlignmentError('invalid-audio', 'resampled mono signal is nonfinite or all-zero')
    return result


def fixed_windows(waveform, window_samples=WINDOW_SAMPLES):
    import numpy as np
    if window_samples <= 0 or len(waveform) == 0:
        raise ValueError('window and waveform lengths must be positive')
    for start in range(0, len(waveform), window_samples):
        window = waveform[start:start + window_samples]
        valid = len(window)
        yield start, valid, np.pad(window, (0, window_samples - valid), mode='constant')


def bounded_cosine(value):
    value = float(value)
    if not math.isfinite(value) or abs(value) > 1.0 + 1e-6:
        raise AlignmentError('scorer-error', 'nonfinite or out-of-range cosine similarity')
    return max(-1.0, min(1.0, value))


def weighted_score(windows):
    if not windows or any(w['valid_samples'] <= 0 for w in windows):
        raise AlignmentError('scorer-error', 'no positively weighted windows')
    return sum(bounded_cosine(w['score']) * w['valid_samples'] for w in windows) / sum(w['valid_samples'] for w in windows)


def embedding(output):
    import torch
    value = output if isinstance(output, torch.Tensor) else output.pooler_output
    norm = torch.linalg.vector_norm(value, dim=-1)
    if not torch.isfinite(value).all() or not torch.isfinite(norm).all() or torch.any(norm == 0):
        raise AlignmentError('scorer-error', 'nonfinite or zero-norm embedding')
    return torch.nn.functional.normalize(value, dim=-1)


def score_loaded_audio(model, processor, waveform, sample_rate, caption, device='cpu'):
    import torch
    token_count = validate_caption(processor, caption)
    mono, audio = validate_audio(waveform, sample_rate)
    mono = resample_to(mono, sample_rate)
    text_inputs = processor(text=[caption], return_tensors='pt', padding=True, truncation=False)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    windows = []
    with torch.inference_mode():
        text = embedding(model.get_text_features(**text_inputs))
        for offset, valid, samples in fixed_windows(mono):
            # Exact ten-second inputs bypass random cropping. Final zero padding
            # is applied here, never the extractor's default repeat padding.
            inputs = processor(audio=[samples], sampling_rate=SAMPLE_RATE, return_tensors='pt',
                               padding='pad', truncation='rand_trunc')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            audio_emb = embedding(model.get_audio_features(**inputs))
            score = bounded_cosine((text * audio_emb).sum(dim=-1).item())
            windows.append({'offset_samples': offset, 'valid_samples': valid,
                            'window_samples': WINDOW_SAMPLES, 'score': score})
    return {'score': weighted_score(windows), 'windows': windows, 'audio': audio,
            'min_window_score': min(w['score'] for w in windows),
            'max_window_score': max(w['score'] for w in windows), 'caption_tokens': token_count}


def provenance(manifest):
    import torch
    packages = ('torch', 'transformers', 'numpy', 'scipy', 'tokenizers', 'huggingface-hub')
    # platform.platform() probes the processor with an external `uname` on Unix.
    # These uname fields come from the OS directly, without a subprocess.
    system = platform.uname()
    return {'model_id': manifest['model_id'], 'revision': manifest['revision'], 'files': manifest['files'],
            'device': 'cpu', 'dtype': 'float32', 'threads': 1,
            'inter_op_threads': torch.get_num_interop_threads(),
            'wave_decoder': {'name': 'scipy.io.wavfile', 'version': importlib.metadata.version('scipy')}, 'python': platform.python_version(),
            'platform': '-'.join((system.system, system.release, system.machine)),
            'dependencies': {name: importlib.metadata.version(name) for name in packages},
            'sampling_rate': SAMPLE_RATE, 'window_samples': WINDOW_SAMPLES,
            'mono': 'arithmetic-mean', 'resampler': 'scipy.signal.resample_poly',
            'resampler_window': ['kaiser', 5.0], 'resampler_padtype': 'constant',
            'window_policy': 'contiguous-from-zero-final-zero-pad-duration-weighted',
            'cosine_boundary_tolerance': 1e-6, 'text_policy': 'caption-only-no-truncation'}


def score_file(wav, caption, model_dir, manifest_path):
    started = time.perf_counter()
    result = {'metric': 'clap-cosine', 'policy_version': POLICY_VERSION,
              'status': 'scorer-error', 'score': None, 'reason': None, 'windows': [],
              'caption_sha256': hashlib.sha256(caption.encode('utf-8')).hexdigest()}
    try:
        model, processor, manifest = load_clap_local(model_dir, manifest_path)
        result['provenance'] = provenance(manifest)
        try:
            result['wav_sha256'] = sha256_file(wav)
            audio, rate = read_wav(wav)
        except (OSError, RuntimeError) as exc:
            raise AlignmentError('invalid-audio', f'WAV decode failed: {exc}') from exc
        result.update(score_loaded_audio(model, processor, audio, rate, caption))
        result.update(status='ok', reason=None)
    except ImportError as exc:
        result.update(status='dependency-missing', score=None, reason=str(exc))
    except AlignmentError as exc:
        result.update(status=exc.status, score=None, reason=str(exc))
    except Exception as exc:
        result.update(status='scorer-error', score=None, reason=str(exc))
    result['scorer_seconds'] = time.perf_counter() - started
    return result
