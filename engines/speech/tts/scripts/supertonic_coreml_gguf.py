"""GGUF-side helpers for the Supertonic Core ML vocoder exporter.

Torch-free on purpose: export-supertonic-coreml.py imports this module for
tensor-name resolution, dequantization, and the sidecar stem rule, and
test/test_export_supertonic_coreml.py unit-tests these paths without torch.
"""
import numpy as np

QUANT_TAGS = ('f32', 'f16', 'bf16', 'q8_0', 'q5_0', 'q4_0')
FLOAT_TYPE_NAMES = ('F32', 'F16', 'BF16')
VOCODER_PREFIX = 'vocoder:'

EMBED_W = 'vocoder:node:/decoder/embed/net/Conv#1'
EMBED_B = 'vocoder:node:/decoder/embed/net/Conv#2'
HEAD_PRELU = 'vocoder:node:/decoder/head/act/PRelu#1'

# Pre-v3 converters shipped no alias arrays; mirror the runtime's
# kLegacyV2Aliases roster (supertonic_gguf.cpp) for the vocoder names.
LEGACY_V2_ALIASES = (
    (EMBED_W, 'vocoder:onnx::Conv_1440'),
    (EMBED_B, 'vocoder:onnx::Conv_1441'),
    (HEAD_PRELU, 'vocoder:onnx::PRelu_1505'),
) + tuple(
    (f'text_encoder:tts.ttl.speech_prompted_text_encoder.attention{layer}.{name}.linear.weight',
     f'text_encoder:onnx::MatMul_{number}')
    for layer, numbers in ((1, (3678, 3680, 3681)), (2, (3682, 3684, 3685)))
    for name, number in zip(('W_query', 'W_value', 'out_fc'), numbers)
)


def resolve_tensor_names(sources, names, aliases=(), alias_targets=()):
    """Source name -> storage name, honoring the GGUF alias arrays and the
    legacy pre-v3 fallbacks, exactly like the C++ loader."""
    resolved = dict(zip(sources, names))
    for alias, target in zip(aliases, alias_targets):
        if target in resolved:
            resolved[alias] = resolved[target]
    for canonical, legacy in LEGACY_V2_ALIASES:
        if canonical not in resolved and legacy in resolved:
            resolved[canonical] = resolved[legacy]
    return resolved


def is_block_quantized(type_name):
    return type_name not in FLOAT_TYPE_NAMES and not type_name.startswith('I')


def dequantized(t):
    """A GGUFReader tensor as a float32 array in ONNX row-major shape."""
    if t.tensor_type.name in ('F32', 'F16'):
        data = np.asarray(t.data).astype(np.float32)
    elif t.tensor_type.name == 'BF16':
        raw = np.asarray(t.data).view(np.uint16).reshape(-1)
        data = (raw.astype(np.uint32) << 16).view(np.float32)
    else:
        from gguf.quants import dequantize
        data = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)
    shape = [int(d) for d in reversed(t.shape)]
    return data.reshape(shape)


def collect_vocoder_tensors(resolved, tensors_by_name, path):
    """Every vocoder float tensor by source name, plus how many were
    block-quantized (the caller warns: their dequantized values would bake
    into the sidecar under the shared per-model name)."""
    tensors = {}
    quantized = 0
    for source, name in resolved.items():
        if not source.startswith(VOCODER_PREFIX):
            continue
        t = tensors_by_name.get(name)
        if t is None:
            raise ValueError(f'{path}: metadata names {name} for {source} but the tensor is missing')
        if t.tensor_type.name.startswith('I'):
            continue
        if is_block_quantized(t.tensor_type.name):
            quantized += 1
        tensors[source] = dequantized(t)
    return tensors, quantized


def read_gguf(path):
    from gguf import GGUFReader

    reader = GGUFReader(str(path))

    def field(name):
        return reader.fields['supertonic.' + name].contents()

    def field_strings(name):
        key = 'supertonic.' + name
        if key not in reader.fields:
            return []
        return [str(s) for s in field(name)]

    resolved = resolve_tensor_names(field_strings('source_names'),
                                    field_strings('tensor_names'),
                                    field_strings('source_aliases'),
                                    field_strings('source_alias_targets'))
    tensors, quantized = collect_vocoder_tensors(
        resolved, {t.name: t for t in reader.tensors}, path)
    if quantized:
        print(f'[load] WARNING: {quantized} vocoder tensors in {path} are block-quantized; '
              'the sidecar bakes their dequantized values under the shared per-model name -- '
              'export from the f32 or f16 tier for tier-independent numerics')

    hp = {
        'latent_channels': int(field('latent_channels')),
        'base_chunk_size': int(field('base_chunk_size')),
        'ttl_chunk_compress_factor': int(field('ttl_chunk_compress_factor')),
        'sample_rate': int(field('sample_rate')),
    }
    return tensors, hp


def sidecar_stem(gguf_path):
    """coreml_vocoder_sidecar_path's rule: drop the extension and a trailing
    quantisation tag, so every tier of the model shares one sidecar."""
    from pathlib import Path

    stem = Path(gguf_path).stem
    for sep in ('-', '.'):
        head, _, tag = stem.rpartition(sep)
        if head and tag.lower() in QUANT_TAGS:
            return head + '-vocoder'
    return stem + '-vocoder'
