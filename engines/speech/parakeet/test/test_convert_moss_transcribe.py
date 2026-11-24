#!/usr/bin/env python3
"""Converter coverage for scripts/convert-moss-transcribe-to-gguf.py: fabricates a
tiny MOSS-Transcribe-Diarize checkpoint on disk, runs the converter, and
validates the emitted GGUF (tensor census, layouts, token types, stored prompt
ids, quantization fallback) with the gguf reader. Skips cleanly when numpy,
gguf, or tokenizers are absent.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
    import gguf
    import tokenizers  # noqa: F401
except ImportError as error:
    print(f"SKIP: {error.name} is not installed")
    sys.exit(0)

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

TD_MELS = 4
TD_FFT = 16
TD_ENC = 8
TD_ENC_FF = 16
TD_ENC_CTX = 8
TD_MERGE = 4
TD_TEXT = 32
TD_TEXT_FF = 64
TD_HEAD_DIM = 8
TD_ADDED = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|audio_start|>", "<|audio_end|>",
            "<|audio_pad|>", "<think>"]



def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, dict] = {}
    blobs = []
    offset = 0
    for name, tensor in tensors.items():
        data = np.ascontiguousarray(tensor, dtype=np.float32).tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    payload = json.dumps(header).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(payload)))
        f.write(payload)
        for blob in blobs:
            f.write(blob)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def byte_level_vocab() -> dict[str, int]:
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    extra = [byte for byte in range(256) if byte not in printable]
    mapping = {byte: byte for byte in printable} | {byte: 256 + i for i, byte in enumerate(extra)}
    return {chr(mapping[byte]): byte for byte in range(256)}


def transcribe_tokenizer_json() -> dict:
    vocab = byte_level_vocab()
    added = [{"id": len(vocab) + i, "content": token, "special": token != "<think>", "single_word": False,
              "lstrip": False, "rstrip": False, "normalized": False} for i, token in enumerate(TD_ADDED)]
    return {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": None,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                              "use_regex": True},
            "post_processor": None, "decoder": {"type": "ByteLevel", "add_prefix_space": False,
                                                "trim_offsets": True, "use_regex": True},
            "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                      "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
                      "ignore_merges": False, "vocab": vocab, "merges": []}}


def transcribe_encoder_tensors() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    prefix = "model.whisper_encoder."
    tensors = {prefix + "conv1.weight": rng.standard_normal((TD_ENC, TD_MELS, 3)),
               prefix + "conv1.bias": rng.standard_normal(TD_ENC),
               prefix + "conv2.weight": rng.standard_normal((TD_ENC, TD_ENC, 3)),
               prefix + "conv2.bias": rng.standard_normal(TD_ENC),
               prefix + "embed_positions.weight": rng.standard_normal((TD_ENC_CTX, TD_ENC)),
               prefix + "layer_norm.weight": np.ones(TD_ENC), prefix + "layer_norm.bias": np.zeros(TD_ENC)}
    layer = prefix + "layers.0."
    for name in ("self_attn.q_proj", "self_attn.v_proj", "self_attn.out_proj"):
        tensors[layer + name + ".weight"] = rng.standard_normal((TD_ENC, TD_ENC))
        tensors[layer + name + ".bias"] = rng.standard_normal(TD_ENC)
    tensors[layer + "self_attn.k_proj.weight"] = rng.standard_normal((TD_ENC, TD_ENC))
    for name in ("self_attn_layer_norm", "final_layer_norm"):
        tensors[layer + name + ".weight"] = np.ones(TD_ENC)
        tensors[layer + name + ".bias"] = np.zeros(TD_ENC)
    tensors[layer + "fc1.weight"] = rng.standard_normal((TD_ENC_FF, TD_ENC))
    tensors[layer + "fc1.bias"] = rng.standard_normal(TD_ENC_FF)
    tensors[layer + "fc2.weight"] = rng.standard_normal((TD_ENC, TD_ENC_FF))
    tensors[layer + "fc2.bias"] = rng.standard_normal(TD_ENC)
    return tensors


def transcribe_text_tensors(vocab_size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(4)
    prefix = "model.language_model."
    layer = prefix + "layers.0."
    q_dim, kv_dim = 2 * TD_HEAD_DIM, TD_HEAD_DIM
    return {prefix + "embed_tokens.weight": rng.standard_normal((vocab_size, TD_TEXT)),
            prefix + "norm.weight": np.ones(TD_TEXT),
            layer + "input_layernorm.weight": np.ones(TD_TEXT),
            layer + "self_attn.q_proj.weight": rng.standard_normal((q_dim, TD_TEXT)),
            layer + "self_attn.k_proj.weight": rng.standard_normal((kv_dim, TD_TEXT)),
            layer + "self_attn.v_proj.weight": rng.standard_normal((kv_dim, TD_TEXT)),
            layer + "self_attn.o_proj.weight": rng.standard_normal((TD_TEXT, q_dim)),
            layer + "self_attn.q_norm.weight": np.ones(TD_HEAD_DIM),
            layer + "self_attn.k_norm.weight": np.ones(TD_HEAD_DIM),
            layer + "post_attention_layernorm.weight": np.ones(TD_TEXT),
            layer + "mlp.gate_proj.weight": rng.standard_normal((TD_TEXT_FF, TD_TEXT)),
            layer + "mlp.up_proj.weight": rng.standard_normal((TD_TEXT_FF, TD_TEXT)),
            layer + "mlp.down_proj.weight": rng.standard_normal((TD_TEXT, TD_TEXT_FF))}


def transcribe_adaptor_tensors() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    prefix = "model.vq_adaptor.layers."
    return {prefix + "0.weight": rng.standard_normal((TD_TEXT, TD_ENC * TD_MERGE)),
            prefix + "0.bias": rng.standard_normal(TD_TEXT),
            prefix + "2.weight": rng.standard_normal((TD_TEXT, TD_TEXT)),
            prefix + "2.bias": rng.standard_normal(TD_TEXT),
            prefix + "3.weight": np.ones(TD_TEXT), prefix + "3.bias": np.zeros(TD_TEXT)}


def transcribe_tensors(vocab_size: int) -> dict[str, np.ndarray]:
    return {**transcribe_encoder_tensors(), **transcribe_text_tensors(vocab_size), **transcribe_adaptor_tensors()}


def transcribe_config(vocab_size: int) -> dict:
    return {"audio_token_id": 256 + TD_ADDED.index("<|audio_pad|>"), "audio_merge_size": TD_MERGE,
            "adaptor_input_dim": TD_ENC * TD_MERGE,
            "text_config": {"vocab_size": vocab_size, "hidden_size": TD_TEXT, "intermediate_size": TD_TEXT_FF,
                            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
                            "head_dim": TD_HEAD_DIM, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
                            "rope_theta": 1000000},
            "audio_config": {"num_mel_bins": TD_MELS, "d_model": TD_ENC, "encoder_layers": 1,
                             "encoder_attention_heads": 2, "encoder_ffn_dim": TD_ENC_FF,
                             "max_source_positions": TD_ENC_CTX, "scale_embedding": False}}


def make_transcribe_checkpoint(root: Path) -> Path:
    model_dir = root / "transcribe"
    model_dir.mkdir()
    vocab_size = 256 + len(TD_ADDED)
    write_safetensors(model_dir / "model.safetensors", transcribe_tensors(vocab_size))
    write_json(model_dir / "config.json", transcribe_config(vocab_size))
    write_json(model_dir / "preprocessor_config.json", {
        "sampling_rate": 16000, "n_fft": TD_FFT, "hop_length": 4, "feature_size": TD_MELS,
        "n_samples": 4 * 2 * TD_ENC_CTX, "nb_max_frames": 2 * TD_ENC_CTX})
    write_json(model_dir / "processor_config.json", {"audio_tokens_per_second": 12.5,
                                                     "time_marker_every_seconds": 5, "enable_time_marker": True})
    write_json(model_dir / "generation_config.json", {"max_new_tokens": 64})
    write_json(model_dir / "tokenizer.json", transcribe_tokenizer_json())
    return model_dir


def reader_string(path: Path, key: str) -> str:
    field = gguf.GGUFReader(str(path)).fields[key]
    return bytes(field.parts[field.data[0]]).decode("utf-8")


def check_transcribe(path: Path, model_dir: Path) -> None:
    from tokenizers import Tokenizer

    names = reader_tensors(path)
    expected = 2 + 11 + 7 + 15 + 6 + 1
    assert len(names) == expected, f"transcribe tensor census: {len(names)} != {expected}"
    assert names["enc.conv1.bias"] == (1, TD_ENC), "conv biases become columns"
    assert names["audio.mel_filters"] == (TD_FFT // 2 + 1, TD_MELS), "mel filters are stored bins x mels"
    conv = reader_tensor(path, "enc.conv1.weight")
    assert conv.tensor_type == gguf.GGMLQuantizationType.F16, "conv kernels stay F16 for im2col"
    assert "enc.blk.0.attn_k.bias" not in names, "the key projection has no bias"
    types = [int(value) for value in reader_field(path, "tokenizer.ggml.token_type")]
    first_added = 256
    assert types[0] == int(gguf.TokenType.NORMAL), "byte tokens are normal"
    assert types[first_added] == int(gguf.TokenType.CONTROL), "special added tokens are control"
    assert types[first_added + TD_ADDED.index("<think>")] == int(gguf.TokenType.USER_DEFINED)
    stored = [int(value) for value in reader_field(path, "moss-transcribe.default_prompt_ids")]
    reference = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompt = reader_string(path, "moss-transcribe.default_prompt")
    assert stored == reference.encode(prompt, add_special_tokens=False).ids, "prompt ids come from tokenizers"
    assert reader_string(path, "moss-transcribe.hotword_prefix") == "热词提示：", "hotword prefix follows upstream"
    assert reader_string(path, "moss-transcribe.hotword_separator") == ", "
    assert int(reader_field(path, "moss-transcribe.token.audio_pad")[0]) == first_added + 5
    print("transcribe converter: PASS")


def check_transcribe_quantized(path: Path) -> None:
    tensor = reader_tensor(path, "text.blk.0.attn_q.weight")
    assert tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0, "q8_0 quantizes decoder matrices"
    narrow = reader_tensor(path, "enc.blk.0.attn_q.weight")
    assert narrow.tensor_type == gguf.GGMLQuantizationType.F16, "rows shorter than a q8_0 block fall back to f16"
    assert reader_tensor(path, "enc.conv2.weight").tensor_type == gguf.GGMLQuantizationType.F16
    assert reader_tensor(path, "enc.pos_embd").tensor_type == gguf.GGMLQuantizationType.F32
    print("transcribe q8_0 converter: PASS")


def expect_transcribe_failure(model_dir: Path, out: Path, needle: str, label: str) -> None:
    result = subprocess.run([sys.executable, str(SCRIPTS / "convert-moss-transcribe-to-gguf.py"), str(model_dir),
                             "--outfile", str(out)], capture_output=True, text=True)
    assert result.returncode != 0, f"{label}: converter accepted the checkpoint"
    assert needle in result.stderr, f"{label}: wrong failure: {result.stderr[-400:]}"
    print(f"transcribe converter rejects {label}: PASS")


def run_transcribe_checks(root: Path) -> None:
    model_dir = make_transcribe_checkpoint(root)
    out = root / "transcribe.gguf"
    run_converter("convert-moss-transcribe-to-gguf.py", [str(model_dir), "--outtype", "f32", "--outfile", str(out)])
    check_transcribe(out, model_dir)
    quantized = root / "transcribe-q8.gguf"
    run_converter("convert-moss-transcribe-to-gguf.py",
                  [str(model_dir), "--outtype", "q8_0", "--outfile", str(quantized)])
    check_transcribe_quantized(quantized)
    tensors = transcribe_tensors(256 + len(TD_ADDED))
    tensors["model.whisper_encoder.layers.0.unknown.weight"] = np.ones(TD_ENC)
    write_safetensors(model_dir / "model.safetensors", tensors)
    expect_transcribe_failure(model_dir, root / "bad.gguf", "unmapped tensor", "an unmapped tensor")


def reader_tensor(path: Path, name: str):
    reader = gguf.GGUFReader(str(path))
    return next(t for t in reader.tensors if t.name == name)


def reader_field(path: Path, key: str) -> list:
    field = gguf.GGUFReader(str(path)).fields[key]
    return [field.parts[index][0] for index in field.data]


def run_converter(script: str, args: list[str]) -> None:
    result = subprocess.run([sys.executable, str(SCRIPTS / script), *args],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"{script} failed:\n{result.stdout}\n{result.stderr}")


def reader_tensors(path: Path) -> dict[str, tuple[int, ...]]:
    reader = gguf.GGUFReader(str(path))
    return {t.name: tuple(int(d) for d in t.shape) for t in reader.tensors}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_transcribe_checks(Path(tmp))
    print("moss transcribe converter: OK")


if __name__ == "__main__":
    main()
