#!/usr/bin/env python3
"""Convert ACE-Step safetensors checkpoints to the stage GGUFs the audiogen
engine loads (text encoder, 5 Hz LM, DiT, VAE).

Adapted from convert.py in acestep.cpp (github.com/ServeurpersoCom/acestep.cpp,
MIT), the implementation this engine's loaders were validated against, so the
emitted layout matches what src/acestep expects: one self-contained BF16 GGUF
per stage carrying weights, config metadata, the BPE tokenizer (LM and text
encoder), and the DiT's silence latent. Stage classification downstream is by
filename stem (see engine.cpp), which the `<checkpoint-name>-BF16.gguf` output
naming satisfies for every checkpoint download-acestep-checkpoints.sh fetches.

Quantize the BF16 files afterwards with the `acestep-quantize` binary (built
with AUDIOGEN_BUILD_EXECUTABLES); the VAE always stays BF16.

    scripts/download-acestep-checkpoints.sh --dir checkpoints
    python3 scripts/convert-acestep-to-gguf.py --checkpoints checkpoints --out models/bf16
"""

import argparse
import json
import os
import struct
import sys
import zipfile

DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_OUTPUT_DIR = "models/bf16"

SILENCE_LATENT_FILE = "silence_latent.pt"
SILENCE_LATENT_CHANNELS = 64
SILENCE_LATENT_FRAMES = 15000

ARCHS = {
    "lm": "acestep-lm",
    "dit": "acestep-dit",
    "text-enc": "acestep-text-enc",
    "vae": "acestep-vae",
}

DIT_UINT32_KEYS = (
    "in_channels", "audio_acoustic_hidden_dim", "patch_size",
    "sliding_window", "fsq_dim", "text_hidden_dim", "timbre_hidden_dim",
    "num_lyric_encoder_hidden_layers", "num_timbre_encoder_hidden_layers",
    "num_audio_decoder_hidden_layers", "num_attention_pooler_hidden_layers",
    "encoder_hidden_size", "encoder_intermediate_size",
    "encoder_num_attention_heads", "encoder_num_key_value_heads",
)


def log(msg):
    print("[GGUF] %s" % msg, file=sys.stderr, flush=True)


def read_sf_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))
    meta.pop("__metadata__", None)
    return meta, 8 + n


def find_sf_files(model_dir):
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, "r", encoding="utf-8") as f:
            idx = json.load(f)
        shards = sorted(set(idx["weight_map"].values()))
        return [os.path.join(model_dir, s) for s in shards]
    diffusers = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
    if os.path.exists(diffusers):
        return [diffusers]
    return []


def classify(name):
    if name.startswith("acestep-5Hz-lm"):
        return "lm"
    if name.startswith("acestep-v15"):
        return "dit"
    if name.startswith("Qwen3-Embedding"):
        return "text-enc"
    if name == "vae":
        return "vae"
    return None


def normalize_tensor_name(name, model_type):
    if model_type == "lm" and not name.startswith("model."):
        return "model." + name
    return name


def add_metadata(w, cfg, model_type):
    if "num_hidden_layers" in cfg:
        w.add_block_count(cfg["num_hidden_layers"])
    if "hidden_size" in cfg:
        w.add_embedding_length(cfg["hidden_size"])
    if "intermediate_size" in cfg:
        w.add_feed_forward_length(cfg["intermediate_size"])
    if "num_attention_heads" in cfg:
        w.add_head_count(cfg["num_attention_heads"])
    if "num_key_value_heads" in cfg:
        w.add_head_count_kv(cfg["num_key_value_heads"])
    if "head_dim" in cfg:
        w.add_key_length(cfg["head_dim"])
    if "vocab_size" in cfg:
        w.add_vocab_size(cfg["vocab_size"])
    if "max_position_embeddings" in cfg:
        w.add_context_length(cfg["max_position_embeddings"])
    if "rms_norm_eps" in cfg:
        w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    rope = cfg.get("rope_theta")
    if rope:
        w.add_rope_freq_base(float(rope))

    if model_type == "lm" and cfg.get("tie_word_embeddings"):
        w.add_bool("acestep.tie_word_embeddings", True)

    if model_type == "dit":
        add_dit_metadata(w, cfg)

    w.add_string("acestep.config_json", json.dumps(cfg, separators=(",", ":")))


def add_dit_metadata(w, cfg):
    for key in DIT_UINT32_KEYS:
        if key in cfg:
            w.add_uint32("acestep.%s" % key, cfg[key])
    if cfg.get("is_turbo"):
        w.add_bool("acestep.is_turbo", True)
    levels = cfg.get("fsq_input_levels")
    if levels:
        w.add_array("acestep.fsq_input_levels", levels)


def add_tensors_from_sf(w, sf_path, model_type):
    import gguf
    import numpy as np

    meta, hdr_size = read_sf_header(sf_path)
    names = sorted(meta.keys(), key=lambda n: meta[n]["data_offsets"][0])
    count = 0
    total = 0
    with open(sf_path, "rb") as f:
        for name in names:
            info = meta[name]
            out_name = normalize_tensor_name(name, model_type)
            dtype_str = info["dtype"]
            shape = info["shape"]
            off0, off1 = info["data_offsets"]
            nbytes = off1 - off0

            f.seek(hdr_size + off0)
            raw = f.read(nbytes)

            if dtype_str == "BF16":
                arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                w.add_tensor(out_name, arr, raw_dtype=gguf.GGMLQuantizationType.BF16)
            elif dtype_str == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
                w.add_tensor(out_name, arr)
            elif dtype_str == "F32":
                arr = f32_bits_to_bf16(np.frombuffer(raw, dtype=np.uint32).reshape(shape))
                w.add_tensor(out_name, arr, raw_dtype=gguf.GGMLQuantizationType.BF16)
                nbytes = nbytes // 2
            else:
                log("  skip %s: dtype %s" % (name, dtype_str))
                continue

            count += 1
            total += nbytes
    return count, total


def f32_bits_to_bf16(bits):
    return (bits >> 16).astype("uint16")


def read_silence_latent(model_dir):
    import numpy as np

    pt_path = os.path.join(model_dir, SILENCE_LATENT_FILE)
    if not os.path.exists(pt_path):
        return None
    with zipfile.ZipFile(pt_path) as z:
        for entry in z.namelist():
            if entry.endswith("/data/0"):
                raw = z.read(entry)
                src = np.frombuffer(raw, dtype=np.float32).reshape(
                    SILENCE_LATENT_CHANNELS, SILENCE_LATENT_FRAMES)
                return src.T.copy()
    return None


def read_bpe_merges(merges_path):
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line or line.startswith("#version:"):
                continue
            merges.append(line)
    return merges


def read_bpe_tokens(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    tokens = [""] * len(vocab)
    for tok_str, tok_id in vocab.items():
        if 0 <= tok_id < len(tokens):
            tokens[tok_id] = tok_str
    return tokens


def add_bpe_tokenizer(w, model_dir):
    vocab_path = os.path.join(model_dir, "vocab.json")
    merges_path = os.path.join(model_dir, "merges.txt")
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        return False

    tokens = read_bpe_tokens(vocab_path)
    merges = read_bpe_merges(merges_path)

    w.add_tokenizer_model("gpt2")
    w.add_token_list(tokens)
    w.add_token_merges(merges)

    log("  tokenizer: %d vocab, %d merges" % (len(tokens), len(merges)))
    return True


def convert_model(name, model_dir, output_path, model_type):
    import gguf

    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        log("skip %s: no config.json" % name)
        return False

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sf_files = find_sf_files(model_dir)
    if not sf_files:
        log("skip %s: no safetensors" % name)
        return False

    arch = ARCHS[model_type]
    log("%s (%s, %d shard%s) -> %s" % (
        name, arch, len(sf_files), "" if len(sf_files) == 1 else "s",
        os.path.basename(output_path)))

    if model_type in ("lm", "text-enc"):
        vocab_path = os.path.join(model_dir, "vocab.json")
        merges_path = os.path.join(model_dir, "merges.txt")
        if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
            log("ERROR: vocab.json and merges.txt required for %s" % name)
            return False

    if model_type == "dit" and read_silence_latent(model_dir) is None:
        log("ERROR: %s required for DiT conversion" % SILENCE_LATENT_FILE)
        return False

    w = gguf.GGUFWriter(output_path, arch, use_temp_file=True)
    w.add_name(name)
    add_metadata(w, cfg, model_type)

    if model_type in ("lm", "text-enc"):
        add_bpe_tokenizer(w, model_dir)

    n_tensors = 0
    n_bytes = 0
    for sf in sf_files:
        c, b = add_tensors_from_sf(w, sf, model_type)
        n_tensors += c
        n_bytes += b
        if len(sf_files) > 1:
            log("  %s: %d tensors" % (os.path.basename(sf), c))

    if model_type == "dit":
        n_tensors, n_bytes = add_silence_latent(w, model_dir, n_tensors, n_bytes)

    log("  total: %d tensors, %.1f GB" % (n_tensors, n_bytes / (1 << 30)))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()

    out_mb = os.path.getsize(output_path) / (1 << 20)
    log("  wrote %.0f MB -> %s" % (out_mb, output_path))
    return True


def add_silence_latent(w, model_dir, n_tensors, n_bytes):
    sl = read_silence_latent(model_dir)
    w.add_tensor("silence_latent", sl)
    log("  silence_latent: [%d, %d] f32 (%.1f MB)" % (
        sl.shape[0], sl.shape[1], sl.nbytes / (1 << 20)))
    return n_tensors + 1, n_bytes + sl.nbytes


def select_checkpoints(checkpoint_dir, only):
    selected = []
    skipped = []
    for name in sorted(os.listdir(checkpoint_dir)):
        model_dir = os.path.join(checkpoint_dir, name)
        if not os.path.isdir(model_dir):
            continue
        if only and name not in only:
            continue
        model_type = classify(name)
        if model_type is None:
            skipped.append(name)
            continue
        selected.append((name, model_dir, model_type))
    return selected, skipped


def convert_all(selected, output_dir):
    converted = 0
    failed = 0
    for name, model_dir, model_type in selected:
        output_path = os.path.join(output_dir, "%s-BF16.gguf" % name)
        if os.path.exists(output_path):
            log("skip %s: %s exists" % (name, os.path.basename(output_path)))
            converted += 1
            continue
        if convert_model(name, model_dir, output_path, model_type):
            converted += 1
        else:
            failed += 1
    return converted, failed


def parse_args(argv):
    ap = argparse.ArgumentParser(
        description="ACE-Step safetensors -> stage GGUFs for the audiogen engine.")
    ap.add_argument("--checkpoints", default=DEFAULT_CHECKPOINT_DIR, metavar="DIR",
                    help="directory populated by download-acestep-checkpoints.sh "
                         "(default: %(default)s)")
    ap.add_argument("--out", default=DEFAULT_OUTPUT_DIR, metavar="DIR",
                    help="output directory for the BF16 GGUFs (default: %(default)s)")
    ap.add_argument("--only", action="append", metavar="NAME",
                    help="convert only this checkpoint directory (repeatable)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not os.path.isdir(args.checkpoints):
        log("%s not found, run download-acestep-checkpoints.sh first" % args.checkpoints)
        return 1

    os.makedirs(args.out, exist_ok=True)

    selected, skipped = select_checkpoints(args.checkpoints, args.only)
    if not selected:
        log("no recognized checkpoints in %s" % args.checkpoints)
        return 1

    converted, failed = convert_all(selected, args.out)

    if skipped:
        log("skipped (unknown): %s" % ", ".join(skipped))
    log("done: %d model(s) in %s" % (converted, args.out))
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
