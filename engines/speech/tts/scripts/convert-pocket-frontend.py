#!/usr/bin/env python3
"""Export native Pocket text metadata and a prepared voice from local assets."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import tempfile
import unicodedata
import gguf
import numpy as np
from safetensors import safe_open
from sentencepiece import sentencepiece_model_pb2 as pb
import yaml


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def tokenizer_data(tokenizer, config, weights_hash):
    model = pb.ModelProto()
    model.ParseFromString(tokenizer.read_bytes())
    t, n = model.trainer_spec, model.normalizer_spec
    if (t.model_type != 1 or not t.byte_fallback or n.name != "identity" or
            not n.add_dummy_prefix or n.remove_extra_whitespaces or not n.escape_whitespaces or
            n.precompiled_charsmap or t.treat_whitespace_as_suffix):
        raise ValueError("Native Pocket frontend requires identity unigram with byte fallback")
    if len(model.pieces) != config["flow_lm"]["lookup_table"]["n_bins"]:
        raise ValueError("Tokenizer vocabulary does not match checkpoint")
    if any(p.type not in (1, 2, 3, 6) or not np.isfinite(p.score) for p in model.pieces):
        raise ValueError("Unsupported tokenizer piece type or score")
    # Match the Python reference's Unicode casing and whitespace semantics
    # without depending on the host locale or another runtime library.
    upper, spaces, digits = {}, [], []
    for cp in range(0x110000):
        char = chr(cp)
        if char.upper() != char:
            upper[str(cp)] = char.upper()
        if char.isspace():
            spaces.append(cp)
        if char.isdigit():
            digits.append(cp)
    return dict(architecture="pocket-tts-frontend", schema_version=1,
                source_sha256=weights_hash, tokenizer_sha256=digest(tokenizer),
                unicode_version=unicodedata.unidata_version, uppercase=upper, whitespace=spaces, digits=digits,
                pieces=[dict(text=p.piece, score=p.score, type=p.type) for p in model.pieces],
                pad_short=config.get("pad_with_spaces_for_short_inputs", False),
                remove_semicolons=config.get("remove_semicolons", False),
                append_punctuation=config.get("append_terminal_punctuation", True),
                frames_after_eos=config.get("model_recommended_frames_after_eos"))


def voice_data(path, config):
    t = config["flow_lm"]["transformer"]
    data, used, length = {}, set(), None
    with safe_open(str(path), framework="pt") as f:
        for i in range(t["num_layers"]):
            prefix = f"transformer.layers.{i}.self_attn/"
            offset_key = prefix + "offset"
            old_key = prefix + "current_end"
            if offset_key in f.keys():
                offset = f.get_tensor(offset_key)
                if offset.dtype.is_floating_point or offset.numel() != 1:
                    raise ValueError("Invalid voice offset")
                end = int(offset.item()); used.add(offset_key)
            elif old_key in f.keys():
                end = f.get_tensor(old_key).shape[0]; used.add(old_key)
            else:
                raise ValueError("Missing voice offset")
            if not 0 < end < 8192 or (length is not None and end != length):
                raise ValueError("Inconsistent or unsupported voice length")
            length = end
            pad_key = prefix + "pad"
            if pad_key in f.keys():
                if f.get_tensor(pad_key).count_nonzero().item():
                    raise ValueError("Batched/padded voices are unsupported")
                used.add(pad_key)
            key = prefix + "cache"
            cache = f.get_tensor(key)
            expected = [2, 1, t["num_heads"], t["d_model"] // t["num_heads"]]
            if (cache.ndim != 5 or [cache.shape[j] for j in [0,1,3,4]] != expected or
                    cache.shape[2] < end or not cache.is_floating_point()):
                raise ValueError("Invalid voice cache shape/type")
            cache = cache[:, 0, :end].float().numpy()
            if not np.isfinite(cache).all():
                raise ValueError("Non-finite active voice cache")
            used.add(key)
            for kv, name in enumerate(("key", "value")):
                data[f"layer.{i}.{name}"] = np.ascontiguousarray(cache[kv])
        if used != set(f.keys()):
            raise ValueError("Unrecognized voice tensors")
    return length, data


def convert(weights, config_path, tokenizer, voice, output):
    config = yaml.safe_load(config_path.read_text())
    sha = digest(weights)
    frontend = tokenizer_data(tokenizer, config, sha)
    length, data = voice_data(voice, config)
    output.mkdir(parents=True, exist_ok=True)
    for dest in (output / "frontend.json", output / "voice.gguf"):
        if dest.resolve() in {p.resolve() for p in (weights, config_path, tokenizer, voice)}:
            raise ValueError("Output must not replace a source asset")
    # Stage both artifacts completely before publishing either. Keep backups
    # so a filesystem error during pair publication restores the previous pair.
    # Source hashes make a mixed-generation read fail closed if another process
    # opens the pair in the small interval between the two atomic renames.
    temporary, backups, published, retained = [], {}, [], set()
    def temp(prefix):
        fd, name = tempfile.mkstemp(prefix=prefix, dir=output)
        os.close(fd)
        path = Path(name)
        temporary.append(path)
        return path
    writer = None
    try:
        voice_temp = temp("voice.")
        frontend_temp = temp("frontend.")
        writer = gguf.GGUFWriter(str(voice_temp), "pocket-tts-voice")
        writer.add_uint32("pocket.schema_version", 1)
        writer.add_uint32("pocket.voice_frames", length)
        writer.add_string("pocket.source_sha256", sha)
        writer.add_string("pocket.voice_sha256", digest(voice))
        for name, value in data.items():
            writer.add_tensor(name, value)
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file()
        writer.close(); writer = None
        frontend_temp.write_text(json.dumps(frontend, ensure_ascii=False))
        pairs = [(voice_temp, output / "voice.gguf"), (frontend_temp, output / "frontend.json")]
        for _, dest in pairs:
            if dest.exists():
                backup = temp("backup.")
                shutil.copyfile(dest, backup)
                backups[dest] = backup
        try:
            for staged, dest in pairs:
                os.replace(staged, dest)
                published.append(dest)
        except BaseException as publication_error:
            recovery_errors = []
            for dest in reversed(published):
                try:
                    if dest in backups:
                        os.replace(backups[dest], dest)
                    else:
                        dest.unlink(missing_ok=True)
                except BaseException as recovery_error:
                    recovery_errors.append(str(recovery_error))
            if recovery_errors:
                retained.update(path for path in backups.values() if path.exists())
                raise RuntimeError(f"Publication failed: {publication_error}; restoration failed: "
                                   f"{recovery_errors}. Recoverable backups retained at "
                                   f"{sorted(str(path) for path in retained)}") from publication_error
            raise
    finally:
        if writer:
            writer.close()
        for path in temporary:
            if path not in retained:
                path.unlink(missing_ok=True)



if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("weights", "config", "tokenizer", "voice", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    a = p.parse_args()
    convert(a.weights, a.config, a.tokenizer, a.voice, a.output)
