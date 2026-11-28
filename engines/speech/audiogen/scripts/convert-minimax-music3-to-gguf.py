#!/usr/bin/env python3

import argparse
import json
import mmap
import os
import re
import struct
import sys

import numpy as np
import gguf

CONVERTER_VERSION = 1

LM_LAYERS = 36
LM_DIM = 4096
LM_FF = 12288
LM_HEADS = 32
LM_KV_HEADS = 8
LM_HEAD_DIM = 128
LM_VOCAB = 200000
LM_CTX = 10240
LM_RMS_EPS = 1e-6
LM_ROPE_THETA = 1000000.0

AUDIO_CODE_OFFSET = 151675
SEMANTIC_VOCAB = 16384
ACOUSTIC_VOCAB = 1024
NUM_CODEBOOKS = 8
FRAME_RATE = 25
MAX_AUDIO_FRAMES = 9000
MAX_PROMPT_TOKENS = 5000
AR_CFG_SCALE = 1.5
AR_TOP_K = 50
SPECIAL_TOKEN_IDS = {
    "im_start": 151644,
    "im_end": 151645,
    "audio_cfg": 151654,
    "audio_start": 151669,
    "audio_end": 151670,
    "caption_start": 151671,
    "caption_end": 151672,
    "lyrics_start": 151673,
    "lyrics_end": 151674,
}

DEPTH_LAYERS = 4
DEPTH_DIM = 4096
DEPTH_FF = 6144
DEPTH_HEADS = 16
DEPTH_HEAD_DIM = DEPTH_DIM // DEPTH_HEADS
DEPTH_MAX_POS = 16
DEPTH_RMS_EPS = 1e-6
DEPTH_AUDIO_EMBD_ROWS = ACOUSTIC_VOCAB * (NUM_CODEBOOKS - 1)

COND_LAYERS = 8
COND_HIDDEN = 4096
COND_OUT = 2048
COND_KERNEL = 3
COND_PADDING = 1
COND_IN_SR = 24000
COND_IN_HOP = 960
COND_OUT_SR = 44100
COND_OUT_HOP = 512

DIT_LAYERS = 36
DIT_DIM = 2048
DIT_HEADS = 32
DIT_HEAD_DIM = 64
DIT_FF_INNER = 8192
DIT_IN_CH = 128
DIT_COND_DIM = 2048
DIT_CONCAT_CH = DIT_IN_CH * 2 + DIT_COND_DIM
DIT_ROPE_DIM = 32
DIT_ROPE_THETA = 10000.0
DIT_FOURIER_DIM = 256
DIT_LN_EPS = 1e-5
DIT_WINDOW_FRAMES = 200
DIT_HOP_FRAMES = 100
DIT_STEPS = 20
DIT_CFG_SCALE = 1.7

VOC_LATENT_CH = 128
VOC_FOLD_CH = 64
VOC_DEC_IN_DIM = 1024
VOC_HIDDEN = 1536
VOC_UPSAMPLE = (8, 8, 4, 2)
VOC_RES_DILATIONS = (1, 3, 9)
VOC_SR = 44100
VOC_SNAKE_EPS = 1e-9

def latent_length(frames):
    return max(1, int(frames * COND_OUT_SR / COND_IN_SR * COND_IN_HOP / COND_OUT_HOP))

def log(msg):
    print(f"[convert-minimax-music3] {msg}", file=sys.stderr, flush=True)

def die(msg):
    raise SystemExit(f"[convert-minimax-music3] ERROR: {msg}")

def require_tokenizer(tokenizer):
    if tokenizer is None:
        die("tokenizer metadata is required; pass --tokenizer or include tokenizer.json in --src")
    return tokenizer

def temporary_output_path(final_path, transaction_id, index):
    return f"{final_path}.tmp-{transaction_id}-{index}"

def backup_output_path(final_path, transaction_id, index):
    return f"{final_path}.backup-{transaction_id}-{index}"

def remove_paths(paths):
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

def backup_restore_error(failures):
    details = "; ".join(
        f"backup={backup!r}, destination={destination!r}, "
        f"error={type(error).__name__}: {error}; original remains at backup"
        for backup, destination, error in failures
    )
    return f"output rollback failed: {details}"

class OutputTransaction:
    def __init__(self, transaction_id=None):
        self.transaction_id = transaction_id or f"{os.getpid()}-{id(self):x}"
        self.entries = []
        self.backups = []
        self.restored_backups = set()
        self.committed = []
        self.finished = False

    def write(self, bundle, final_path, quant, role):
        index = len(self.entries)
        temporary = temporary_output_path(final_path, self.transaction_id, index)
        remove_paths([temporary])
        self.entries.append((temporary, final_path, role))
        validate_bundle(bundle, role)
        bundle.write(temporary, quant)

    def create_backups(self):
        for index, (_, final_path, _) in enumerate(self.entries):
            if not os.path.exists(final_path):
                continue
            backup = backup_output_path(final_path, self.transaction_id, index)
            remove_paths([backup])
            os.replace(final_path, backup)
            self.backups.append((backup, final_path))

    def publish(self):
        for temporary, final_path, _ in self.entries:
            os.replace(temporary, final_path)
            self.committed.append(final_path)

    def rollback(self):
        remove_paths(self.committed)
        failures = self.restore_backups()
        if failures:
            raise RuntimeError(backup_restore_error(failures))

    def restore_backups(self):
        failures = []
        for backup, final_path in reversed(self.backups):
            try:
                os.replace(backup, final_path)
                self.restored_backups.add(backup)
            except BaseException as error:
                failures.append((backup, final_path, error))
        return failures

    def commit(self):
        try:
            self.create_backups()
            self.publish()
        except BaseException as publish_error:
            try:
                self.rollback()
            except BaseException as rollback_error:
                self.cleanup()
                raise rollback_error from publish_error
            self.cleanup()
            raise
        remove_paths([backup for backup, _ in self.backups])
        self.finished = True

    def cleanup(self):
        if not self.finished:
            remove_paths([temporary for temporary, _, _ in self.entries])

_ST_DTYPES = {"F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "BOOL"}

class SafeTensorsFile:
    def __init__(self, path):
        self.path = path
        self._fh = open(path, "rb")
        n = struct.unpack("<Q", self._fh.read(8))[0]
        self.header = json.loads(self._fh.read(n).decode("utf-8"))
        self.metadata = self.header.pop("__metadata__", {})
        self._base = 8 + n
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        for name, spec in self.header.items():
            if spec["dtype"] not in _ST_DTYPES:
                die(f"{path}: tensor {name} has unsupported dtype {spec['dtype']}")

    def keys(self):
        return self.header.keys()

    def shape(self, name):
        return tuple(self.header[name]["shape"])

    def dtype(self, name):
        return self.header[name]["dtype"]

    def raw(self, name):
        spec = self.header[name]
        start, end = spec["data_offsets"]
        return self._mm[self._base + start:self._base + end]

    def get(self, name):
        spec = self.header[name]
        buf = self.raw(name)
        dt = spec["dtype"]
        shape = tuple(spec["shape"])
        if dt == "BF16":
            u16 = np.frombuffer(buf, dtype=np.uint16)
            return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        np_dt = {
            "F64": np.float64, "F32": np.float32, "F16": np.float16,
            "I64": np.int64, "I32": np.int32, "I16": np.int16,
            "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_,
        }[dt]
        arr = np.frombuffer(buf, dtype=np_dt).reshape(shape)
        if np_dt in (np.float64, np.float32, np.float16):
            return arr.astype(np.float32)
        return arr

    def close(self):
        try:
            self._mm.close()
        finally:
            self._fh.close()

class TorchFile:
    def __init__(self, path):
        try:
            import torch
        except ImportError:
            die(f"{path} is a .pth checkpoint; install torch or use the "
                f"safetensors sources instead")
        self.path = path
        sd = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
        if not isinstance(sd, dict):
            die(f"{path}: not a state dict")
        self._sd = sd

    def keys(self):
        return self._sd.keys()

    def shape(self, name):
        return tuple(self._sd[name].shape)

    def dtype(self, name):
        return str(self._sd[name].dtype).replace("torch.", "").upper()

    def get(self, name):
        import torch
        t = self._sd[name].detach()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.to(torch.float32).contiguous().numpy()

    def close(self):
        self._sd = {}

REFUSE_PATTERNS = [
    (re.compile(r"int8|convrot|fp8|nf4", re.I),
     "packed int8/convrot/fp8 weights cannot be dequantized by this converter; "
     "use the fp16 or fp32 variant"),
    (re.compile(r"pruned", re.I),
     "the '_pruned_' repack fuses qkv/gate_up and splits the embedding into "
     "embed_tokens_prefill/embed_tokens_audio + lm_head_pruned; it is a "
     "different model surface. Use the plain bf16 text encoder"),
]

def _file_score(path):
    b = os.path.basename(path).lower()
    score = 0
    if "fp32" in b or "f32" in b:
        score += 30
    elif "bf16" in b:
        score += 20
    elif "fp16" in b or "f16" in b:
        score += 10
    if path.lower().endswith(".pth"):
        score -= 5
    return score

class Source:
    def __init__(self, paths, verbose=False):
        self.files = []
        self.index = {}
        self.consumed = set()
        self.verbose = verbose
        found = []
        for p in paths:
            p = os.path.abspath(p)
            if os.path.isdir(p):
                for root, _dirs, names in os.walk(p):
                    for n in sorted(names):
                        if n.endswith(".safetensors") or n.endswith(".pth"):
                            found.append(os.path.join(root, n))
            elif os.path.isfile(p):
                found.append(p)
            else:
                die(f"--src path does not exist: {p}")
        if not found:
            die(f"no .safetensors or .pth files found under: {', '.join(paths)}")

        keep = []
        for path in found:
            base = os.path.basename(path)
            refused = None
            for pat, why in REFUSE_PATTERNS:
                if pat.search(base):
                    refused = why
                    break
            if refused:
                log(f"skipping {base}: {refused}")
                continue
            keep.append(path)
        if not keep:
            die("every candidate file was refused; see the messages above")

        for path in sorted(keep, key=_file_score):
            f = SafeTensorsFile(path) if path.endswith(".safetensors") else TorchFile(path)
            self.files.append(f)
            n_new = n_over = 0
            for name in f.keys():
                if name in self.index:
                    n_over += 1
                else:
                    n_new += 1
                self.index[name] = f
            note = f", {n_over} overriding earlier files" if n_over else ""
            log(f"indexed {os.path.relpath(path)}: {n_new + n_over} tensors{note}")

        self.layout = self._detect_layout()
        log(f"source layout: {self.layout}")

    def _detect_layout(self):
        if "diffusion_transformer.preprocess_conv.weight" in self.index:
            return "comfy"
        if "preprocess_conv.weight" in self.index or "transformer_blocks.0.attn.to_q.weight" in self.index:
            return "diffusers"
        if "model.audio_decoder.norm.weight" in self.index:
            return "comfy"
        if "audio_embeddings.weight" in self.index:
            return "diffusers"
        return "unknown"

    def has(self, *names):
        return any(n in self.index for n in names)

    def resolve(self, *names):
        for n in names:
            if n in self.index:
                return n
        return None

    def get(self, *names, expect=None, label=None):
        name = self.resolve(*names)
        if name is None:
            die(f"missing tensor {label or names[0]}; tried: {', '.join(names)}")
        arr = self.index[name].get(name)
        self.consumed.add(name)
        if expect is not None and tuple(arr.shape) != tuple(expect):
            die(f"{name}: expected shape {tuple(expect)}, checkpoint has "
                f"{tuple(arr.shape)} -- this is not a MiniMax-Music3 checkpoint "
                f"this converter understands")
        return np.ascontiguousarray(arr, dtype=np.float32)

    def get_bytes(self, name):
        if name not in self.index:
            return None
        f = self.index[name]
        self.consumed.add(name)
        if isinstance(f, SafeTensorsFile):
            return bytes(f.raw(name))
        return bytes(f.get(name).astype(np.uint8).tobytes())

    def unconsumed(self, predicate):
        return sorted(n for n in self.index if predicate(n) and n not in self.consumed)

    def close(self):
        for f in self.files:
            f.close()

F32 = "f32"
HALF = "half"
QUANT = "quant"

F16_MAX = 65504.0

class Bundle:
    def __init__(self, arch):
        self.arch = arch
        self.kv = []
        self.tensors = []

    def meta(self, fn, *args):
        self.kv.append((fn, args))

    def put(self, name, arr, policy):
        if name in {n for n, _, _ in self.tensors}:
            die(f"duplicate output tensor {name}")
        self.tensors.append((name, np.ascontiguousarray(arr, dtype=np.float32), policy))

    def write(self, path, quant):
        w = gguf.GGUFWriter(path, self.arch)
        try:
            for fn, args in self.kv:
                getattr(w, fn)(*args)
            n = {"f32": 0, "f16": 0, "q8_0": 0}
            for name, arr, policy in self.tensors:
                ne0 = arr.shape[-1] if arr.ndim else 1
                if policy == F32 or arr.ndim < 2:
                    w.add_tensor(name, arr)
                    n["f32"] += 1
                    continue
                peak = float(np.abs(arr).max()) if arr.size else 0.0
                if peak > F16_MAX:
                    log(f"  {name}: |w|max={peak:.1f} exceeds f16 range, storing F32")
                    w.add_tensor(name, arr)
                    n["f32"] += 1
                    continue
                if quant == "q8_0" and policy == QUANT and ne0 % 32 == 0:
                    w.add_tensor(name, gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0),
                                 raw_dtype=gguf.GGMLQuantizationType.Q8_0)
                    n["q8_0"] += 1
                else:
                    w.add_tensor(name, arr.astype(np.float16).view(np.uint16),
                                 raw_shape=arr.shape,
                                 raw_dtype=gguf.GGMLQuantizationType.F16)
                    n["f16"] += 1
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
        finally:
            w.close()
        size = os.path.getsize(path) / 1e9
        log(f"wrote {os.path.basename(path)}: {len(self.tensors)} tensors "
            f"({n['f32']} F32, {n['f16']} F16, {n['q8_0']} Q8_0), {size:.2f} GB")

def bundle_custom_keys(bundle):
    return {args[0] for fn, args in bundle.kv
            if fn in {"add_string", "add_uint32", "add_float32", "add_array"} and args}

def validate_bundle(bundle, role):
    if not bundle.tensors:
        die(f"{role} output has no tensors")
    functions = {fn for fn, _ in bundle.kv}
    custom = bundle_custom_keys(bundle)
    required_functions = {"add_license", "add_file_type", "add_quantization_version"}
    required_custom = {"mm3.model", "mm3.converter_version", "mm3.source_layout"}
    if not required_functions.issubset(functions) or not required_custom.issubset(custom):
        die(f"{role} output is missing required common metadata")
    if role == "lm":
        tokenizer_functions = {
            "add_tokenizer_model", "add_tokenizer_pre", "add_token_list",
            "add_token_types", "add_token_merges",
        }
        if not tokenizer_functions.issubset(functions):
            die("LM output is missing tokenizer metadata")
    if role == "synth" and "mm3.synth.components" not in custom:
        die("synth output is missing component metadata")

def fold_weight_norm(g, v, label):
    if g.shape[0] != v.shape[0]:
        die(f"{label}: weight_g first dim {g.shape[0]} != weight_v {v.shape[0]}")
    axes = tuple(range(1, v.ndim))
    norm = np.sqrt(np.sum(v.astype(np.float64) ** 2, axis=axes, keepdims=True))
    if not np.all(norm > 0):
        die(f"{label}: weight_v has a zero-norm slice; refusing to fold")
    g = g.astype(np.float64).reshape((g.shape[0],) + (1,) * (v.ndim - 1))
    return (g * v.astype(np.float64) / norm).astype(np.float32)

def build_lm(src, bundle, tokenizer, embed_tokenizer_json):
    tokenizer = require_tokenizer(tokenizer)
    if src.has("model.embed_tokens_prefill.weight", "model.lm_head_pruned.weight",
               "model.layers.0.self_attn.qkv_proj.weight"):
        die("this text encoder is the '_pruned_' repack (fused qkv / split "
            "embeddings). Convert from the plain bf16 text encoder or from "
            "MiniMaxAI/MiniMax-Music3 language_model/ instead")

    b = bundle
    b.meta("add_name", "MiniMax-Music3 LM")
    b.meta("add_description",
           "MiniMax-Music3 global LM: Qwen3-8B with an extended audio vocabulary")
    b.meta("add_context_length", LM_CTX)
    b.meta("add_embedding_length", LM_DIM)
    b.meta("add_block_count", LM_LAYERS)
    b.meta("add_feed_forward_length", LM_FF)
    b.meta("add_head_count", LM_HEADS)
    b.meta("add_head_count_kv", LM_KV_HEADS)
    b.meta("add_key_length", LM_HEAD_DIM)
    b.meta("add_value_length", LM_HEAD_DIM)
    b.meta("add_layer_norm_rms_eps", LM_RMS_EPS)
    b.meta("add_rope_freq_base", LM_ROPE_THETA)
    b.meta("add_vocab_size", LM_VOCAB)

    b.meta("add_uint32", "mm3.semantic_vocab_offset", AUDIO_CODE_OFFSET)
    b.meta("add_uint32", "mm3.semantic_vocab_size", SEMANTIC_VOCAB)
    b.meta("add_uint32", "mm3.acoustic_vocab_size", ACOUSTIC_VOCAB)
    b.meta("add_uint32", "mm3.num_codebooks", NUM_CODEBOOKS)
    b.meta("add_uint32", "mm3.eos_audio", SPECIAL_TOKEN_IDS["audio_end"])
    b.meta("add_uint32", "mm3.frame_rate", FRAME_RATE)
    b.meta("add_uint32", "mm3.max_audio_frames", MAX_AUDIO_FRAMES)
    b.meta("add_uint32", "mm3.max_prompt_tokens", MAX_PROMPT_TOKENS)
    b.meta("add_float32", "mm3.ar.cfg_scale", AR_CFG_SCALE)
    b.meta("add_uint32", "mm3.ar.top_k", AR_TOP_K)
    b.meta("add_float32", "mm3.ar.embedding_scale", float(NUM_CODEBOOKS) ** -0.5)
    for k, v in SPECIAL_TOKEN_IDS.items():
        b.meta("add_uint32", f"mm3.token.{k}", v)

    add_tokenizer_kv(b, tokenizer, embed_tokenizer_json)

    b.put("token_embd.weight",
          src.get("model.embed_tokens.weight", expect=(LM_VOCAB, LM_DIM)), QUANT)
    b.put("output_norm.weight",
          src.get("model.norm.weight", expect=(LM_DIM,)), F32)
    b.put("output.weight",
          src.get("model.lm_head.weight", "lm_head.weight",
                  expect=(LM_VOCAB, LM_DIM)), QUANT)

    q_dim = LM_HEADS * LM_HEAD_DIM
    kv_dim = LM_KV_HEADS * LM_HEAD_DIM
    for i in range(LM_LAYERS):
        p = f"model.layers.{i}"
        d = f"blk.{i}"
        b.put(f"{d}.attn_norm.weight",
              src.get(f"{p}.input_layernorm.weight", expect=(LM_DIM,)), F32)
        b.put(f"{d}.attn_q.weight",
              src.get(f"{p}.self_attn.q_proj.weight", expect=(q_dim, LM_DIM)), QUANT)
        b.put(f"{d}.attn_k.weight",
              src.get(f"{p}.self_attn.k_proj.weight", expect=(kv_dim, LM_DIM)), QUANT)
        b.put(f"{d}.attn_v.weight",
              src.get(f"{p}.self_attn.v_proj.weight", expect=(kv_dim, LM_DIM)), QUANT)
        b.put(f"{d}.attn_output.weight",
              src.get(f"{p}.self_attn.o_proj.weight", expect=(LM_DIM, q_dim)), QUANT)
        b.put(f"{d}.attn_q_norm.weight",
              src.get(f"{p}.self_attn.q_norm.weight", expect=(LM_HEAD_DIM,)), F32)
        b.put(f"{d}.attn_k_norm.weight",
              src.get(f"{p}.self_attn.k_norm.weight", expect=(LM_HEAD_DIM,)), F32)
        b.put(f"{d}.ffn_norm.weight",
              src.get(f"{p}.post_attention_layernorm.weight", expect=(LM_DIM,)), F32)
        b.put(f"{d}.ffn_gate.weight",
              src.get(f"{p}.mlp.gate_proj.weight", expect=(LM_FF, LM_DIM)), QUANT)
        b.put(f"{d}.ffn_up.weight",
              src.get(f"{p}.mlp.up_proj.weight", expect=(LM_FF, LM_DIM)), QUANT)
        b.put(f"{d}.ffn_down.weight",
              src.get(f"{p}.mlp.down_proj.weight", expect=(LM_DIM, LM_FF)), QUANT)

    def belongs(n):
        return (n.startswith("model.layers.")
                or n in ("model.embed_tokens.weight", "model.norm.weight",
                         "model.lm_head.weight", "lm_head.weight"))
    return belongs

def add_tokenizer_kv(b, tok, embed_raw):
    model = tok.get("model") or {}
    if model.get("type") != "BPE":
        die(f"tokenizer.json model.type is {model.get('type')!r}, expected 'BPE'")
    vocab = model.get("vocab") or {}
    added = tok.get("added_tokens") or []

    reverse = {i: s for s, i in vocab.items()}
    special_ids, user_ids = set(), set()
    for a in added:
        reverse[a["id"]] = a["content"]
        (special_ids if a.get("special") else user_ids).add(a["id"])

    tokens, toktypes = [], []
    n_pad = n_code = 0
    for i in range(LM_VOCAB):
        if i in reverse:
            tokens.append(reverse[i])
            if i in special_ids:
                toktypes.append(gguf.TokenType.CONTROL)
            elif i in user_ids:
                toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                toktypes.append(gguf.TokenType.NORMAL)
        elif AUDIO_CODE_OFFSET <= i < AUDIO_CODE_OFFSET + SEMANTIC_VOCAB:

            tokens.append(f"<|audio_code_{i - AUDIO_CODE_OFFSET}|>")
            toktypes.append(gguf.TokenType.USER_DEFINED)
            n_code += 1
        else:
            tokens.append(f"[PAD{i}]")
            toktypes.append(gguf.TokenType.UNUSED)
            n_pad += 1

    merges = []
    for m in model.get("merges") or []:
        merges.append(" ".join(m) if isinstance(m, (list, tuple)) else m)

    b.meta("add_tokenizer_model", "gpt2")
    b.meta("add_tokenizer_pre", "qwen2")
    b.meta("add_token_list", tokens)
    b.meta("add_token_types", toktypes)
    b.meta("add_token_merges", merges)
    b.meta("add_eos_token_id", SPECIAL_TOKEN_IDS["im_end"])
    b.meta("add_pad_token_id", 151643)
    b.meta("add_add_bos_token", False)
    if embed_raw is not None:
        b.meta("add_string", "mm3.tokenizer_json", embed_raw)
    log(f"tokenizer: {len(tokens)} tokens ({len(vocab)} BPE + {len(added)} added "
        f"+ {n_code} audio codes + {n_pad} unused), {len(merges)} merges")

def find_tokenizer(src, explicit, search_paths):
    if explicit:
        with open(explicit, "r", encoding="utf-8") as f:
            raw = f.read()
        return json.loads(raw), raw

    blob = src.get_bytes("tokenizer_json")
    if blob:
        raw = blob.decode("utf-8")
        log("tokenizer: taken from the 'tokenizer_json' tensor in the source")
        return json.loads(raw), raw
    for p in search_paths:
        p = os.path.abspath(p)
        roots = [p] if os.path.isdir(p) else [os.path.dirname(p), os.path.dirname(os.path.dirname(p))]
        for root in roots:
            for cand in ("tokenizer.json",
                         os.path.join("tokenizer", "tokenizer.json"),
                         os.path.join("qwen_7B", "qwen3-8B-tokenizer-music", "tokenizer.json")):
                f = os.path.join(root, cand)
                if os.path.isfile(f):
                    with open(f, "r", encoding="utf-8") as fh:
                        raw = fh.read()
                    log(f"tokenizer: {os.path.relpath(f)}")
                    return json.loads(raw), raw
    return None, None

def build_depth(src, b):
    b.meta("add_uint32", "mm3.depth.block_count", DEPTH_LAYERS)
    b.meta("add_uint32", "mm3.depth.embedding_length", DEPTH_DIM)
    b.meta("add_uint32", "mm3.depth.feed_forward_length", DEPTH_FF)
    b.meta("add_uint32", "mm3.depth.head_count", DEPTH_HEADS)
    b.meta("add_uint32", "mm3.depth.head_dim", DEPTH_HEAD_DIM)
    b.meta("add_uint32", "mm3.depth.max_position", DEPTH_MAX_POS)
    b.meta("add_float32", "mm3.depth.rms_eps", DEPTH_RMS_EPS)
    b.meta("add_uint32", "mm3.depth.num_codebooks", NUM_CODEBOOKS)
    b.meta("add_uint32", "mm3.depth.audio_vocab_size", ACOUSTIC_VOCAB)
    b.meta("add_uint32", "mm3.depth.audio_embd_rows", DEPTH_AUDIO_EMBD_ROWS)
    b.meta("add_bool", "mm3.depth.causal", True)
    b.meta("add_bool", "mm3.depth.rope", False)

    P = ["model.audio_decoder.", ""]

    def a(suffix):
        return tuple(p + suffix for p in P)

    b.put("depth.proj.weight",
          src.get(*a("projection.weight"), expect=(DEPTH_DIM, DEPTH_DIM)), QUANT)
    b.put("depth.pos_embd.weight",
          src.get(*a("pos_embedding.weight"), expect=(DEPTH_MAX_POS, DEPTH_DIM)), F32)
    b.put("depth.output_norm.weight",
          src.get(*a("norm.weight"), expect=(DEPTH_DIM,)), F32)
    b.put("depth.audio_embd.weight",
          src.get("model.audio_extra_embedding.weight", "audio_embeddings.weight",
                  expect=(DEPTH_AUDIO_EMBD_ROWS, DEPTH_DIM)), QUANT)
    for c in range(NUM_CODEBOOKS - 1):
        b.put(f"depth.head.{c}.weight",
              src.get(*a(f"audio_heads.{c}.weight"),
                      expect=(ACOUSTIC_VOCAB, DEPTH_DIM)), QUANT)

    for i in range(DEPTH_LAYERS):
        d = f"depth.blk.{i}"
        b.put(f"{d}.attn_norm.weight",
              src.get(*a(f"layers.{i}.input_layernorm.weight"), expect=(DEPTH_DIM,)), F32)
        for tag, comfy, diff in (("attn_q", "self_attn.q_proj", "attn.to_q"),
                                 ("attn_k", "self_attn.k_proj", "attn.to_k"),
                                 ("attn_v", "self_attn.v_proj", "attn.to_v"),
                                 ("attn_output", "self_attn.o_proj", "attn.to_out")):
            b.put(f"{d}.{tag}.weight",
                  src.get(f"model.audio_decoder.layers.{i}.{comfy}.weight",
                          f"layers.{i}.{diff}.weight",
                          expect=(DEPTH_DIM, DEPTH_DIM)), QUANT)
        b.put(f"{d}.ffn_norm.weight",
              src.get(*a(f"layers.{i}.post_attention_layernorm.weight"),
                      expect=(DEPTH_DIM,)), F32)
        for tag, comfy, diff, shape in (
                ("ffn_gate", "mlp.gate_proj", "gate_proj", (DEPTH_FF, DEPTH_DIM)),
                ("ffn_up", "mlp.up_proj", "up_proj", (DEPTH_FF, DEPTH_DIM)),
                ("ffn_down", "mlp.down_proj", "down_proj", (DEPTH_DIM, DEPTH_FF))):
            b.put(f"{d}.{tag}.weight",
                  src.get(f"model.audio_decoder.layers.{i}.{comfy}.weight",
                          f"layers.{i}.{diff}.weight", expect=shape), QUANT)

    def belongs(n):
        return (n.startswith("model.audio_decoder.")
                or n == "model.audio_extra_embedding.weight"
                or n == "audio_embeddings.weight"
                or (re.match(r"layers\.\d+\.", n) and not n.startswith("model."))
                or n in ("projection.weight", "pos_embedding.weight", "norm.weight")
                or n.startswith("audio_heads."))
    return belongs

def build_cond(src, b):
    b.meta("add_uint32", "mm3.cond.num_layers", COND_LAYERS)
    b.meta("add_uint32", "mm3.cond.hidden_dim", COND_HIDDEN)
    b.meta("add_uint32", "mm3.cond.out_dim", COND_OUT)
    b.meta("add_uint32", "mm3.cond.kernel_size", COND_KERNEL)
    b.meta("add_uint32", "mm3.cond.padding", COND_PADDING)
    b.meta("add_uint32", "mm3.cond.input_sampling_rate", COND_IN_SR)
    b.meta("add_uint32", "mm3.cond.input_hop_length", COND_IN_HOP)
    b.meta("add_uint32", "mm3.cond.output_sampling_rate", COND_OUT_SR)
    b.meta("add_uint32", "mm3.cond.output_hop_length", COND_OUT_HOP)
    b.meta("add_string", "mm3.cond.interpolation", "nearest")
    b.meta("add_string", "mm3.cond.layer_mix", "softmax")

    b.put("cond.layer_logits",
          src.get("cond_layer_logits", "layer_weight_logits", expect=(COND_LAYERS,)), F32)
    b.put("cond.layer_scale",
          src.get("cond_layer_scale", "layer_scale", expect=(1,)), F32)
    b.put("cond.proj.weight",
          src.get("latent_conditioners.0.weight", "proj.weight",
                  expect=(COND_OUT, COND_HIDDEN, COND_KERNEL)), HALF)
    b.put("cond.proj.bias",
          src.get("latent_conditioners.0.bias", "proj.bias", expect=(COND_OUT,)), F32)

    def belongs(n):
        return n in ("cond_layer_logits", "cond_layer_scale", "layer_weight_logits",
                     "layer_scale", "proj.weight", "proj.bias") \
            or n.startswith("latent_conditioners.")
    return belongs

def build_dit(src, b):
    b.meta("add_uint32", "mm3.dit.block_count", DIT_LAYERS)
    b.meta("add_uint32", "mm3.dit.embedding_length", DIT_DIM)
    b.meta("add_uint32", "mm3.dit.head_count", DIT_HEADS)
    b.meta("add_uint32", "mm3.dit.head_dim", DIT_HEAD_DIM)
    b.meta("add_uint32", "mm3.dit.ff_inner", DIT_FF_INNER)
    b.meta("add_uint32", "mm3.dit.in_channels", DIT_IN_CH)
    b.meta("add_uint32", "mm3.dit.condition_dim", DIT_COND_DIM)
    b.meta("add_uint32", "mm3.dit.concat_channels", DIT_CONCAT_CH)
    b.meta("add_uint32", "mm3.dit.rope_dim", DIT_ROPE_DIM)
    b.meta("add_float32", "mm3.dit.rope_theta", DIT_ROPE_THETA)
    b.meta("add_string", "mm3.dit.rope_type", "neox")
    b.meta("add_uint32", "mm3.dit.fourier_dim", DIT_FOURIER_DIM)
    b.meta("add_float32", "mm3.dit.layer_norm_eps", DIT_LN_EPS)
    b.meta("add_string", "mm3.dit.glu_order", "value_gate")
    b.meta("add_bool", "mm3.dit.timestep_token_prepended", True)
    b.meta("add_bool", "mm3.dit.pre_post_conv_residual", True)
    b.meta("add_bool", "mm3.dit.attn_bias", False)
    b.meta("add_uint32", "mm3.dit.window_frames", DIT_WINDOW_FRAMES)
    b.meta("add_uint32", "mm3.dit.hop_frames", DIT_HOP_FRAMES)
    b.meta("add_uint32", "mm3.dit.window_latents", latent_length(DIT_WINDOW_FRAMES))
    b.meta("add_uint32", "mm3.dit.hop_latents", latent_length(DIT_HOP_FRAMES))
    b.meta("add_string", "mm3.flow.scheduler", "FlowMatchEulerDiscrete")
    b.meta("add_uint32", "mm3.flow.steps", DIT_STEPS)
    b.meta("add_float32", "mm3.flow.cfg_scale", DIT_CFG_SCALE)
    b.meta("add_bool", "mm3.flow.invert_sigmas", True)
    b.meta("add_float32", "mm3.flow.shift", 1.0)
    b.meta("add_uint32", "mm3.flow.num_train_timesteps", 1)

    C = "diffusion_transformer."
    b.put("dit.preprocess_conv.weight",
          src.get(C + "preprocess_conv.weight", "preprocess_conv.weight",
                  expect=(DIT_CONCAT_CH, DIT_CONCAT_CH, 1)), HALF)
    b.put("dit.postprocess_conv.weight",
          src.get(C + "postprocess_conv.weight", "postprocess_conv.weight",
                  expect=(DIT_IN_CH, DIT_IN_CH, 1)), HALF)

    b.put("dit.time_fourier.weight",
          src.get(C + "timestep_features.weight", "time_proj.weight",
                  expect=(DIT_FOURIER_DIM // 2, 1)), F32)
    b.put("dit.time_embd.0.weight",
          src.get(C + "to_timestep_embed.0.weight", "time_embed.linear_1.weight",
                  expect=(DIT_DIM, DIT_FOURIER_DIM)), QUANT)
    b.put("dit.time_embd.0.bias",
          src.get(C + "to_timestep_embed.0.bias", "time_embed.linear_1.bias",
                  expect=(DIT_DIM,)), F32)
    b.put("dit.time_embd.1.weight",
          src.get(C + "to_timestep_embed.2.weight", "time_embed.linear_2.weight",
                  expect=(DIT_DIM, DIT_DIM)), QUANT)
    b.put("dit.time_embd.1.bias",
          src.get(C + "to_timestep_embed.2.bias", "time_embed.linear_2.bias",
                  expect=(DIT_DIM,)), F32)
    b.put("dit.proj_in.weight",
          src.get(C + "transformer.project_in.weight", "proj_in.weight",
                  expect=(DIT_DIM, DIT_CONCAT_CH)), QUANT)
    b.put("dit.proj_out.weight",
          src.get(C + "transformer.project_out.weight", "proj_out.weight",
                  expect=(DIT_IN_CH, DIT_DIM)), QUANT)

    inv_freq = 1.0 / (DIT_ROPE_THETA ** (
        np.arange(0, DIT_ROPE_DIM, 2, dtype=np.float64) / DIT_ROPE_DIM))
    stored = None
    if src.has(C + "transformer.rotary_pos_emb.inv_freq"):
        stored = src.get(C + "transformer.rotary_pos_emb.inv_freq",
                         expect=(DIT_ROPE_DIM // 2,))
        rel = np.abs(stored.astype(np.float64) - inv_freq) / np.maximum(inv_freq, 1e-12)
        if rel.max() > 1e-2:
            die(f"dit rope inv_freq mismatch: checkpoint disagrees with "
                f"theta={DIT_ROPE_THETA} by up to {rel.max():.3%}. The DiT rope "
                f"base is not 10000 -- fix DIT_ROPE_THETA before converting")
        log(f"dit rope inv_freq: checkpoint matches theta={DIT_ROPE_THETA:.0f} "
            f"(max rel dev {rel.max():.2e}), using the exact F32 basis")
    b.put("dit.rope_inv_freq", inv_freq.astype(np.float32), F32)

    for i in range(DIT_LAYERS):
        d = f"dit.blk.{i}"
        cp = f"{C}transformer.layers.{i}"
        dp = f"transformer_blocks.{i}"
        b.put(f"{d}.attn_norm.weight",
              src.get(f"{cp}.pre_norm.gamma", f"{dp}.norm1.weight", expect=(DIT_DIM,)), F32)
        b.put(f"{d}.attn_norm.bias",
              src.get(f"{cp}.pre_norm.beta", f"{dp}.norm1.bias", expect=(DIT_DIM,)), F32)

        if src.has(f"{cp}.self_attn.to_qkv.weight"):
            qkv = src.get(f"{cp}.self_attn.to_qkv.weight", expect=(3 * DIT_DIM, DIT_DIM))
        else:
            qkv = np.concatenate([
                src.get(f"{dp}.attn.to_q.weight", expect=(DIT_DIM, DIT_DIM)),
                src.get(f"{dp}.attn.to_k.weight", expect=(DIT_DIM, DIT_DIM)),
                src.get(f"{dp}.attn.to_v.weight", expect=(DIT_DIM, DIT_DIM)),
            ], axis=0)
        b.put(f"{d}.attn_qkv.weight", qkv, QUANT)
        b.put(f"{d}.attn_output.weight",
              src.get(f"{cp}.self_attn.to_out.weight", f"{dp}.attn.to_out.0.weight",
                      expect=(DIT_DIM, DIT_DIM)), QUANT)
        b.put(f"{d}.ffn_norm.weight",
              src.get(f"{cp}.ff_norm.gamma", f"{dp}.norm2.weight", expect=(DIT_DIM,)), F32)
        b.put(f"{d}.ffn_norm.bias",
              src.get(f"{cp}.ff_norm.beta", f"{dp}.norm2.bias", expect=(DIT_DIM,)), F32)
        b.put(f"{d}.ffn_in.weight",
              src.get(f"{cp}.ff.ff.0.proj.weight", f"{dp}.ff_in.weight",
                      expect=(2 * DIT_FF_INNER, DIT_DIM)), QUANT)
        b.put(f"{d}.ffn_in.bias",
              src.get(f"{cp}.ff.ff.0.proj.bias", f"{dp}.ff_in.bias",
                      expect=(2 * DIT_FF_INNER,)), F32)
        b.put(f"{d}.ffn_out.weight",
              src.get(f"{cp}.ff.ff.2.weight", f"{dp}.ff_out.weight",
                      expect=(DIT_DIM, DIT_FF_INNER)), QUANT)
        b.put(f"{d}.ffn_out.bias",
              src.get(f"{cp}.ff.ff.2.bias", f"{dp}.ff_out.bias", expect=(DIT_DIM,)), F32)

    def belongs(n):
        return (n.startswith("diffusion_transformer.")
                or n.startswith("transformer_blocks.")
                or n in ("preprocess_conv.weight", "postprocess_conv.weight",
                         "proj_in.weight", "proj_out.weight", "time_proj.weight")
                or n.startswith("time_embed."))
    return belongs

def build_vocoder(src, b):
    b.meta("add_uint32", "mm3.voc.latent_channels", VOC_LATENT_CH)
    b.meta("add_uint32", "mm3.voc.fold_channels", VOC_FOLD_CH)
    b.meta("add_uint32", "mm3.voc.dec_in_dim", VOC_DEC_IN_DIM)
    b.meta("add_uint32", "mm3.voc.hidden_dim", VOC_HIDDEN)
    b.meta("add_array", "mm3.voc.upsample_rates", list(VOC_UPSAMPLE))
    b.meta("add_array", "mm3.voc.res_dilations", list(VOC_RES_DILATIONS))
    b.meta("add_uint32", "mm3.voc.total_upsample", int(np.prod(VOC_UPSAMPLE)))
    b.meta("add_uint32", "mm3.voc.sampling_rate", VOC_SR)
    b.meta("add_uint32", "mm3.voc.channels", 2)
    b.meta("add_float32", "mm3.voc.snake_eps", VOC_SNAKE_EPS)
    b.meta("add_bool", "mm3.voc.final_tanh", True)
    b.meta("add_bool", "mm3.voc.weight_norm_folded", True)
    b.meta("add_string", "mm3.voc.snake",
           "x + 1/(alpha+eps) * sin(alpha*x)^2")

    def wn(dst, comfy, diffusers, expect_v, expect_bias):
        g = src.get(f"{comfy}.weight_g", f"{diffusers}.weight_g",
                    expect=(expect_v[0], 1, 1))
        v = src.get(f"{comfy}.weight_v", f"{diffusers}.weight_v", expect=expect_v)
        b.put(f"{dst}.weight", fold_weight_norm(g, v, dst), F32)
        b.put(f"{dst}.bias", src.get(f"{comfy}.bias", f"{diffusers}.bias",
                                     expect=(expect_bias,)), F32)

    def alpha(dst, comfy, diffusers, ch):
        b.put(dst, src.get(comfy, diffusers, expect=(1, ch, 1)), F32)

    b.put("voc.dec_in.weight",
          src.get("dec_in_proj.weight", expect=(VOC_DEC_IN_DIM, VOC_FOLD_CH, 1)), F32)
    b.put("voc.dec_in.bias",
          src.get("dec_in_proj.bias", expect=(VOC_DEC_IN_DIM,)), F32)
    wn("voc.conv_in", "decoder.model.0", "conv_in",
       (VOC_HIDDEN, VOC_DEC_IN_DIM, 7), VOC_HIDDEN)

    ch = VOC_HIDDEN
    for bi, stride in enumerate(VOC_UPSAMPLE):
        out_ch = ch // 2
        m = bi + 1
        alpha(f"voc.blk.{bi}.snake.alpha",
              f"decoder.model.{m}.block.0.alpha", f"blocks.{bi}.snake1.alpha", ch)

        wn(f"voc.blk.{bi}.convt", f"decoder.model.{m}.block.1", f"blocks.{bi}.conv_t1",
           (ch, out_ch, 2 * stride), out_ch)
        for ri, dil in enumerate(VOC_RES_DILATIONS):
            u = ri + 2
            alpha(f"voc.blk.{bi}.res.{ri}.snake1.alpha",
                  f"decoder.model.{m}.block.{u}.block.0.alpha",
                  f"blocks.{bi}.res_unit{ri + 1}.snake1.alpha", out_ch)
            wn(f"voc.blk.{bi}.res.{ri}.conv1",
               f"decoder.model.{m}.block.{u}.block.1",
               f"blocks.{bi}.res_unit{ri + 1}.conv1",
               (out_ch, out_ch, 7), out_ch)
            alpha(f"voc.blk.{bi}.res.{ri}.snake2.alpha",
                  f"decoder.model.{m}.block.{u}.block.2.alpha",
                  f"blocks.{bi}.res_unit{ri + 1}.snake2.alpha", out_ch)
            wn(f"voc.blk.{bi}.res.{ri}.conv2",
               f"decoder.model.{m}.block.{u}.block.3",
               f"blocks.{bi}.res_unit{ri + 1}.conv2",
               (out_ch, out_ch, 1), out_ch)
        ch = out_ch

    n_blk = len(VOC_UPSAMPLE)
    alpha("voc.snake_out.alpha",
          f"decoder.model.{n_blk + 1}.alpha", "snake_out.alpha", ch)
    wn("voc.conv_out", f"decoder.model.{n_blk + 2}", "conv_out", (1, ch, 7), 1)

    def belongs(n):
        return (n.startswith("decoder.model.") or n.startswith("blocks.")
                or n.startswith("dec_in_proj.") or n.startswith("conv_in.")
                or n.startswith("conv_out.") or n.startswith("snake_out."))
    return belongs

ALL_COMPONENTS = ("lm", "depth", "cond", "dit", "vocoder")
SYNTH_COMPONENTS = ("depth", "cond", "dit", "vocoder")

EPILOG = """\
components:
  lm       global LM (Qwen3-8B + extended audio vocab)   -> mm3-lm-<quant>.gguf
  depth    RVQ depth decoder (0.6B)          \\
  cond     condition encoder (~25M)           }-> mm3-synth-<quant>.gguf
  dit      flow-matching transformer (2.4B)   |
  vocoder  DAC-style vocoder decoder (54M)   /

quantisation:
  --quant f16   (default) 2-D weights F16, norms/biases/alphas/scales F32.
  --quant q8_0  additionally Q8_0 for 2-D weights whose innermost dim is a
                multiple of 32. The vocoder is ALWAYS F32 (54M params; the
                Snake + folded weight-norm path is quality-critical) and the
                DiT timestep Fourier basis is always F32.

  You do not have to quantise here. Emitting f16 and running the existing
  `acestep-quantize` tool over mm3-lm-f16.gguf and mm3-synth-f16.gguf is the
  better-trodden path (it gives access to the k-quants this script does not
  implement); acestep-quantize quantizes the synth file's DiT, holds the RVQ
  depth decoder at q8_0, and keeps the condition encoder and vocoder at their
  converted precision.

sources (--src may be repeated; directories are scanned recursively):
  preferred  Comfy-Org/MiniMax-Music-3     (single-file safetensors)
  fallback   MiniMaxAI/MiniMax-Music3      (diffusers subfolder layout)
  refused    *_int8_convrot.safetensors, *_pruned_*.safetensors

"""

def main():
    ap = argparse.ArgumentParser(
        prog="convert-minimax-music3-to-gguf.py",
        description="MiniMax-Music3 safetensors -> GGUF for the audiogen-cpp engine.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", required=True, metavar="PATH",
                    help="model dir or .safetensors/.pth file (repeatable)")
    ap.add_argument("--out", required=True, metavar="DIR", help="output directory")
    ap.add_argument("--components", default=",".join(ALL_COMPONENTS),
                    help=f"comma list of {','.join(ALL_COMPONENTS)} (default: all)")
    ap.add_argument("--quant", default="f16", choices=("f16", "q8_0"))
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing output files (default: skip)")
    ap.add_argument("--tokenizer", metavar="PATH",
                    help="tokenizer.json (default: found in --src, or read from "
                         "the source's embedded tokenizer_json tensor)")
    ap.add_argument("--embed-tokenizer-json", action="store_true",
                    help="also store tokenizer.json verbatim as mm3.tokenizer_json "
                         "(+11 MB)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    want = [c.strip() for c in args.components.split(",") if c.strip()]
    bad = [c for c in want if c not in ALL_COMPONENTS]
    if bad:
        die(f"unknown component(s) {bad}; valid: {', '.join(ALL_COMPONENTS)}")
    if not want:
        die("--components selected nothing")

    os.makedirs(args.out, exist_ok=True)
    lm_path = os.path.join(args.out, f"mm3-lm-{args.quant}.gguf")
    synth_path = os.path.join(args.out, f"mm3-synth-{args.quant}.gguf")

    want_lm = "lm" in want
    want_synth = [c for c in SYNTH_COMPONENTS if c in want]
    if want_lm and os.path.exists(lm_path) and not args.force:
        log(f"skip (exists): {lm_path} -- pass --force to overwrite")
        want_lm = False
    if want_synth and os.path.exists(synth_path) and not args.force:
        log(f"skip (exists): {synth_path} -- pass --force to overwrite")
        want_synth = []
    if not want_lm and not want_synth:
        log("nothing to do")
        return

    src = Source(args.src, verbose=args.verbose)
    transaction = OutputTransaction()
    try:
        predicates = []

        if want_lm:
            tok, raw = find_tokenizer(src, args.tokenizer, args.src)
            require_tokenizer(tok)
            b = Bundle("qwen3")
            common_meta(b, src, args, ["lm"])
            predicates.append(("lm", build_lm(
                src, b, tok, raw if args.embed_tokenizer_json else None)))
            transaction.write(b, lm_path, args.quant, "lm")

        if want_synth:
            if len(want_synth) != len(SYNTH_COMPONENTS):
                log(f"WARNING: partial synth file -- only {want_synth} included. "
                    f"The engine loader expects all of {list(SYNTH_COMPONENTS)}")
            b = Bundle("mm3")
            b.meta("add_name", "MiniMax-Music3 synth")
            b.meta("add_description",
                   "MiniMax-Music3 synthesis stack: RVQ depth decoder, condition "
                   "encoder, flow-matching transformer, DAC-style vocoder")
            common_meta(b, src, args, want_synth)
            b.meta("add_array", "mm3.synth.components", list(want_synth))
            builders = {"depth": build_depth, "cond": build_cond,
                        "dit": build_dit, "vocoder": build_vocoder}
            for c in want_synth:
                predicates.append((c, builders[c](src, b)))
            transaction.write(b, synth_path, args.quant, "synth")

        leftover = []
        for comp, pred in predicates:
            for n in src.unconsumed(pred):
                leftover.append((comp, n))
        if leftover:
            log(f"ERROR: {len(leftover)} source tensors matched a converted "
                f"component but were not written:")
            for comp, n in leftover[:40]:
                f = src.index[n]
                log(f"    [{comp}] {n} {tuple(f.shape(n))} {f.dtype(n)}")
            if len(leftover) > 40:
                log(f"    ... and {len(leftover) - 40} more")
            die("the converter does not understand this checkpoint layout -- "
                "refusing to leave a partial model in place. Delete the output "
                "files and report the list above.")

        if args.verbose:
            others = src.unconsumed(lambda n: True)
            if others:
                log(f"note: {len(others)} source tensors outside the converted "
                    f"components were ignored, e.g. {others[:5]}")
        transaction.commit()
        log("Done.")
    finally:
        transaction.cleanup()
        src.close()

def common_meta(b, src, args, components):

    b.meta("add_license", "MiniMax-Music3 Community License")
    b.meta("add_string", "mm3.model", "MiniMax-Music3")
    b.meta("add_uint32", "mm3.converter_version", CONVERTER_VERSION)
    b.meta("add_string", "mm3.source_layout", src.layout)
    b.meta("add_file_type",
           int(gguf.LlamaFileType.MOSTLY_Q8_0 if args.quant == "q8_0"
               else gguf.LlamaFileType.MOSTLY_F16))
    b.meta("add_quantization_version", gguf.GGML_QUANT_VERSION)

if __name__ == "__main__":
    main()
