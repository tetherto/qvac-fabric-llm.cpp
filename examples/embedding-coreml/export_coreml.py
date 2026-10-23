"""Export a local EmbeddingGemma 2 safetensors checkpoint for the Core ML adapter."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


LENGTHS = (32, 48, 64, 128, 256, 512)
TOKEN_TABLE = "language_model.embed_tokens.weight"


def rms_normalize(values, eps: float):
    # Scale before squaring so FP16 activations cannot overflow.
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    scaled = values / scale
    variance = (scaled * scaled).mean(dim=-1, keepdim=True) + eps / scale / scale
    return scaled * variance.rsqrt()


def l2_normalize(values):
    scaled = values / values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    return scaled / (scaled * scaled).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)


def read_config(source: Path) -> dict:
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "embedding_gemma2":
        raise ValueError("Expected an EmbeddingGemma 2 checkpoint")
    text = config.get("text_config", {})
    expected = {
        "hidden_size": 512, "embedding_dim": 768, "vocab_size": 262144,
        "hidden_size_per_layer_input": 512, "num_hidden_layers": 24,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 256,
        "intermediate_size": 2048, "sliding_window": 512, "rms_norm_eps": 1e-6,
        "hidden_activation": "gelu_pytorch_tanh", "attention_bias": False,
        "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0,
    }
    for name, value in expected.items():
        if text.get(name) != value:
            raise ValueError(f"Unsupported text_config.{name}: {text.get(name)!r}; expected {value!r}")
    layer_types = ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(24)]
    if text.get("layer_types") != layer_types:
        raise ValueError("Unsupported attention layer layout")
    per_layer = {int(key): value for key, value in text.get("per_layer_config", {}).items()}
    if per_layer != {i: {"head_dim": 512, "num_key_value_heads": 1} for i in (5, 11, 17, 23)}:
        raise ValueError("Unsupported per-layer head configuration")
    for name, theta in (("full_attention", 1000000.0), ("sliding_attention", 10000.0)):
        if text.get("rope_parameters", {}).get(name) != {"rope_type": "default", "rope_theta": theta}:
            raise ValueError(f"Unsupported {name} rotary configuration")
    return text


def function_contracts(slots: int) -> dict:
    if slots not in (8, 16):
        raise ValueError("Packing requires 8 or 16 slots")
    contracts = {
        f"embed_{length}": {"inputs_embeds": (1, length, 512), "attention_mask": (1, length)}
        for length in LENGTHS
    }
    contracts["pack_256"] = {
        "inputs_embeds": (1, 256, 512), "attention_bias": (1, 1, 256, 256),
        "positions": (256, 1), "pool": (slots, 256),
    }
    return contracts


def checkpoint_files(source: Path) -> list[Path]:
    index = source / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        names = sorted({name for key, name in mapping.items() if key.startswith("language_model.")})
    else:
        names = ["model.safetensors"]
    if not names:
        raise ValueError("No text encoder safetensors found")
    files = []
    for name in names:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Checkpoint shards must be safetensors files in the source directory")
        path = source / name
        if not path.is_file():
            raise ValueError(f"Missing checkpoint shard: {path}")
        files.append(path)
    return files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_functions(description, contracts: dict, slots: int, dtype: int, raw_dtype: int) -> None:
    functions = {function.name: function for function in description.functions}
    if set(functions) != set(contracts):
        raise ValueError("Exported functions do not match the adapter contract")
    for name, inputs in contracts.items():
        function = functions[name]
        if {feature.name for feature in function.input} != set(inputs):
            raise ValueError(f"Unexpected exported inputs for {name}")
        for feature in function.input:
            array = feature.type.multiArrayType
            if tuple(array.shape) != inputs[feature.name] or array.dataType != dtype:
                raise ValueError(f"Unexpected exported shape or type for {name}.{feature.name}")
        if len(function.output) != 2 or {feature.name for feature in function.output} != {"embedding", "embedding_raw"}:
            raise ValueError(f"Unexpected exported outputs for {name}")
        for feature in function.output:
            array = feature.type.multiArrayType
            expected_dtype = raw_dtype if feature.name == "embedding_raw" else dtype
            if tuple(array.shape) != (slots if name == "pack_256" else 1, 768) or array.dataType != expected_dtype:
                raise ValueError(f"Unexpected exported embedding shape or type for {name}.{feature.name}")


def load_encoder(source: Path, destination: Path, text: dict):
    import torch
    from safetensors import safe_open
    from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import EmbeddingGemma2TextConfig
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
        EmbeddingGemma2RotaryEmbedding, EmbeddingGemma2TextModel,
    )

    config = EmbeddingGemma2TextConfig(**text)
    config._attn_implementation = "eager"
    # Build without allocating the token table or the unused media encoders.
    with torch.device("meta"):
        model = EmbeddingGemma2TextModel(config)
    setattr(model, "embed_tokens", torch.nn.Identity())
    model.rotary_emb = EmbeddingGemma2RotaryEmbedding(config)
    expected = model.state_dict()
    state = {}
    table_found = False
    provenance = {}
    for path in checkpoint_files(source):
        provenance[path.name] = sha256(path)
        with safe_open(path, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                if not key.startswith("language_model."):
                    continue
                if key == TOKEN_TABLE:
                    if table_found:
                        raise ValueError("Duplicate token table")
                    table = tensors.get_slice(key)
                    if table.get_shape() != [262144, 512] or table.get_dtype() != "BF16":
                        raise ValueError("Expected the original 262144 x 512 BF16 token table")
                    with (destination / "embeddings.bf16").open("wb") as output:
                        for start in range(0, 262144, 4096):
                            rows = table[start:start + 4096].contiguous()
                            output.write(rows.view(torch.int16).numpy().astype("<i2", copy=False).tobytes())
                    table_found = True
                    continue
                name = key.removeprefix("language_model.")
                if name not in expected or name in state:
                    raise ValueError(f"Unexpected or duplicate text tensor: {key}")
                tensor = tensors.get_tensor(key)
                if tensor.shape != expected[name].shape or not tensor.is_floating_point():
                    raise ValueError(f"Unexpected shape or type for {key}")
                state[name] = tensor.float()
    if not table_found:
        raise ValueError("Missing BF16 token table")
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval(), provenance


def export_wrapper(model, packed: bool):
    import torch
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2RMSNorm

    class SafeRMSNorm(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.eps = original.eps
            self.weight = original.weight if original.with_scale else None

        def forward(self, values):  # ty: ignore[invalid-mutable-override]
            result = rms_normalize(values, self.eps)
            return result if self.weight is None else result * self.weight

    def replace_norms(module):
        for name, child in list(module.named_children()):
            if isinstance(child, EmbeddingGemma2RMSNorm):
                setattr(module, name, SafeRMSNorm(child))
            else:
                replace_norms(child)

    replace_norms(model)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = model.layers
            self.ple = model.ple
            self.norm = model.norm
            self.projection = model.embedding_projection
            self.layer_types = model.config.layer_types
            self.packed = packed
            for kind in sorted(set(self.layer_types)):
                cos, sin = model.rotary_emb(torch.empty(1), torch.arange(512).unsqueeze(0), kind)
                self.register_buffer(kind + "_cos", cos.squeeze(0))
                self.register_buffer(kind + "_sin", sin.squeeze(0))

        def forward(self, inputs_embeds, attention_mask, positions=None, pool=None):  # ty: ignore[invalid-mutable-override]
            length = inputs_embeds.shape[1]
            if self.packed:
                assert positions is not None and pool is not None
                bias = attention_mask
            else:
                bias = (1.0 - attention_mask[:, None, None, :]) * -10000.0
                positions = torch.arange(length)
                pool = attention_mask / attention_mask.sum(dim=-1, keepdim=True)
            rotations = {}
            for kind in sorted(set(self.layer_types)):
                cos = getattr(self, kind + "_cos")
                sin = getattr(self, kind + "_sin")
                if self.packed:
                    indices = positions.reshape(-1).to(torch.int32)
                    cos = torch.nn.functional.embedding(indices, cos)
                    sin = torch.nn.functional.embedding(indices, sin)
                else:
                    cos, sin = cos[:length], sin[:length]
                rotations[kind] = (cos.unsqueeze(0), sin.unsqueeze(0))
            per_layer = self.ple(inputs_embeds)
            hidden = inputs_embeds
            for i, layer in enumerate(self.layers):
                hidden = layer(hidden, per_layer[:, :, i, :], attention_mask=bias,
                               position_embeddings=rotations[self.layer_types[i]])
            hidden = self.projection(self.norm(hidden)).squeeze(0)
            pooled = torch.matmul(pool, hidden)
            return l2_normalize(pooled), pooled

    return Encoder().eval()


def export(source: Path, destination: Path, slots: int) -> None:
    text = read_config(source)
    contracts = function_contracts(slots)
    checkpoint_files(source)
    if not (source / "tokenizer.json").is_file():
        raise ValueError("Missing tokenizer.json")
    if destination.exists():
        raise ValueError("Destination already exists; refusing to overwrite model assets")

    import coremltools as ct
    from coremltools.proto.FeatureTypes_pb2 import ArrayFeatureType
    import numpy as np
    import torch
    import transformers

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete export. Failed conversions leave no destination.
    with tempfile.TemporaryDirectory(prefix=".embeddinggemma-export-", dir=destination.parent) as temporary:
        work = Path(temporary)
        assets = work / "assets"
        assets.mkdir()
        model, checkpoints = load_encoder(source, assets, text)
        functions = ct.utils.MultiFunctionDescriptor()
        with torch.inference_mode():
            for name, shapes in contracts.items():
                packed = name == "pack_256"
                wrapper = export_wrapper(model, packed)
                examples = [torch.zeros(shape, dtype=torch.float32) for shape in shapes.values()]
                if packed:
                    examples[2][:, 0] = torch.arange(256)
                    examples[3][:, :16] = 1.0 / 16
                else:
                    examples[1][:] = 1
                print(f"Exporting {name}", flush=True)
                traced = torch.jit.trace(wrapper, tuple(examples), check_trace=False)
                converted = ct.convert(
                    traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                    compute_precision=ct.precision.FLOAT16, skip_model_load=True,
                    inputs=[ct.TensorType(name=key, shape=shape, dtype=np.float16) for key, shape in shapes.items()],
                    outputs=[ct.TensorType(name="embedding", dtype=np.float16), ct.TensorType(name="embedding_raw", dtype=np.float32)],
                )
                path = work / (name + ".mlpackage")
                converted.save(str(path))
                functions.add_function(str(path), "main", name)
        functions.default_function_name = "embed_32"
        package = assets / "EmbeddingGemma2Text.mlpackage"
        ct.utils.save_multifunction(functions, str(package))
        validate_functions(ct.utils.load_spec(str(package)).description, contracts, slots,
                           getattr(ArrayFeatureType, "FLOAT16"), getattr(ArrayFeatureType, "FLOAT32"))
        shutil.copy2(source / "tokenizer.json", assets / "tokenizer.json")
        shutil.copy2(source / "config.json", assets / "source-config.json")
        manifest = {
            "source_checkpoints_sha256": checkpoints,
            "source_config_sha256": sha256(source / "config.json"),
            "tokenizer_sha256": sha256(assets / "tokenizer.json"),
            "token_table_sha256": sha256(assets / "embeddings.bf16"),
            "model_sha256": sha256(package / "Data/com.apple.CoreML/model.mlmodel"),
            "pack_slots": slots, "functions": contracts,
            "outputs": {"embedding": "float16_l2_normalized", "embedding_raw": "float32_mean_pooled"},
            "torch": torch.__version__, "transformers": transformers.__version__, "coremltools": ct.__version__,
        }
        (assets / "export.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if destination.exists():
            raise ValueError("Destination appeared during export; refusing to overwrite it")
        assets.rename(destination)
    print(f"Exported assets to {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="local Hugging Face checkpoint directory")
    parser.add_argument("destination", type=Path, help="new output directory")
    parser.add_argument("--pack-slots", type=int, choices=(8, 16), default=8)
    args = parser.parse_args()
    try:
        export(args.source, args.destination, args.pack_slots)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
