#!/usr/bin/env python3
"""Export fixed-shape Supertonic vector-estimator buckets from an F32/F16 GGUF.

Time embeddings and length-normalized rotary tables are supplied by the host.
Padded latent frames are edge-filled before each convolution; padded text keys
are excluded from attention. Guidance and Euler integration stay in FP32 on
the host; the graph returns conditional/unconditional velocities.
"""

import argparse
import importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from supertonic_coreml_gguf import dequantized, resolve_tensor_names, sidecar_stem


def sibling(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


encoder = sibling("encoder_export", "export-supertonic-encoder-coreml.py")
parameter, Linear, Norm, ConvNeXt = (
    encoder.parameter,
    encoder.Linear,
    encoder.Norm,
    encoder.ConvNeXt,
)
PREFIX = "vector_estimator:tts.ttl.vector_field."


class Projection(nn.Module):
    def __init__(self, tensors, prefix):
        super().__init__()
        weight = tensors[prefix + ".weight"]
        self.weight = parameter(weight.reshape(weight.shape[0], -1))

    def forward(self, x):
        return F.linear(x, self.weight)


class TextAttention(nn.Module):
    def __init__(self, tensors, prefix, heads):
        super().__init__()
        self.heads = heads
        for name in ("W_query", "W_key", "W_value", "out_fc"):
            setattr(
                self,
                name,
                Linear(tensors, prefix + "." + name + ".linear", matmul=True),
            )

    def forward(self, x, text, key_mask, cos_q, sin_q, cos_k, sin_k):
        batch, length, _ = x.shape
        text_length = text.shape[1]
        q = self.W_query(x).reshape(batch, length, self.heads, 64).permute(0, 2, 1, 3)
        k = (
            self.W_key(text)
            .reshape(batch, text_length, self.heads, 64)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.W_value(text)
            .reshape(batch, text_length, self.heads, 64)
            .permute(0, 2, 1, 3)
        )

        def rotate(y, cos, sin):
            a, b = y[..., :32], y[..., 32:]
            return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)

        q, k = rotate(q, cos_q, sin_q), rotate(k, cos_k, sin_k)
        probs = torch.softmax(
            torch.matmul(q, k.transpose(-1, -2)) / 16.0 + key_mask, dim=-1
        )
        y = (
            torch.matmul(probs, v)
            .permute(0, 2, 1, 3)
            .reshape(batch, length, self.heads * 64)
        )
        return self.out_fc(y)


class StyleAttention(nn.Module):
    def __init__(self, tensors, prefix):
        super().__init__()
        for name in ("W_query", "W_key", "W_value", "out_fc"):
            setattr(
                self,
                name,
                Linear(tensors, prefix + "." + name + ".linear", matmul=True),
            )

    def forward(self, x, key, value):
        batch, length, _ = x.shape
        q = self.W_query(x).reshape(batch, length, 2, 128).permute(0, 2, 1, 3)
        k = torch.tanh(self.W_key(key)).reshape(batch, 50, 2, 128).permute(0, 2, 3, 1)
        v = self.W_value(value).reshape(batch, 50, 2, 128).permute(0, 2, 1, 3)
        probs = torch.softmax(torch.matmul(q, k) / 16.0, dim=-1)
        y = torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(batch, length, 256)
        return self.out_fc(y)


class Group(nn.Module):
    def __init__(self, tensors, index, heads):
        super().__init__()
        base = PREFIX + "main_blocks."
        i = index * 6
        self.first = nn.ModuleList(
            [
                ConvNeXt(tensors, base + f"{i}.convnext.{j}", d)
                for j, d in enumerate((1, 2, 4, 8))
            ]
        )
        self.time = Linear(tensors, base + f"{i + 1}.linear.linear", matmul=True)
        self.mid = ConvNeXt(tensors, base + f"{i + 2}.convnext.0", 1)
        self.text_attn = TextAttention(tensors, base + f"{i + 3}.attn", heads)
        self.text_norm = Norm(tensors, base + f"{i + 3}.norm.norm")
        self.end = ConvNeXt(tensors, base + f"{i + 4}.convnext.0", 1)
        self.style_attn = StyleAttention(tensors, base + f"{i + 5}.attention")
        self.style_norm = Norm(tensors, base + f"{i + 5}.norm.norm")

    def forward(
        self, x, text, style_key, style, time, valid, last, key_mask, cq, sq, ck, sk
    ):
        for block in self.first:
            x = block(x, valid, last)
        x = self.mid(x + self.time(time)[:, None, :], valid, last)
        x = self.text_norm(x + self.text_attn(x, text, key_mask, cq, sq, ck, sk))
        x = self.end(x, valid, last)
        return self.style_norm(x + self.style_attn(x, style_key, style))


class VectorEstimator(nn.Module):
    def __init__(self, tensors, heads=8, cond_scale=4.0, uncond_scale=3.0):
        super().__init__()
        self.cond_scale, self.uncond_scale = cond_scale, uncond_scale
        self.cfg = uncond_scale != 0
        self.proj_in = Projection(tensors, PREFIX + "proj_in.net")
        self.proj_out = Projection(tensors, PREFIX + "proj_out.net")
        self.groups = nn.ModuleList([Group(tensors, i, heads) for i in range(4)])
        self.tail = nn.ModuleList(
            [
                ConvNeXt(tensors, PREFIX + f"last_convnext.convnext.{i}", 1)
                for i in range(4)
            ]
        )
        self.style_key = parameter(
            tensors["vector_estimator:/Expand_output_0"].reshape(1, 50, 256)
        )
        if self.cfg:
            null = "vector_estimator:tts.ttl.uncond_masker."
            self.null_text = parameter(
                tensors[null + "text_special_token"].reshape(1, 1, 256)
            )
            self.null_style_key = parameter(
                tensors[null + "style_key_special_token"].reshape(1, 50, 256)
            )
            self.null_style_value = parameter(
                tensors[null + "style_value_special_token"].reshape(1, 50, 256)
            )

    def velocity(
        self,
        noisy_latent,
        text_emb,
        style_ttl,
        latent_mask,
        text_mask,
        time_emb,
        cos_q,
        sin_q,
        cos_k,
        sin_k,
    ):
        valid = latent_mask.transpose(1, 2)
        # Prefix masks only: first invalid position marks the last valid frame.
        last = valid - F.pad(valid[:, 1:, :], (0, 0, 0, 1))
        key_mask = (1.0 - text_mask[:, None, :, :]) * -10000.0
        text = text_emb.transpose(1, 2)
        style, key = style_ttl, self.style_key
        x = noisy_latent.transpose(1, 2)
        if self.cfg:
            x = torch.cat((x, x), dim=0)
            text = torch.cat(
                (text, self.null_text.expand(1, text.shape[1], 256)), dim=0
            )
            style = torch.cat((style, self.null_style_value), dim=0)
            key = torch.cat((key, self.null_style_key), dim=0)
        x = self.proj_in(x) * valid
        for group in self.groups:
            x = group(
                x,
                text,
                key,
                style,
                time_emb,
                valid,
                last,
                key_mask,
                cos_q,
                sin_q,
                cos_k,
                sin_k,
            )
        for block in self.tail:
            x = block(x, valid, last)
        return self.proj_out(x).transpose(1, 2)

    def forward(
        self,
        noisy_latent,
        text_emb,
        style_ttl,
        latent_mask,
        text_mask,
        time_emb,
        cos_q,
        sin_q,
        cos_k,
        sin_k,
        inv_steps,
    ):
        velocity = self.velocity(
            noisy_latent,
            text_emb,
            style_ttl,
            latent_mask,
            text_mask,
            time_emb,
            cos_q,
            sin_q,
            cos_k,
            sin_k,
        )
        if self.cfg:
            velocity = self.cond_scale * velocity[:1] - self.uncond_scale * velocity[1:]
        return noisy_latent + velocity * latent_mask * inv_steps


class VelocityField(nn.Module):
    """The deployed graph leaves CFG subtraction and Euler accumulation in FP32
    on the host, preserving the unrounded input latent across iterations."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        return self.model.velocity(*inputs)


def load_vector(path):
    from gguf import GGUFReader

    reader = GGUFReader(str(path))

    def value(name, default=None):
        field = reader.fields.get("supertonic." + name)
        return field.contents() if field else default

    names = resolve_tensor_names(
        value("source_names", []),
        value("tensor_names", []),
        value("source_aliases", []),
        value("source_alias_targets", []),
    )
    # Legacy v2 logical weights predate the converter's canonical alias arrays.
    time_ids = (3095, 3140, 3185, 3230)
    for g in range(4):
        base = PREFIX + "main_blocks."
        aliases = {base + f"{g * 6 + 1}.linear.linear.weight": time_ids[g]}
        for offset, group_offset, kind in ((6, 3, "attn"), (21, 5, "attention")):
            ids = [time_ids[g] + offset + j for j in range(3)] + [
                time_ids[g] + (15 if kind == "attn" else 24)
            ]
            for name, number in zip(("W_query", "W_key", "W_value", "out_fc"), ids):
                aliases[
                    base + f"{g * 6 + group_offset}.{kind}.{name}.linear.weight"
                ] = number
        for canonical, number in aliases.items():
            legacy = f"vector_estimator:onnx::MatMul_{number}"
            if canonical not in names and legacy in names:
                names[canonical] = names[legacy]
    storage = {t.name: t for t in reader.tensors}
    tensors = {}
    for source, name in names.items():
        if not source.startswith("vector_estimator:"):
            continue
        tensor = storage[name]
        if tensor.tensor_type.name.startswith("I"):
            continue
        if tensor.tensor_type.name not in ("F32", "F16", "BF16"):
            raise ValueError("export from an F32/F16 GGUF, not block-quantized weights")
        tensors[source] = dequantized(tensor)
    model = VectorEstimator(
        tensors,
        int(value("vector_text_attn_heads", 4)),
        float(value("cfg_cond_scale", 1.0)),
        float(value("cfg_uncond_scale", 0.0)),
    ).eval()
    theta = tensors[PREFIX + "main_blocks.3.attn.theta"].reshape(32)
    return model, tensors, theta


def inputs_for(
    tensors, theta, latent, text, style, step, steps, capacity, text_capacity
):
    length, text_length = latent.shape[-1], text.shape[-1]
    if (
        not 0 < length <= capacity
        or not 0 < text_length <= text_capacity
        or not 0 <= step < steps
    ):
        raise ValueError("invalid lengths or denoising schedule")

    def rotary(size, actual):
        angles = (
            np.arange(size, dtype=np.float32)[:, None]
            / np.float32(actual)
            * theta[None, :]
        )
        return np.cos(angles)[None, None], np.sin(angles)[None, None]

    cq, sq = rotary(capacity, length)
    ck, sk = rotary(text_capacity, text_length)
    phase = (np.float32(step) / np.float32(steps) * np.float32(1000)) * np.exp(
        np.arange(32, dtype=np.float32) * np.float32(-np.log(10000) / 31)
    )
    time = np.concatenate((np.sin(phase), np.cos(phase)))
    p = PREFIX + "time_encoder.mlp."
    time = F.linear(
        torch.from_numpy(time),
        torch.from_numpy(tensors[p + "0.linear.weight"]),
        torch.from_numpy(tensors[p + "0.linear.bias"]),
    )
    time = F.mish(time)
    time = F.linear(
        time,
        torch.from_numpy(tensors[p + "2.linear.weight"]),
        torch.from_numpy(tensors[p + "2.linear.bias"]),
    ).numpy()[None]

    def pad(a, size):
        return np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, size - a.shape[-1])])

    lm = np.zeros((1, 1, capacity), np.float32)
    lm[..., :length] = 1
    tm = np.zeros((1, 1, text_capacity), np.float32)
    tm[..., :text_length] = 1
    return tuple(
        torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        for a in (
            pad(latent, capacity),
            pad(text, text_capacity),
            style,
            lm,
            tm,
            time,
            cq,
            sq,
            ck,
            sk,
            np.array([1 / steps], np.float32),
        )
    )


NAMES = (
    "noisy_latent",
    "text_emb",
    "style_ttl",
    "latent_mask",
    "text_mask",
    "time_emb",
    "cos_q",
    "sin_q",
    "cos_k",
    "sin_k",
    "inv_steps",
)
FIELD_NAMES = NAMES[:-1]


def palettize_field(model, bits):
    """Compress projection/convolution weights; keep normalization and other constants."""
    import coremltools.optimize.coreml as cto

    if bits != 8:
        raise ValueError("only 8-bit palettization is supported")
    config = cto.OpPalettizerConfig(mode="kmeans", nbits=bits, weight_threshold=2048)
    return cto.palettize_weights(
        model,
        cto.OptimizationConfig(op_type_configs={"conv": config, "linear": config}),
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True)
    p.add_argument(
        "--buckets", type=int, nargs="+", choices=[64, 128, 256, 512], default=[64, 128]
    )
    p.add_argument("--text-capacity", type=int, default=128)
    p.add_argument("--compile-dir", type=Path)
    p.add_argument("--parity-dir", type=Path)
    p.add_argument("--no-export", action="store_true")
    p.add_argument(
        "--layout",
        choices=["3d", "4d"],
        default="4d",
        help="internal vector-field layout (default: 4d channel-first convolutions)",
    )
    p.add_argument(
        "--palettize",
        type=int,
        choices=[0, 8],
        default=0,
        help="optional 8-bit weight LUT compression (default: 0, uncompressed)",
    )
    p.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="FP16 for ANE inference (default); FP32 for numerical diagnostics",
    )
    args = p.parse_args(argv)
    if not 1 <= args.text_capacity <= 4096 or any(b < 1 for b in args.buckets):
        p.error("text capacity must be 1..4096 and latent capacities positive")
    return args


def main():
    args = parse_args()
    model, tensors, theta = load_vector(args.gguf)
    if args.layout == "4d":
        from supertonic_coreml_vector_4d import VectorEstimator4D

        model = VectorEstimator4D(model).eval()
    if args.parity_dir:
        r = args.parity_dir
        latent = np.load(r / "noise.npy")
        text = np.load(r / "text_emb.npy")
        style = np.load(r / "style_ttl.npy")
        refs = sorted(r.glob("vector_step_*.npy"))
        if not refs:
            raise ValueError("no denoising step references found")
        for step, ref in enumerate(refs):
            inp = inputs_for(
                tensors,
                theta,
                latent,
                text,
                style,
                step,
                len(refs),
                max(args.buckets),
                args.text_capacity,
            )
            with torch.no_grad():
                got = model(*inp).numpy()[..., : latent.shape[-1]]
            want = np.load(ref)
            rmse = np.sqrt(np.mean((got - want) ** 2))
            print(
                f"[parity] step={step} rmse={rmse:.6g} max_abs={np.max(np.abs(got - want)):.6g}",
                flush=True,
            )
            np.testing.assert_allclose(got, want, atol=0.003, rtol=0.003)
            latent = want
    if args.no_export:
        return
    import coremltools as ct

    helper = sibling("vocoder_export", "export-supertonic-coreml.py")
    for bucket in sorted(set(args.buckets)):
        sample = inputs_for(
            tensors,
            theta,
            np.zeros((1, 144, min(17, bucket)), np.float32),
            np.zeros((1, 256, min(21, args.text_capacity)), np.float32),
            np.zeros((1, 50, 256), np.float32),
            0,
            5,
            bucket,
            args.text_capacity,
        )
        sample = sample[:-1]
        with torch.no_grad():
            traced = torch.jit.trace(VelocityField(model).eval(), sample)
        exported = ct.convert(
            traced,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS14,
            compute_units=ct.ComputeUnit.CPU_AND_NE
            if args.precision == "fp16"
            else ct.ComputeUnit.CPU_AND_GPU,
            compute_precision=ct.precision.FLOAT16
            if args.precision == "fp16"
            else ct.precision.FLOAT32,
            inputs=[
                ct.TensorType(name=name, shape=x.shape, dtype=np.float32)
                for name, x in zip(FIELD_NAMES, sample)
            ],
            outputs=[ct.TensorType(name="velocity", dtype=np.float32)],
        )
        if args.palettize:
            exported = palettize_field(exported, args.palettize)
        exported.user_defined_metadata.update(
            {
                "supertonic.vector_protocol": "2",
                "supertonic.vector_layout": args.layout,
                "supertonic.vector_lut_bits": str(args.palettize),
                "supertonic.vector_heads": str(model.groups[0].text_attn.heads),
                "supertonic.cfg_cond_scale": str(model.cond_scale),
                "supertonic.cfg_uncond_scale": str(model.uncond_scale),
                "supertonic.weight_bits": str(16 if args.precision == "fp16" else 32),
            }
        )
        stem = sidecar_stem(args.gguf).removesuffix("-vocoder") + f"-vector-L{bucket}"
        out = Path(args.gguf).with_name(stem + ".mlpackage")
        exported.save(str(out))
        print(f"[export] saved {out}", flush=True)
        if args.compile_dir:
            compiled = helper.compile_model(out, args.compile_dir)
            print(f"[export] compiled {compiled}", flush=True)
            helper.report_placement(
                compiled,
                ct.ComputeUnit.CPU_AND_NE
                if args.precision == "fp16"
                else ct.ComputeUnit.CPU_AND_GPU,
            )


if __name__ == "__main__":
    main()
