from __future__ import annotations

from typing import Iterable, cast

import torch
from torch import Tensor

import gguf
import numpy as np

from .base import ModelBase
from .qwen import _LinearAttentionVReorderBase, _Qwen35MRopeMixin
from .qwen3vl import Qwen3VLVisionModel


@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    """Qwen3.8-Flash-Next.

    Shares the Qwen3.5 gated delta net and interleaved mrope, and adds three things:
    hyper-connections in place of every layer norm, QSA sparse attention on the full
    attention layers, and PLE n-gram hash embeddings on a single layer.
    """

    model_arch = gguf.MODEL_ARCH.QWEN4EXP

    # the MTP block is a separate draft head; vLLM drops it too
    supports_mtp_export = False
    no_mtp = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # only the shard names, so the table itself is never held
        self._ple_shards: dict[int, str] = {}
        self._ple_row_dim: int | None = None
        self._ple_weight_scale: float | None = None
        self._fp8_expert_buffer: list[dict[str, Tensor]] | None = None
        self._fp8_gguf_weights: set[str] = set()

        if self._preserve_fp8_experts:
            quant = self.hparams.get("quantization_config") or {}
            if quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [128, 128]:
                raise ValueError("--preserve-fp8-experts requires E4M3 weights with 128x128 block scales")
            if not self.fuse_gate_up_exps:
                raise ValueError("--preserve-fp8-experts currently requires --fuse-gate-up-exps")

    def should_preserve_fp8(self, name: str) -> bool:
        return self._preserve_fp8_experts and ".mlp.experts." in name and name.endswith(
            (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"))

    def _read_hash_constants(self, suffix: str) -> list[int]:
        """Read an int64 PLE constant straight from the checkpoint.

        prepare_tensors() casts every non-float dtype to float32 before
        modify_tensors() sees it (base.py), which would silently round these
        45-bit multipliers. Reading the lazy tensor here bypasses that.
        """
        for name, gen in self.model_tensors.items():
            if name.endswith(suffix):
                t = gen()
                if t.dtype != torch.int64:
                    t = t.to(torch.int64)
                return [int(x) for x in t.tolist()]
        raise ValueError(f"PLE constant {suffix!r} missing from the checkpoint")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        self.gguf_writer.add_hyper_connection_count(hp["hc_count"])
        self.gguf_writer.add_hyper_connection_low_rank(hp["hc_lowrank"])

        n_layer = hp["num_hidden_layers"]
        self.gguf_writer.add_indexer_head_count(hp["indexer_n_heads"])
        self.gguf_writer.add_indexer_key_length(hp["indexer_head_dim"])
        self.gguf_writer.add_indexer_top_k(hp["indexer_budget"])
        ratio = hp["indexer_compress_ratio"]
        layer_types = hp["layer_types"]
        self.gguf_writer.add_attention_compress_ratios(
            [ratio if layer_types[i] == "full_attention" else 0 for i in range(n_layer)]
        )

        # ple_layer_ids is 1-based in the HF config; empty means no n-gram table,
        # so emit no PLE keys rather than optional ones
        ple_layers = [i - 1 for i in hp["ple_layer_ids"]]
        if not ple_layers:
            return
        self.gguf_writer.add_ple_layers(ple_layers)
        self.gguf_writer.add_ple_ngram_size(hp["ngram_size"])
        self.gguf_writer.add_ple_heads_per_ngram(hp["heads_per_ngram"])
        self.gguf_writer.add_ple_conv_kernel(hp["ple_conv_kernel_size"])
        self.gguf_writer.add_ple_eos_token_id(self._eos_token_id())
        # an image is decoded as an embeddings-only batch, so the graph has no placeholder
        # ids to hash; carry the id and let it stand in for those positions
        _img = self._image_token_id()
        if _img is not None:
            self.gguf_writer.add_ple_image_token_id(int(_img))
        if self._ple_row_dim is not None:
            self.gguf_writer.add_embedding_length_per_layer_input(self._ple_row_dim)

        self.gguf_writer.add_ple_layer_multipliers(
            self._read_hash_constants("ple_embedding.layer_multipliers"))
        self.gguf_writer.add_ple_head_offsets(
            self._read_hash_constants("ple_embedding.ngram_heads_offsets"))
        self.gguf_writer.add_ple_head_vocab_sizes(
            self._read_hash_constants("ple_embedding.ngram_heads_vocab_sizes"))

    def _image_token_id(self) -> int | None:
        img = self.hparams.get("image_token_id")
        return None if img is None else int(img)

    def _eos_token_id(self) -> int:
        eos = self.hparams.get("eos_token_id")
        if isinstance(eos, list):
            # the PLE hash resets n-grams on the primary EOS
            return int(eos[-1])
        if eos is None:
            raise ValueError("eos_token_id is required: the PLE hash resets its n-grams on it")
        return int(eos)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if self._preserve_fp8_experts and ".mlp.experts." in name:
            if bid is None:
                raise ValueError(f"missing layer id for expert tensor {name}")
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            if self._fp8_expert_buffer is None:
                self._fp8_expert_buffer = [{} for _ in range(self.block_count)]
            buffer = self._fp8_expert_buffer[bid]
            buffer[name] = data_torch

            if len(buffer) < n_experts * 6:
                return []
            if len(buffer) != n_experts * 6:
                raise ValueError(f"unexpected FP8 expert tensor count in layer {bid}: {len(buffer)}")

            weights: dict[str, Tensor] = {}
            scales: dict[str, Tensor] = {}
            for proj in ("gate_proj", "up_proj", "down_proj"):
                proj_weights = []
                proj_scales = []
                for xid in range(n_experts):
                    base = f"model.layers.{bid}.mlp.experts.{xid}.{proj}.weight"
                    try:
                        weight = buffer.pop(base)
                        scale = buffer.pop(base + "_scale_inv")
                    except KeyError as exc:
                        raise ValueError(f"missing FP8 expert tensor {exc.args[0]}") from exc
                    if weight.dtype != torch.float8_e4m3fn:
                        raise ValueError(f"{base} is {weight.dtype}, expected torch.float8_e4m3fn")
                    proj_weights.append(weight)
                    proj_scales.append(scale)
                weights[proj] = torch.stack(proj_weights, dim=0)
                scales[proj] = torch.stack(proj_scales, dim=0)

            gate_up = torch.cat([weights["gate_proj"], weights["up_proj"]], dim=1)
            gate_up_scale = torch.cat([scales["gate_proj"], scales["up_proj"]], dim=1)
            gate_up_name = self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_UP_EXP, bid)
            down_name = self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN_EXP, bid)
            self._fp8_gguf_weights.update((gate_up_name, down_name))
            return [
                (gate_up_name, gate_up.contiguous().view(torch.uint8)),
                (gate_up_name.removesuffix(".weight") + ".scale", gate_up_scale.contiguous()),
                (down_name, weights["down_proj"].contiguous().view(torch.uint8)),
                (down_name.removesuffix(".weight") + ".scale", scales["down_proj"].contiguous()),
            ]

        # int64 hash constants must stay exact; 1-D tensors force F32, so use KV
        if name.endswith("ple_embedding.layer_multipliers"):
            self._ple_multipliers = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_offsets"):
            self._ple_head_offsets = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_vocab_sizes"):
            self._ple_head_vocab_sizes = [int(x) for x in data_torch.tolist()]
            return []

        # The official FP8 checkpoint stores the PLE table as E4M3 shards with
        # one global scale. PLE is gathered on the CPU, so dequantize lazily
        # per shard and write Q8_0 rather than keeping a CUDA-only raw FP8 type.
        if name.endswith("ple_embedding.ngram_embedding.weight_scale"):
            if data_torch.numel() != 1:
                raise ValueError(f"PLE weight scale must be scalar, got shape {tuple(data_torch.shape)}")
            from .base import LazyTorchTensor
            scale = LazyTorchTensor.to_eager(data_torch)
            self._ple_weight_scale = float(scale.float().item())
            return []

        if ".ngram_embedding.shard_" in name:
            return self._place_ple_shard(data_torch, name)

        # one projection feeds indexer q and k; split it, as minimax-m3 does
        if ".indexer.index_qk_proj.weight" in name:
            n_q = self.hparams["indexer_n_heads"] * self.hparams["indexer_head_dim"]
            q = data_torch[:n_q]
            k = data_torch[n_q:]
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_PROJ, bid, ".weight"), q),
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_K_PROJ, bid, ".weight"), k),
            ]

        # Gemma zero-centred gammas the inherited norm.weight rule misses
        if name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight",
                          ".indexer.q_layernorm.weight", ".indexer.k_layernorm.weight")):
            return [(self.map_tensor_name(name), data_torch + 1)]

        if name.endswith(".ple.conv1d.weight"):
            return [(self.map_tensor_name(name), data_torch.squeeze())]

        return super().modify_tensors(data_torch, name, bid)

    # the shards concatenate into a tensor of well over 100 GB
    # use LazyChunkedTensor here, a single shard resident at a time
    def _place_ple_shard(self, data_torch: Tensor, name: str) -> Iterable[tuple[str, Tensor]]:

        idx = int(name.rpartition(".shard_")[2].partition(".")[0])
        n_parts = self.hparams["split_ngram_parts"]

        self._ple_shards[idx] = name
        self._ple_row_dim = int(data_torch.shape[-1])

        if len(self._ple_shards) < n_parts:
            return []

        # the checkpoint may yield the shards in any order, the row order is by index
        shards = [self._ple_shards[i] for i in sorted(self._ple_shards)]
        rows = 0
        for shard in shards:
            shape = self.model_tensors[shard]().shape
            if int(shape[-1]) != self._ple_row_dim:
                raise ValueError(
                    f"PLE shard {shard} has row dim {int(shape[-1])}, expected {self._ple_row_dim}")
            rows += int(shape[0])

        table = gguf.LazyChunkedTensor(
            [self._load_ple_shard(shard) for shard in shards],
            shape=(rows, self._ple_row_dim),
            dtype=np.float32,
        )
        gguf_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD]
        return [(gguf_name + ".weight", cast(Tensor, table))]

    def _load_ple_shard(self, name: str):
        def load() -> np.ndarray:
            from .base import LazyTorchTensor

            if self._ple_weight_scale is None:
                raise ValueError("FP8 PLE shards require ple_embedding.ngram_embedding.weight_scale")
            # a fresh lazy tensor every call, or to_eager() memoizes every shard
            eager = LazyTorchTensor.to_eager(self.model_tensors[name]())
            return eager.to(torch.float32).mul_(self._ple_weight_scale).contiguous().numpy()
        return load

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int):
        if new_name in self._fp8_gguf_weights:
            return gguf.GGMLQuantizationType.F8_E4M3
        if new_name == gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD] + ".weight":
            return gguf.GGMLQuantizationType.Q8_0
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._fp8_expert_buffer is not None:
            unprocessed = [name for layer in self._fp8_expert_buffer for name in layer]
            if unprocessed:
                raise ValueError(f"unprocessed FP8 expert tensors: {unprocessed[:8]}")
        n_parts = self.hparams.get("split_ngram_parts", 0)
        if self._ple_shards and len(self._ple_shards) != n_parts:
            raise ValueError(
                f"got {len(self._ple_shards)} PLE embedding shards, expected {n_parts}"
            )
        if self._ple_shards and self._ple_weight_scale is None:
            raise ValueError("FP8 PLE shards are missing ple_embedding.ngram_embedding.weight_scale")


@ModelBase.register("Qwen4ExpForConditionalGeneration")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpVisionModel(Qwen3VLVisionModel):
    """The vision tower is an unmodified Qwen3-VL ViT."""
