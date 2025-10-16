from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    import torch

from .base import ModelBase, TextModel, gguf, LazyTorchTensor


@ModelBase.register("BitnetForCausalLM", "BitNetForCausalLM")
@ModelBase.example("microsoft/bitnet-b1.58-2B-4T")
class BitnetModel(TextModel):
    model_arch = gguf.MODEL_ARCH.BITNET

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bitnet_weight_scales: dict[str, torch.Tensor] = {}

    def set_vocab(self):
        if (self.dir_model / "tokenizer.model").is_file():
            self._set_vocab_sentencepiece()
        else:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        self.gguf_writer.add_rope_scaling_factor(1.0)

    @staticmethod
    def _unpack_bitnet_weights(packed: torch.Tensor) -> torch.Tensor:
        if packed.dtype != torch.uint8:
            raise ValueError(f"Expected packed BitNet weights to be torch.uint8, got {packed.dtype}")

        values_per_item = 4
        rows = packed.shape[0]
        rest = packed.shape[1:]

        unpacked_chunks: list[torch.Tensor] = []
        mapping = torch.tensor([-1.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=packed.device)

        for i in range(values_per_item):
            chunk = (packed >> (2 * i)) & 0x03
            chunk = mapping[chunk.long()].reshape((rows, *rest))
            unpacked_chunks.append(chunk)

        if not unpacked_chunks:
            raise ValueError("Failed to unpack BitNet weights: no chunks produced")

        return torch.cat(unpacked_chunks, dim=0)

    def weight_quant(self, weight: Tensor) -> Tensor:
        dtype = weight.dtype
        weight = weight.float()
        scale = weight.abs().mean().clamp(min=1e-5)
        iscale = 1 / scale
        # TODO: multiply by the scale directly instead of inverting it twice
        # (this is also unnecessarily doubly inverted upstream)
        # ref: https://huggingface.co/1bitLLM/bitnet_b1_58-3B/blob/af89e318d78a70802061246bf037199d2fb97020/utils_quant.py#L10
        result = (weight * iscale).round().clamp(-1, 1) / iscale
        return result.type(dtype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        weight_scale_suffix = ".weight_scale"
        if name.endswith(weight_scale_suffix):
            weight_name = name[:-len(weight_scale_suffix)] + ".weight"
            mapped_weight_name = self.map_tensor_name(weight_name)
            if isinstance(data_torch, LazyTorchTensor):
                data_torch = LazyTorchTensor.to_eager(data_torch)

            scale_tensor = data_torch.to(torch.float32)
            self._bitnet_weight_scales[mapped_weight_name] = scale_tensor
            return []

        new_name = self.map_tensor_name(name)

        ternary_weight = False

        if name.endswith(".weight"):
            scale_tensor = self._bitnet_weight_scales.pop(new_name, None)
            if scale_tensor is not None:
                scale_tensor = scale_tensor.to(torch.float32)
                if scale_tensor.numel() != 1:
                    raise ValueError(f"Expected scalar weight_scale for '{name}', got shape {tuple(scale_tensor.shape)}")

                if isinstance(data_torch, LazyTorchTensor):
                    data_torch = LazyTorchTensor.to_eager(data_torch)

                packed = data_torch.to(torch.uint8)
                unpacked = self._unpack_bitnet_weights(packed)
                scale_value = scale_tensor.reshape(-1)[0].item()
                data_torch = unpacked * scale_value
                ternary_weight = True

        if any(self.match_model_tensor_name(new_name, key, bid) for key in [
            gguf.MODEL_TENSOR.ATTN_Q,
            gguf.MODEL_TENSOR.ATTN_K,
            gguf.MODEL_TENSOR.ATTN_V,
            gguf.MODEL_TENSOR.ATTN_OUT,
            gguf.MODEL_TENSOR.FFN_UP,
            gguf.MODEL_TENSOR.FFN_DOWN,
            gguf.MODEL_TENSOR.FFN_GATE,
        ]) and not ternary_weight:
            # transform weight into 1/0/-1 (in fp32)
            data_torch = self.weight_quant(data_torch)

        yield from super().modify_tensors(data_torch, name, bid)
