from typing import Any

import pytest
import torch

from conversion.qwen import DSparkModel


@pytest.mark.parametrize("suffix", (
    "q_proj.weight", "k_proj.weight", "q_norm.weight", "k_norm.weight",
))
def test_dspark_interleaved_rope_permuted_once(suffix):
    model: Any = object.__new__(DSparkModel)
    model.hparams = {"rope_is_neox_style": False, "head_dim": 8, "vocab_size": 16}
    model._n_vocab_draft = 16
    model.is_rerank = False
    model.hf_arch = "DSparkDraftModel"
    model.fuse_gate_up_exps = False
    model.fuse_qkv = False
    model.map_tensor_name = lambda name, try_suffixes=(".weight", ".bias"): name

    weight = torch.arange(8)
    name = f"model.layers.0.self_attn.{suffix}"
    converted = list(model.modify_tensors(weight, name, 0))

    assert len(converted) == 1
    assert converted[0][0] == name
    assert converted[0][1].tolist() == [0, 2, 4, 6, 1, 3, 5, 7]
