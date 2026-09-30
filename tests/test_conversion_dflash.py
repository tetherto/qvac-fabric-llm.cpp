import json
from typing import Any

import pytest
import torch

from conversion.qwen import DFlashModel, DSparkModel, Qwen3Model


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


@pytest.mark.parametrize("target_config, uses_mrope", (
    ({"architectures": ["Qwen3_5ForCausalLM"]}, True),
    ({"architectures": ["Qwen3_5MoeForCausalLM"], "text_config": {"hidden_size": 8}}, True),
    ({"architectures": ["Qwen3ForCausalLM"], "text_config": {"architectures": ["Qwen3_5MoeForConditionalGeneration"]}}, True),
    ({"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {"architectures": ["Qwen3ForCausalLM"]}}, False),
    ({"architectures": ["Qwen3ForCausalLM"], "rope_parameters": {"mrope_section": [11, 11, 10]}}, True),
    ({"architectures": ["Qwen3ForCausalLM"], "rope_scaling": {"mrope_section": [11, 11, 10]}, "text_config": {"hidden_size": 8}}, True),
    ({"architectures": ["Qwen3ForCausalLM"]}, False),
))
def test_dflash_target_mrope_sections(tmp_path, monkeypatch, target_config, uses_mrope):
    (tmp_path / "config.json").write_text(json.dumps(target_config), encoding="utf-8")
    monkeypatch.setattr(Qwen3Model, "set_gguf_parameters", lambda _self: None)

    class Writer:
        def __init__(self):
            self.sections = []

        def add_block_size(self, _size):
            pass

        def add_rope_dimension_sections(self, sections):
            self.sections.append(sections)

    model: Any = object.__new__(DFlashModel)
    model.target_model_dir = tmp_path
    model.hparams = {"head_dim": 8}
    model.gguf_writer = Writer()

    model.set_gguf_parameters()

    assert model.gguf_writer.sections == ([[4, 0, 0, 0]] if uses_mrope else [])
