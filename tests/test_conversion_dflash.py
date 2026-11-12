import copy
import json
from typing import Any

import gguf
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


DSPINE_LAYERS = 5
DSPINE_BRIDGE_EPS = 1e-06

# the DSpine keys of the released tencent/Qwen3-8B-DSpine-b16 config
DSPINE_CONFIG = {
    "draft_block_causal": True,
    "selected_prefix_code_bridge": {"code_size": 1024, "eps": DSPINE_BRIDGE_EPS, "kappa": 10.0, "rank": 512},
    "selected_prefix_decode_mode": "serial",
    "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
    "selected_prefix_feature_write_mode": "independent",
    "selected_prefix_feature_writes": True,
    "selected_prefix_last_write_refine": {"head_topk": 16, "kind": "exact_sender", "version": 1, "weight": 1.0},
    "selected_prefix_terminal": False,
}

# tencent/Qwen3.8-27B-DSpine-b16 also sets is_causal and a second pass of the last layer
DSPINE_27B_HPARAMS = {"is_causal": True, "code_bridge_second_pass": {"first_exact": 4, "head_topk": 16, "mode": "serial_table"}}


def dspine_model(config: dict, extra_hparams: dict | None = None) -> Any:
    model: Any = object.__new__(DFlashModel)
    model.hparams = {"dflash_config": config, "vocab_size": 16, **(extra_hparams or {})}
    model.block_count = DSPINE_LAYERS
    model.rope_parameters = {"rope_type": "default"}
    model.is_rerank = False
    model.hf_arch = "DFlashDraftModel"
    model.fuse_gate_up_exps = False
    model.fuse_qkv = False
    model.tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.DFLASH, DSPINE_LAYERS)
    return model


def with_config(**changes) -> dict:
    config = copy.deepcopy(DSPINE_CONFIG)
    config.update(changes)
    return config


def converted(model: Any, name: str, data: torch.Tensor) -> dict[str, torch.Tensor]:
    return dict(model.modify_tensors(data, name, None))


@pytest.mark.parametrize("extra_hparams", ({}, DSPINE_27B_HPARAMS))
def test_dspine_released_configs(extra_hparams):
    assert dspine_model(DSPINE_CONFIG, extra_hparams)._dspine_head_topk() == 16


class RecordingWriter:
    """Records every add_* call, so a test checks only the keys it names."""

    def __init__(self):
        self.calls: dict[str, list[tuple]] = {}

    def __getattr__(self, name: str):
        if not name.startswith("add_"):
            raise AttributeError(name)
        return lambda *args: self.calls.setdefault(name, []).append(args)


@pytest.mark.parametrize("extra_hparams", ({}, DSPINE_27B_HPARAMS))
def test_dspine_writes_bridge_metadata(monkeypatch, extra_hparams):
    monkeypatch.setattr(Qwen3Model, "set_gguf_parameters", lambda _self: None)
    model = dspine_model(DSPINE_CONFIG, extra_hparams)
    model.target_model_dir = None
    model.gguf_writer = RecordingWriter()

    model.set_gguf_parameters()

    calls = model.gguf_writer.calls
    assert calls["add_dspine_head_topk"] == [(16,)]
    assert calls["add_dspine_eps"] == [(DSPINE_BRIDGE_EPS,)]
    assert calls["add_causal_attention"] == [(True,)]


@pytest.mark.parametrize("config, extra_hparams", (
    (with_config(selected_prefix_terminal=True), {}),
    (with_config(selected_prefix_decode_mode="parallel"), {}),
    (with_config(selected_prefix_feature_layers=[1, 2, 3]), {}),
    (with_config(selected_prefix_feature_write_mode="shared"), {}),
    (with_config(draft_block_causal=False), {}),
    (with_config(selector_top_k=16), {}),
    (with_config(selected_prefix_last_write_refine={"head_topk": 16, "kind": "soft_sender", "version": 1}), {}),
    (with_config(selected_prefix_last_write_refine={"head_topk": 12, "kind": "exact_sender", "version": 1}), {}),
    (DSPINE_CONFIG, {"code_bridge_second_pass": {"first_exact": 3, "head_topk": 16, "mode": "serial_table"}}),
    (DSPINE_CONFIG, {"code_bridge_second_pass": {"first_exact": 4, "head_topk": 8, "mode": "serial_table"}}),
    (with_config(selected_prefix_last_write_refine=None), {}),
))
def test_dspine_rejects_unsupported_configs(config, extra_hparams):
    with pytest.raises(ValueError, match="unsupported DSpine draft"):
        dspine_model(config, extra_hparams)._dspine_head_topk()


def test_dspine_splits_stacked_bridge_weights():
    model = dspine_model(DSPINE_CONFIG)
    early = torch.arange(float((DSPINE_LAYERS - 1) * 6)).reshape(DSPINE_LAYERS - 1, 2, 3)
    writes = torch.arange(float(DSPINE_LAYERS * 6)).reshape(DSPINE_LAYERS, 3, 2)
    last = torch.full((2, 3), -1.0)

    reads = converted(model, "model.selected_prefix_bridge.early_read_weights", early)
    reads |= converted(model, "model.selected_prefix_bridge.read_weight", last)
    written = converted(model, "model.selected_prefix_bridge.feature_write_weights", writes)

    assert sorted(reads) == [f"blk.{il}.dspine_read.weight" for il in range(DSPINE_LAYERS)]
    assert sorted(written) == [f"blk.{il}.dspine_write.weight" for il in range(DSPINE_LAYERS)]
    for il in range(DSPINE_LAYERS - 1):
        assert torch.equal(reads[f"blk.{il}.dspine_read.weight"], early[il])
    assert torch.equal(reads[f"blk.{DSPINE_LAYERS - 1}.dspine_read.weight"], last)
    for il in range(DSPINE_LAYERS):
        assert torch.equal(written[f"blk.{il}.dspine_write.weight"], writes[il])


@pytest.mark.parametrize("short, n_stack", (("early_read_weights", DSPINE_LAYERS), ("feature_write_weights", DSPINE_LAYERS - 1)))
def test_dspine_rejects_wrong_stack_depth(short, n_stack):
    with pytest.raises(ValueError, match="layers, expected"):
        converted(dspine_model(DSPINE_CONFIG), "model.selected_prefix_bridge." + short, torch.zeros(n_stack, 2, 3))


@pytest.mark.parametrize("short, gguf_name", (
    ("codebook", "dspine_codebook.weight"),
    ("gate.weight", "dspine_gate.weight"),
    ("gate.bias", "dspine_gate.bias"),
    ("predecessor_gate.weight", "dspine_pred_gate.weight"),
    ("prev.weight", "dspine_prev.weight"),
))
def test_dspine_bridge_tensor_names(short, gguf_name):
    data = torch.ones(2, 3)
    assert list(converted(dspine_model(DSPINE_CONFIG), "model.selected_prefix_bridge." + short, data)) == [gguf_name]


def test_dspine_terminal_projection_must_be_zero():
    model = dspine_model(DSPINE_CONFIG)
    assert converted(model, "model.selected_prefix_bridge.out.weight", torch.zeros(2, 3)) == {}
    with pytest.raises(ValueError, match="terminal projection must be zero"):
        converted(model, "model.selected_prefix_bridge.out.weight", torch.ones(2, 3))


@pytest.mark.parametrize("name", ("model.draft_block_causal_version", "model.last_write_refine_version"))
def test_dspine_checkpoint_versions(name):
    model = dspine_model(DSPINE_CONFIG)
    assert converted(model, name, torch.tensor([1])) == {}
    with pytest.raises(ValueError, match="unsupported DSpine checkpoint version"):
        converted(model, name, torch.tensor([2]))
