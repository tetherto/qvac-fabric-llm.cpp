import json
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest

from conversion.base import MmprojModel, ModelBase, TextModel, gguf


class FakeWriter:
    def __init__(self):
        self.values = {}

    def add_uint32(self, key, value):
        self.values[key] = value

    def add_string(self, key, value):
        self.values[key] = value

    def add_array(self, key, value):
        self.values[key] = value

    def add_bool(self, key, value):
        self.values[key] = value

    def get_total_parameter_count(self):
        return 0, 0, 0, 0


def write_manifest(path: Path, tensors, block_size=256):
    manifest = {
        "schema_version": 1,
        "kind": "hadamard-weight-fold",
        "status": "requires-matching-runtime",
        "transform": {
            "name": "normalized-signed-sylvester-walsh-hadamard",
            "block_size": block_size,
            "sign_mode": "identity",
        },
        "tensors": tensors,
    }
    (path / "hadamard_packing.json").write_text(json.dumps(manifest), encoding="utf-8")


def make_model(tmp_path: Path, arch, tensors, fuse=False, block_size=256) -> Any:
    write_manifest(tmp_path, tensors, block_size)
    model = object.__new__(ModelBase)
    model.dir_model = tmp_path
    model.model_arch = arch
    model.fuse_gate_up_exps = fuse
    model.gguf_writer = FakeWriter()
    model.filter_tensors = lambda item: item
    model.map_tensor_name = lambda name, try_suffixes=(".weight", ".bias"): name
    model._hadamard_gdn_v_grouped = False
    return model


def folded(name):
    return {"name": name, "axis": -1, "role": "fold-before-matmul"}


def inverse(name):
    return {"name": name, "axis": -1, "role": "inverse-after-lookup"}


def test_hadamard_metadata_records_forward_and_inverse(tmp_path):
    model = make_model(
        tmp_path,
        gguf.MODEL_ARCH.QWEN35,
        [folded("output.weight"), inverse("token_embd.weight")],
    )

    model.add_hadamard_metadata()

    assert model.gguf_writer.values["prism.hadamard.weight_names"] == ["output.weight"]
    assert model.gguf_writer.values["prism.hadamard.inverse_weight_names"] == ["token_embd.weight"]


def test_hadamard_metadata_uses_fused_expert_name(tmp_path):
    model = make_model(
        tmp_path,
        gguf.MODEL_ARCH.QWEN35MOE,
        [folded("blk.0.ffn_up_exps.weight"), folded("blk.0.ffn_gate_exps.weight")],
        fuse=True,
    )

    model.add_hadamard_metadata()

    assert model.gguf_writer.values["prism.hadamard.weight_names"] == ["blk.0.ffn_gate_up_exps.weight"]


@pytest.mark.parametrize(
    "tensors, message",
    [
        ([folded("blk.0.ffn_gate_exps.weight")], "requires both"),
        (
            [
                folded("blk.0.ffn_gate_exps.weight"),
                folded("blk.0.ffn_up_exps.weight"),
                folded("blk.0.ffn_gate_up_exps.weight"),
            ],
            "mixes fused and split",
        ),
        (
            [folded("blk.0.ffn_gate_exps.weight"), folded("blk.0.ffn_gate_exps.weight")],
            "duplicate Hadamard split",
        ),
    ],
)
def test_hadamard_metadata_rejects_unsafe_fusion(tmp_path, tensors, message):
    model = make_model(tmp_path, gguf.MODEL_ARCH.QWEN35MOE, tensors, fuse=True)

    with pytest.raises(ValueError, match=message):
        model.add_hadamard_metadata()


def test_hadamard_metadata_rejects_qwen3_moe(tmp_path):
    model = make_model(tmp_path, gguf.MODEL_ARCH.QWEN3MOE, [folded("output.weight")])

    with pytest.raises(ValueError, match="not verified"):
        model.add_hadamard_metadata()


def test_hadamard_metadata_rejects_oversized_block(tmp_path):
    model = make_model(tmp_path, gguf.MODEL_ARCH.QWEN35, [folded("output.weight")], block_size=16384)

    with pytest.raises(ValueError, match="maximum is 8192"):
        model.add_hadamard_metadata()


def test_hadamard_metadata_runs_only_for_full_text_conversion(tmp_path, monkeypatch):
    monkeypatch.setattr(ModelBase, "prepare_metadata", lambda self, vocab_only: None)

    text: Any = object.__new__(TextModel)
    text.gguf_writer = FakeWriter()
    text.ftype = SimpleNamespace(name="MOSTLY_F16")
    text.fname_out = tmp_path / "model.gguf"
    text.set_vocab = lambda: None
    text_calls = []
    text.add_hadamard_metadata = lambda: text_calls.append(True)

    TextModel.prepare_metadata(text, vocab_only=True)
    assert text_calls == []
    TextModel.prepare_metadata(text, vocab_only=False)
    assert text_calls == [True]

    mmproj: Any = object.__new__(MmprojModel)
    mmproj.ftype = SimpleNamespace(name="MOSTLY_F16")
    mmproj.fname_out = tmp_path / "mmproj.gguf"
    mmproj_calls = []
    mmproj.add_hadamard_metadata = lambda: mmproj_calls.append(True)

    MmprojModel.prepare_metadata(mmproj, vocab_only=False)
    assert mmproj_calls == []
