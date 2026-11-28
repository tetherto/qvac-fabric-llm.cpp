"""Export contract checks that do not import ML dependencies or load weights."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import export_coreml


def source_config() -> dict:
    return {
        "model_type": "embedding_gemma2",
        "text_config": {
            "hidden_size": 512, "embedding_dim": 768, "vocab_size": 262144,
            "hidden_size_per_layer_input": 512, "num_hidden_layers": 24,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 256,
            "intermediate_size": 2048, "sliding_window": 512, "rms_norm_eps": 1e-6,
            "hidden_activation": "gelu_pytorch_tanh", "attention_bias": False,
            "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0,
            "layer_types": ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(24)],
            "per_layer_config": {f"{i:02d}": {"head_dim": 512, "num_key_value_heads": 1} for i in (5, 11, 17, 23)},
            "rope_parameters": {
                "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            },
        },
    }


class TestExportContract(unittest.TestCase):
    def test_reject_invalid_converted_output(self):
        contracts = export_coreml.function_contracts(16)

        def feature(name, shape, dtype=1):
            return SimpleNamespace(name=name, type=SimpleNamespace(multiArrayType=SimpleNamespace(shape=shape, dataType=dtype)))

        functions = [SimpleNamespace(
            name=name, input=[feature(key, shape) for key, shape in shapes.items()],
            output=[feature("embedding", (16 if name == "pack_256" else 1, 768)),
                    feature("embedding_raw", (16 if name == "pack_256" else 1, 768), 2)],
        ) for name, shapes in contracts.items()]
        description = SimpleNamespace(functions=functions)
        export_coreml.validate_functions(description, contracts, 16, 1, 2)
        functions[-1].output[0].type.multiArrayType.shape = (8, 768)
        with self.assertRaisesRegex(ValueError, "embedding shape or type"):
            export_coreml.validate_functions(description, contracts, 16, 1, 2)
        functions[-1].output[0].type.multiArrayType.shape = (16, 768)
        functions[-1].output[1].type.multiArrayType.dataType = 1
        with self.assertRaisesRegex(ValueError, "embedding shape or type"):
            export_coreml.validate_functions(description, contracts, 16, 1, 2)

    def test_adapter_shapes(self):
        for slots in (8, 16):
            functions = export_coreml.function_contracts(slots)
            self.assertEqual(len(functions), 7)
            for length in (32, 48, 64, 128, 256, 512):
                self.assertEqual(functions[f"embed_{length}"], {
                    "inputs_embeds": (1, length, 512), "attention_mask": (1, length),
                })
            self.assertEqual(functions["pack_256"], {
                "inputs_embeds": (1, 256, 512), "attention_bias": (1, 1, 256, 256),
                "positions": (256, 1), "pool": (slots, 256),
            })
        with self.assertRaises(ValueError):
            export_coreml.function_contracts(32)

    def test_reject_incompatible_checkpoint(self):
        valid = source_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "config.json"
            path.write_text(json.dumps(valid))
            self.assertEqual(export_coreml.read_config(root), valid["text_config"])
            mutations = {
                "hidden_size": 1024, "embedding_dim": 256, "vocab_size": 262145,
                "num_hidden_layers": 25, "sliding_window": 128, "bos_token_id": 3,
                "layer_types": ["full_attention"] * 24, "per_layer_config": {}, "rope_parameters": {},
            }
            for key, value in mutations.items():
                with self.subTest(key=key):
                    changed = copy.deepcopy(valid)
                    changed["text_config"][key] = value
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        export_coreml.read_config(root)
            changed = copy.deepcopy(valid)
            del changed["text_config"]["hidden_size"]
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                export_coreml.read_config(root)

    def test_select_only_text_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "text.safetensors").touch()
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                "language_model.embed_tokens.weight": "text.safetensors",
                "language_model.norm.weight": "text.safetensors",
                "vision_tower.weight": "missing-vision.safetensors",
            }}))
            self.assertEqual(export_coreml.checkpoint_files(root), [root / "text.safetensors"])

    def test_reject_unsafe_or_missing_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("../outside.safetensors", "/tmp/outside.safetensors", "weights.bin", "missing.safetensors"):
                with self.subTest(name=name):
                    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                        "language_model.norm.weight": name,
                    }}))
                    with self.assertRaises(ValueError):
                        export_coreml.checkpoint_files(root)

    def test_refuse_existing_destination_before_importing_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(source_config()))
            (root / "model.safetensors").touch()
            (root / "tokenizer.json").write_text("{}")
            output = root / "output"
            output.mkdir()
            sentinel = output / "keep"
            sentinel.write_text("existing assets")
            with self.assertRaisesRegex(ValueError, "Destination already exists"):
                export_coreml.export(root, output, 8)
            self.assertEqual(sentinel.read_text(), "existing assets")


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires the export environment")
class TestNormalization(unittest.TestCase):
    def test_fp16_large_activations_match_fp32(self):
        import torch

        values = torch.tensor([[64000.0, -32000.0, 2000.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float16)
        reference = values.float()
        reference = reference * torch.rsqrt(reference.square().mean(dim=-1, keepdim=True) + 1e-6)
        actual = export_coreml.rms_normalize(values, 1e-6)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.float(), reference, atol=0.002, rtol=0.002)

    def test_l2_normalization_handles_unused_slots(self):
        import torch

        values = torch.tensor([[64000.0, -32000.0, 2000.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float16)
        reference = torch.nn.functional.normalize(values.float(), dim=-1)
        actual = export_coreml.l2_normalize(values)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.float(), reference, atol=0.002, rtol=0.002)


if __name__ == "__main__":
    unittest.main()
