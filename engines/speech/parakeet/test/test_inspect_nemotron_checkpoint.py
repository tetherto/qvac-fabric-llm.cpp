import importlib.util
import sys
import types
import unittest
from pathlib import Path


try:
    import torch
except ModuleNotFoundError:
    torch = types.ModuleType("torch")
    sys.modules["torch"] = torch

try:
    import yaml
except ModuleNotFoundError:
    yaml = types.ModuleType("yaml")
    sys.modules["yaml"] = yaml


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS_DIR / "inspect-nemotron-checkpoint.py"
SPEC = importlib.util.spec_from_file_location(
    "inspect_nemotron_checkpoint",
    SCRIPT,
)
INSPECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECTOR)


class FakeTensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class InspectNemotronCheckpointTest(unittest.TestCase):
    def test_prompt_aliases_are_grouped_by_prompt_index(self):
        config = {
            "model_defaults": {
                "prompt_dictionary": {
                    "en-US": 0,
                    "en": 0,
                    "ja-JP": 10,
                },
            },
        }

        self.assertEqual(
            INSPECTOR.prompt_alias_groups(config),
            {
                0: ["en-US", "en"],
                10: ["ja-JP"],
            },
        )

    def test_relevant_tensor_rows_are_filtered_and_sorted(self):
        state_dict = {
            "joint.enc.weight": FakeTensor((640, 1152), "float32"),
            "encoder.layers.0.weight": FakeTensor((1024, 1024), "float16"),
            "preprocessor.window": FakeTensor((400,), "float32"),
        }

        self.assertEqual(
            INSPECTOR.relevant_tensor_rows(state_dict),
            [
                ("encoder.layers.0.weight", (1024, 1024), "float16"),
                ("joint.enc.weight", (640, 1152), "float32"),
            ],
        )

    def test_nested_value_returns_none_for_missing_path(self):
        config = {"encoder": {"d_model": 1024}}

        self.assertEqual(
            INSPECTOR.nested_value(config, ("encoder", "d_model")),
            1024,
        )
        self.assertIsNone(
            INSPECTOR.nested_value(config, ("decoder", "vocab_size")),
        )


if __name__ == "__main__":
    unittest.main()
