import sys
import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ModuleNotFoundError as error:
    print(f"SKIP: {error.name} is not installed")
    sys.exit(77)



SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS_DIR / "dump-nemotron-reference.py"
SPEC = importlib.util.spec_from_file_location(
    "dump_nemotron_reference",
    SCRIPT,
)
DUMPER = importlib.util.module_from_spec(SPEC)
try:
    SPEC.loader.exec_module(DUMPER)
except ModuleNotFoundError as error:
    print(f"SKIP: {error.name} is not installed")
    sys.exit(77)


class Hypothesis:
    def __init__(self, text, token_ids):
        self.text = text
        self.y_sequence = torch.tensor(token_ids)


class DumpNemotronReferenceTest(unittest.TestCase):
    def test_supported_contexts_map_to_public_chunk_sizes(self):
        self.assertEqual(
            DUMPER.RIGHT_CONTEXT_TO_CHUNK_MS,
            {
                0: 80,
                1: 160,
                3: 320,
                6: 560,
                13: 1120,
            },
        )

    def test_positive_int_rejects_zero(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            DUMPER.positive_int("0")

    def test_prompt_tensors_split_encoder_and_one_hot_features(self):
        prompt_input = torch.zeros((1, 2, 1152))
        prompt_input[..., 1024 + 10] = 1.0
        prompt_output = torch.zeros((1, 2, 1024))

        encoder, prompt, output = DUMPER.prompt_tensors(
            {
                "input": prompt_input,
                "output": prompt_output,
            },
        )

        self.assertEqual(tuple(encoder.shape), (1, 2, 1024))
        self.assertEqual(tuple(prompt.shape), (1, 2, 128))
        self.assertEqual(tuple(output.shape), (1, 2, 1024))
        self.assertTrue(torch.equal(prompt[..., 10], torch.ones((1, 2))))

    def test_hypothesis_helpers_extract_text_and_tokens(self):
        hypotheses = [Hypothesis("hello", [4, 5, 6])]

        self.assertEqual(
            DUMPER.extract_transcriptions(hypotheses),
            ["hello"],
        )
        self.assertTrue(
            torch.equal(
                DUMPER.hypothesis_token_ids(hypotheses),
                torch.tensor([4, 5, 6], dtype=torch.int32),
            ),
        )

    def test_save_tensor_writes_c_contiguous_npy(self):
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tensor.npy"
            DUMPER.save_tensor(path, tensor)
            saved = np.load(path)

        self.assertTrue(saved.flags.c_contiguous)
        self.assertTrue(np.array_equal(saved, tensor.numpy()))


if __name__ == "__main__":
    unittest.main()
