import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class QualitySelectionTests(unittest.TestCase):
    executable = None

    def test_q4_rejects_other_precisions_before_loading(self):
        for precisions in [("f16",), ("q8_0",), ("q4_k_m", "q8_0")]:
            with self.subTest(precisions=precisions), tempfile.TemporaryDirectory() as directory:
                for precision in precisions:
                    for role in ("lm", "synth"):
                        (Path(directory) / f"mm3-{role}-{precision}.gguf").touch()
                environment = dict(os.environ, AUDIOGEN_TEST_MINIMAX_Q4_MODELS_DIR=directory)
                result = subprocess.run([str(self.executable), "--q4"], env=environment,
                                        capture_output=True, text=True, check=False)
                selected = "q8_0" if "q8_0" in precisions else "f16"
                self.assertEqual(result.returncode, 1)
                self.assertIn("FAIL --q4 selected " + selected, result.stderr)
                self.assertNotIn("FAIL mm3_load", result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    args = parser.parse_args()
    QualitySelectionTests.executable = args.executable.resolve(strict=True)
    unittest.main(argv=[sys.argv[0]])
