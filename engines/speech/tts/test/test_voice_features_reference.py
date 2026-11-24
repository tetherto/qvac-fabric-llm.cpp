"""Check generated WAV/mel pairing through the actual C++ parity harness."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import gguf
    import librosa
    import numpy as np
    from scipy.io import wavfile
    import torch
except ImportError:
    print("SKIP: gguf, librosa, numpy, scipy and torch required")
    raise SystemExit(77)

BINARY = sys.argv.pop(1)
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dump-voice-features-reference.py"
spec = importlib.util.spec_from_file_location("voice_features_reference", SCRIPT)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

INPUT_SAMPLE_RATE = 16000
INPUT_SAMPLES = 4096
PCM_AMPLITUDE = 12000


class VoiceFeaturesReferenceTests(unittest.TestCase):
    def run_harness(self, output):
        return subprocess.run(
            [BINARY, str(output / "filterbank.gguf"), str(output / "reference.wav"),
             str(output / "prompt_feat.npy")],
            capture_output=True, text=True, timeout=30,
        )

    def test_matched_and_mismatched_references(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            samples = np.random.default_rng(7).integers(
                -PCM_AMPLITUDE, PCM_AMPLITUDE, size=INPUT_SAMPLES, dtype=np.int16,
            )
            source = output / "source.wav"
            wavfile.write(source, INPUT_SAMPLE_RATE, samples)
            reference.dump_reference(source, output)
            rate, saved = wavfile.read(output / "reference.wav")
            self.assertEqual(rate, reference.SAMPLE_RATE)
            self.assertEqual(saved.dtype, np.float32)
            self.assertEqual(saved.size, INPUT_SAMPLES * rate // INPUT_SAMPLE_RATE)
            result = self.run_harness(output)
            self.assertEqual(result.returncode, 0, result.stderr)

            # A same-shaped reference from different audio must be rejected.
            other = output / "other"
            wavfile.write(source, INPUT_SAMPLE_RATE, samples[::-1])
            reference.dump_reference(source, other)
            np.save(output / "prompt_feat.npy", np.load(other / "prompt_feat.npy"))
            result = self.run_harness(output)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("FAIL [prompt_feat]", result.stderr)


if __name__ == "__main__":
    unittest.main()
