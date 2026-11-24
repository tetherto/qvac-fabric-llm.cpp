import sys
import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as error:
    print(f"SKIP: {error.name} is not installed")
    sys.exit(77)


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS_DIR / "dump-unified-reference.py"
SPEC = importlib.util.spec_from_file_location("dump_unified_reference", SCRIPT)
DUMPER = importlib.util.module_from_spec(SPEC)
try:
    SPEC.loader.exec_module(DUMPER)
except ModuleNotFoundError as error:
    print(f"SKIP: {error.name} is not installed")
    sys.exit(77)

SAMPLES_PER_FRAME = DUMPER.SAMPLES_PER_FRAME


class Hypothesis:
    def __init__(self, text, token_ids):
        self.text = text
        self.y_sequence = torch.tensor(token_ids)


class StepWindowTest(unittest.TestCase):
    def test_first_window_has_no_left_context(self):
        windows = DUMPER.step_windows(40 * SAMPLES_PER_FRAME, left=70, chunk=7, right=7)

        self.assertEqual(windows[0]["begin"], 0)
        self.assertEqual(windows[0]["left_frames"], 0)
        self.assertEqual(windows[0]["chunk_frames"], 7)
        self.assertEqual(windows[0]["end"], 14 * SAMPLES_PER_FRAME)

    def test_steady_state_window_carries_full_left_and_right_context(self):
        windows = DUMPER.step_windows(200 * SAMPLES_PER_FRAME, left=70, chunk=7, right=7)
        window = windows[12]

        self.assertEqual(window["left_frames"], 70)
        self.assertEqual(window["begin"], (12 * 7 - 70) * SAMPLES_PER_FRAME)
        self.assertEqual(window["end"], (12 * 7 + 14) * SAMPLES_PER_FRAME)
        self.assertEqual(window["chunk_frames"], 7)

    def test_windows_tile_the_audio_without_gaps(self):
        n_frames = 45
        windows = DUMPER.step_windows(n_frames * SAMPLES_PER_FRAME, left=70, chunk=7, right=7)

        self.assertEqual(sum(w["chunk_frames"] for w in windows), n_frames)
        self.assertEqual(windows[-1]["end"], n_frames * SAMPLES_PER_FRAME)

    def test_partial_last_chunk_is_rounded_up_to_a_frame(self):
        n_samples = 10 * SAMPLES_PER_FRAME + 1
        windows = DUMPER.step_windows(n_samples, left=70, chunk=7, right=7)

        self.assertEqual(windows[-1]["chunk_frames"], 4)
        self.assertEqual(windows[-1]["end"], n_samples)

    def test_right_context_is_clipped_at_the_end_of_audio(self):
        windows = DUMPER.step_windows(20 * SAMPLES_PER_FRAME, left=70, chunk=7, right=7)

        self.assertEqual(windows[-1]["end"], 20 * SAMPLES_PER_FRAME)


class HypothesisHelperTest(unittest.TestCase):
    def test_text_and_tokens_are_extracted(self):
        hypothesis = Hypothesis("hello", [4, 5, 6])

        self.assertEqual(DUMPER.hypothesis_text(hypothesis), "hello")
        self.assertEqual(DUMPER.hypothesis_tokens(hypothesis), [4, 5, 6])

    def test_plain_string_hypothesis_has_no_tokens(self):
        self.assertEqual(DUMPER.hypothesis_text("hello"), "hello")
        self.assertEqual(DUMPER.hypothesis_tokens("hello"), [])


if __name__ == "__main__":
    unittest.main()
