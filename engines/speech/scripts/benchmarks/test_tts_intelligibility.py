#!/usr/bin/env python3
"""Model-free end-to-end scorer tests using a controllable Whisper CLI stub."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave


SCORER = Path(__file__).with_name("compute-tts-intelligibility.py")


class IntelligibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tts scorer spaces ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.audio = self.root / "generated speech.wav"
        with wave.open(str(self.audio), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x00\x01" * 4800)
        self.reference = self.root / "source prompt.txt"
        self.reference.write_text("The quick brown fox.")
        self.model = self.root / "ggml tiny.bin"
        self.model.write_bytes(b"test-model")
        self.binary = self.root / "whisper cli"
        self.binary.write_text(
            "#!" + sys.executable + "\n"
            "import os, pathlib, sys, time\n"
            "mode = os.environ.get('TEST_ASR_MODE', 'ok')\n"
            "if mode == 'timeout': time.sleep(5)\n"
            "prefix = pathlib.Path(sys.argv[sys.argv.index('-of') + 1])\n"
            "if mode != 'missing':\n"
            "    prefix.with_suffix('.txt').write_text(os.environ.get('TEST_ASR_TEXT', 'The quick brown fox.'))\n"
            "print('stub ASR log')\n"
            "sys.exit(7 if mode == 'failure' else 0)\n")
        self.binary.chmod(0o755)
        self.output = self.root / "score.json"
        self.transcript = self.root / "transcript.txt"
        self.log = self.root / "asr.log"

    def run_score(self, text="The quick brown fox.", mode="ok", extra=()):
        command = [sys.executable, str(SCORER),
                   "--audio", str(self.audio), "--reference", str(self.reference),
                   "--asr-binary", str(self.binary), "--asr-model", str(self.model),
                   "--json-out", str(self.output), "--transcript-out", str(self.transcript),
                   "--log-out", str(self.log), *extra]
        run = subprocess.run(command, text=True, capture_output=True,
                             env={**os.environ, "TEST_ASR_TEXT": text, "TEST_ASR_MODE": mode},
                             timeout=10)
        self.assertIn(run.returncode, (0, 1), run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual(run.returncode == 0, result["status"] == "ok")
        if self.output.exists() and self.output != self.reference:
            self.assertEqual(result, json.loads(self.output.read_text()))
        return result

    def test_exact_transcript_spaces_and_reproducible_settings(self):
        result = self.run_score()
        self.assertEqual(result["wer"], 0)
        self.assertEqual(result["n_edits"], {"sub": 0, "ins": 0, "del": 0})
        self.assertEqual(result["asr"]["model_sha256"], hashlib.sha256(b"test-model").hexdigest())
        self.assertEqual(result["audio_metadata"]["sample_rate"], 24000)
        self.assertEqual(result["audio_metadata"]["channels"], 2)
        command = result["asr"]["command"]
        for flag, value in (("-l", "en"), ("-t", "4"), ("-bo", "1"),
                            ("-bs", "1"), ("-tp", "0")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("-ng", command)
        self.assertIn("-nf", command)
        self.assertNotIn("--prompt", command)
        self.assertNotIn(str(self.reference), command)
        self.assertNotIn(self.reference.read_text(), command)
        self.assertIn("stub ASR log", self.log.read_text())

    def test_normalization_and_real_edit_counts(self):
        result = self.run_score("THE slow brown FOX!")
        self.assertEqual(result["wer"], .25)
        self.assertEqual(result["n_edits"], {"sub": 1, "ins": 0, "del": 0})
        self.assertEqual(result["n_ref_words"], 4)
        self.assertEqual(result["n_hyp_words"], 4)

    def test_empty_success_counts_all_deletions(self):
        result = self.run_score("")
        self.assertEqual(result["wer"], 1)
        self.assertEqual(result["n_edits"], {"sub": 0, "ins": 0, "del": 4})
        self.assertEqual(self.transcript.read_text(), "")

    def test_wer_can_exceed_one(self):
        self.assertGreater(self.run_score("a b c d e f g h i")["wer"], 1)

    def test_failure_does_not_score_partial_or_stale_transcript(self):
        self.transcript.write_text(self.reference.read_text())
        result = self.run_score(mode="failure")
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["wer"])
        self.assertEqual(result["asr"]["returncode"], 7)
        self.assertFalse(self.transcript.exists())

    def test_missing_new_transcript_cannot_reuse_stale_file(self):
        self.transcript.write_text(self.reference.read_text())
        result = self.run_score(mode="missing")
        self.assertIsNone(result["wer"])
        self.assertIn("did not write", result["reason"])
        self.assertFalse(self.transcript.exists())

    def test_timeout(self):
        result = self.run_score(mode="timeout", extra=("--timeout", "0.05"))
        self.assertIsNone(result["wer"])
        self.assertIn("timed out", result["reason"])
        self.assertFalse(self.transcript.exists())

    def test_missing_model_unavailable(self):
        self.model.unlink()
        result = self.run_score()
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["wer"])

    def test_missing_binary_unavailable(self):
        self.binary.unlink()
        self.assertEqual(self.run_score()["status"], "unavailable")

    def test_missing_audio_error(self):
        self.audio.unlink()
        self.assertEqual(self.run_score()["status"], "error")

    def test_truncated_wav_error(self):
        self.audio.write_bytes(self.audio.read_bytes()[:-2])
        self.assertIn("truncated", self.run_score()["reason"])

    def test_empty_reference_error(self):
        self.reference.write_text("... !")
        self.assertIn("no words", self.run_score()["reason"])

    def test_invalid_timeout_is_structured_json(self):
        result = self.run_score(extra=("--timeout", "nan"))
        self.assertIsNone(result["asr"]["timeout_seconds"])
        self.assertEqual(result["status"], "error")

    def test_output_alias_does_not_overwrite_reference(self):
        original = self.reference.read_text()
        self.output = self.reference
        result = self.run_score()
        self.assertIn("aliases", result["reason"])
        self.assertEqual(self.reference.read_text(), original)

    def test_hardlink_output_does_not_overwrite_audio(self):
        original = self.audio.read_bytes()
        os.link(self.audio, self.log)
        result = self.run_score()
        self.assertIn("aliases", result["reason"])
        self.assertEqual(self.audio.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
