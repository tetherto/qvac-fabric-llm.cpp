"""Check optional Core ML fixture gating without loading a model."""

import contextlib
import io
import itertools
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_transcribepsy_eu as cli_test


SIDECAR_SUFFIXES = ("encoder", "encoder-other", "encoder-gate")


class InferenceReached(Exception):
    pass


class TranscribePsyEuSkipGuardTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = Path(self.directory.name) / "full-eu.gguf"
        self.model.touch()

    def run_main(self, coreml):
        arguments = ["test_transcribepsy_eu.py", "--binary", "unused-binary",
                     "--model", str(self.model)]
        if coreml:
            arguments.append("--coreml")
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            return cli_test.main()

    def create_sidecars(self, present):
        for suffix, exists in zip(SIDECAR_SUFFIXES, present):
            if exists:
                (self.model.parent / f"{self.model.stem}-{suffix}.mlmodelc").mkdir()

    def test_partial_coreml_fixtures_skip_before_launch(self):
        for present in itertools.product((False, True), repeat=len(SIDECAR_SUFFIXES)):
            if all(present):
                continue
            with self.subTest(present=present):
                self.create_sidecars(present)
                with patch.object(cli_test.subprocess, "run") as launch:
                    self.assertEqual(self.run_main(coreml=True), 77)
                    launch.assert_not_called()
                for sidecar in self.model.parent.glob("*.mlmodelc"):
                    sidecar.rmdir()

    def test_complete_coreml_fixtures_reach_inference(self):
        self.create_sidecars((True,) * len(SIDECAR_SUFFIXES))
        with patch.object(cli_test, "transcribe", side_effect=InferenceReached):
            with self.assertRaises(InferenceReached):
                self.run_main(coreml=True)

    def test_native_fixtures_do_not_require_sidecars(self):
        with patch.object(cli_test, "transcribe", side_effect=InferenceReached):
            with self.assertRaises(InferenceReached):
                self.run_main(coreml=False)


if __name__ == "__main__":
    unittest.main()
