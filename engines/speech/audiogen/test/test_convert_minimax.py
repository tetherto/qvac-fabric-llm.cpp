import importlib.util
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


def load_converter():
    path = pathlib.Path(__file__).parents[1] / "scripts" / "convert-minimax-music3-to-gguf.py"
    spec = importlib.util.spec_from_file_location("convert_minimax_music3", path)
    module = importlib.util.module_from_spec(spec)
    stubs = {}
    for dependency in ("numpy", "gguf"):
        if importlib.util.find_spec(dependency) is None:
            stubs[dependency] = types.ModuleType(dependency)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


converter = load_converter()


class FakeBundle:
    def __init__(self, payload, fail=False):
        self.payload = payload
        self.fail = fail
        self.tensors = [("weight", object(), "f32")]
        self.kv = [
            ("add_license", ("license",)),
            ("add_file_type", (1,)),
            ("add_quantization_version", (1,)),
            ("add_string", ("mm3.model", "MiniMax-Music3")),
            ("add_uint32", ("mm3.converter_version", 1)),
            ("add_string", ("mm3.source_layout", "unit")),
            ("add_array", ("mm3.synth.components", ["depth", "cond", "dit", "vocoder"])),
            ("add_tokenizer_model", ("gpt2",)),
            ("add_tokenizer_pre", ("qwen2",)),
            ("add_token_list", ([],)),
            ("add_token_types", ([],)),
            ("add_token_merges", ([],)),
        ]

    def write(self, path, quant):
        pathlib.Path(path).write_text(f"{self.payload}:{quant}", encoding="utf-8")
        if self.fail:
            raise RuntimeError("write failed")


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def assert_clean(self, transaction_id, finals):
        for index, final in enumerate(finals):
            self.assertFalse(pathlib.Path(
                converter.temporary_output_path(str(final), transaction_id, index)
            ).exists())
            self.assertFalse(pathlib.Path(
                converter.backup_output_path(str(final), transaction_id, index)
            ).exists())

    def test_flow_defaults(self):
        # Written to the synth GGUF as mm3.flow.*. The engine takes the CFG
        # scale from there but uses its own kDefaultFlowSteps for the step
        # count, so the two constants are kept equal by hand.
        self.assertEqual(converter.DIT_STEPS, 20)
        self.assertEqual(converter.DIT_CFG_SCALE, 1.7)

    def test_tokenizer(self):
        with self.assertRaises(SystemExit):
            converter.require_tokenizer(None)
        tokenizer = {"model": {"type": "BPE"}}
        self.assertIs(converter.require_tokenizer(tokenizer), tokenizer)

    def test_safe_load(self):
        calls = []
        torch = types.ModuleType("torch")

        def load(path, map_location, weights_only):
            calls.append((path, map_location, weights_only))
            return {}

        torch.load = load
        with mock.patch.dict(sys.modules, {"torch": torch}):
            source = converter.TorchFile("checkpoint.pth")
            source.close()
        self.assertEqual(calls, [("checkpoint.pth", "cpu", True)])

    def test_temp_path(self):
        final = str(self.root / "model.gguf")
        self.assertEqual(
            converter.temporary_output_path(final, "unit", 2),
            f"{final}.tmp-unit-2",
        )

    def test_lm_metadata(self):
        bundle = FakeBundle("lm")
        bundle.kv = [entry for entry in bundle.kv if not entry[0].startswith("add_token")]
        with self.assertRaises(SystemExit):
            converter.validate_bundle(bundle, "lm")

    def test_commit(self):
        lm = self.root / "mm3-lm-q8_0.gguf"
        synth = self.root / "mm3-synth-q8_0.gguf"
        transaction = converter.OutputTransaction("success")
        transaction.write(FakeBundle("lm"), str(lm), "q8_0", "lm")
        transaction.write(FakeBundle("synth"), str(synth), "q8_0", "synth")
        self.assertFalse(lm.exists())
        self.assertFalse(synth.exists())
        transaction.commit()
        self.assertEqual(lm.read_text(encoding="utf-8"), "lm:q8_0")
        self.assertEqual(synth.read_text(encoding="utf-8"), "synth:q8_0")

    def test_write_cleanup(self):
        final = self.root / "mm3-lm-f16.gguf"
        transaction = converter.OutputTransaction("write-failure")
        with self.assertRaises(RuntimeError):
            transaction.write(FakeBundle("partial", fail=True), str(final), "f16", "lm")
        transaction.cleanup()
        self.assertFalse(final.exists())
        self.assertFalse(pathlib.Path(f"{final}.tmp-write-failure-0").exists())

    def test_restore(self):
        lm = self.root / "mm3-lm-f16.gguf"
        synth = self.root / "mm3-synth-f16.gguf"
        lm.write_text("old-lm", encoding="utf-8")
        synth.write_text("old-synth", encoding="utf-8")
        transaction = converter.OutputTransaction("publish-failure")
        transaction.write(FakeBundle("new-lm"), str(lm), "f16", "lm")
        transaction.write(FakeBundle("new-synth"), str(synth), "f16", "synth")
        original_replace = converter.os.replace

        def fail_second_publish(source, destination):
            if source.endswith(".tmp-publish-failure-1"):
                raise OSError("publish failed")
            original_replace(source, destination)

        converter.os.replace = fail_second_publish
        try:
            with self.assertRaises(OSError):
                transaction.commit()
        finally:
            converter.os.replace = original_replace
        transaction.cleanup()
        transaction.cleanup()
        self.assertEqual(lm.read_text(encoding="utf-8"), "old-lm")
        self.assertEqual(synth.read_text(encoding="utf-8"), "old-synth")
        self.assert_clean("publish-failure", [lm, synth])

    def test_restore_error(self):
        lm = self.root / "mm3-lm-f16.gguf"
        synth = self.root / "mm3-synth-f16.gguf"
        lm.write_text("old-lm", encoding="utf-8")
        synth.write_text("old-synth", encoding="utf-8")
        transaction = converter.OutputTransaction("restore-failure")
        transaction.write(FakeBundle("new-lm"), str(lm), "f16", "lm")
        transaction.write(FakeBundle("new-synth"), str(synth), "f16", "synth")
        original_replace = converter.os.replace

        def fail_restore(source, destination):
            if source.endswith(".tmp-restore-failure-1"):
                raise ValueError("publish failed")
            if source.endswith(".backup-restore-failure-1"):
                raise OSError("restore failed")
            original_replace(source, destination)

        converter.os.replace = fail_restore
        try:
            with self.assertRaises(RuntimeError) as raised:
                try:
                    transaction.commit()
                finally:
                    transaction.cleanup()
                    transaction.cleanup()
        finally:
            converter.os.replace = original_replace

        lm_backup = pathlib.Path(
            converter.backup_output_path(str(lm), "restore-failure", 0)
        )
        synth_backup = pathlib.Path(
            converter.backup_output_path(str(synth), "restore-failure", 1)
        )
        expected = (
            f"output rollback failed: backup={str(synth_backup)!r}, "
            f"destination={str(synth)!r}, error=OSError: restore failed; "
            "original remains at backup"
        )
        self.assertEqual(str(raised.exception), expected)
        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertEqual(str(raised.exception.__cause__), "publish failed")
        self.assertEqual(lm.read_text(encoding="utf-8"), "old-lm")
        self.assertFalse(synth.exists())
        self.assertFalse(lm_backup.exists())
        self.assertEqual(synth_backup.read_text(encoding="utf-8"), "old-synth")
        self.assertEqual(transaction.restored_backups, {str(lm_backup)})
        self.assertFalse(pathlib.Path(
            converter.temporary_output_path(str(lm), "restore-failure", 0)
        ).exists())
        self.assertFalse(pathlib.Path(
            converter.temporary_output_path(str(synth), "restore-failure", 1)
        ).exists())


if __name__ == "__main__":
    unittest.main()
