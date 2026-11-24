#!/usr/bin/env python3
"""Tiny real-GGUF regressions for architecture-specific quantized shapes.

No checkpoints or accelerator are required. Missing numpy/gguf returns 77 so
CTest reports a skip. Direct-converter helper checks additionally need the
conversion imports (torch, safetensors and huggingface_hub); those checks are
explicitly skipped when only the lightweight GGUF dependencies are installed.
"""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parent
CORE_DEPS = ("numpy", "gguf")
CONVERTER_DEPS = ("torch", "safetensors", "huggingface_hub")


def load_script(name):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location("shape_test_" + path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if all(importlib.util.find_spec(name) is not None for name in CORE_DEPS):
    import gguf
    import numpy as np
    POLICY = load_script("requantize-gguf.py")
else:
    POLICY = None


def weights(shape, dtype=np.float32 if POLICY else None):
    return np.linspace(-1, 1, num=int(np.prod(shape)), dtype=dtype).reshape(shape)


def write_gguf(path, arch, tensors):
    writer = gguf.GGUFWriter(str(path), arch)
    for name, array in tensors.items():
        writer.add_tensor(name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class ShapeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="quant-shape-")
        self.directory = Path(self.temp.name)
        self.sequence = 0

    def tearDown(self):
        self.temp.cleanup()

    def requantize(self, arch, tensors, tier):
        self.sequence += 1
        stem = arch + "-" + str(self.sequence)
        src = self.directory / (stem + "-source.gguf")
        dst = self.directory / (stem + "-" + tier + ".gguf")
        write_gguf(src, arch, tensors)
        return self.requantize_path(src, dst, tier)

    def requantize_path(self, src, dst, tier, *options):
        source_hash = hashlib.sha256(src.read_bytes()).hexdigest()
        proc = subprocess.run([sys.executable, str(SCRIPTS / "requantize-gguf.py"), str(src), str(dst), tier, *options],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)
        self.assertEqual(hashlib.sha256(src.read_bytes()).hexdigest(), source_hash)
        return gguf.GGUFReader(str(dst))

    @staticmethod
    def snapshot(reader):
        tensors = {t.name: (tuple(reversed(t.shape)), t.tensor_type, t.data.tobytes())
                   for t in reader.tensors}
        field = reader.fields.get("supertonic.pwconv_squeezed")
        roster = [] if field is None else [bytes(field.parts[i]).decode("utf-8") for i in field.data]
        return tensors, roster

    def assert_kept(self, tensor, original):
        self.assertEqual(tuple(reversed(tensor.shape)), original.shape)
        expected = gguf.GGMLQuantizationType.F16 if original.dtype == np.float16 else gguf.GGMLQuantizationType.F32
        self.assertEqual(tensor.tensor_type, expected)
        self.assertEqual(tensor.data.tobytes(), original.tobytes())

    def test_pointwise_requires_capability(self):
        for tier in ("q4_0", "q8_0"):
            with self.subTest(tier=tier):
                qtype = POLICY._QUANT_TYPE[tier]
                self.assertFalse(POLICY.should_quantize("cfm/down_blocks/0/0/res_conv/weight", (256, 320, 1), qtype))
                self.assertTrue(POLICY.should_quantize("supertonic/vocoder/t0", (256, 320, 1), qtype,
                                                      allow_pointwise_squeeze=True))
                self.assertTrue(POLICY.should_quantize("cfm/attn/q/w", (64, 64), qtype))
                self.assertTrue(POLICY.should_quantize("hift/ups/0/weight", (64, 32, 32), qtype))
                self.assertFalse(POLICY.should_quantize("cfm/small/w", (4, 32), qtype))
                self.assertFalse(POLICY.should_quantize("flow/input_embedding", (64, 64), qtype))

    def test_chatterbox_keeps_3d_shapes(self):
        pointwise_name = "cfm/down_blocks/0/0/res_conv/weight"
        matrix_name = "cfm/attn/q/w"
        aligned_name = "hift/ups/0/weight"
        for tier in ("q4_0", "q8_0"):
            for dtype in (np.float32, np.float16):
                with self.subTest(tier=tier, dtype=dtype):
                    pointwise = weights((256, 320, 1), dtype)
                    reader = self.requantize("chatterbox-s3gen", {
                        pointwise_name: pointwise,
                        matrix_name: weights((64, 64), dtype),
                        aligned_name: weights((64, 32, 32), dtype),
                    }, tier)
                    tensors = {t.name: t for t in reader.tensors}
                    self.assert_kept(tensors[pointwise_name], pointwise)
                    for name, shape in ((matrix_name, (64, 64)), (aligned_name, (64, 32, 32))):
                        self.assertEqual(tensors[name].tensor_type, POLICY._QUANT_TYPE[tier])
                        self.assertEqual(tuple(reversed(tensors[name].shape)), shape)
                    self.assertNotIn("supertonic.pwconv_squeezed", reader.fields)
                    del reader

    def test_other_arch_keeps_pointwise_shape(self):
        original = weights((256, 320, 1))
        for tier in ("q4_0", "q8_0"):
            with self.subTest(tier=tier):
                reader = self.requantize("another-engine", {"projection/weight": original}, tier)
                self.assert_kept(reader.tensors[0], original)
                self.assertNotIn("supertonic.pwconv_squeezed", reader.fields)

    def test_supertonic_squeeze_remains_supported(self):
        pointwise = weights((256, 320, 1))
        name = "supertonic/vocoder/t0"
        raw_name = "supertonic/duration/t0"
        for tier in ("q4_0", "q8_0"):
            with self.subTest(tier=tier):
                reader = self.requantize("supertonic", {name: pointwise, raw_name: pointwise}, tier)
                tensors = {t.name: t for t in reader.tensors}
                tensor = tensors[name]
                self.assertEqual(tuple(reversed(tensor.shape)), (256, 320))
                self.assertEqual(tensor.tensor_type, POLICY._QUANT_TYPE[tier])
                expected = gguf.quants.quantize(pointwise[:, :, 0], POLICY._QUANT_TYPE[tier])
                self.assertEqual(tensor.data.tobytes(), expected.tobytes())
                field = reader.fields["supertonic.pwconv_squeezed"]
                self.assertEqual([bytes(field.parts[i]).decode("utf-8") for i in field.data], [name])
                self.assert_kept(tensors[raw_name], pointwise)

    def test_supertonic_squeeze_roster_survives_filtered_passes(self):
        vocoder_name = "supertonic/vocoder/t0"
        vector_name = "supertonic/vector_estimator/t0"
        vocoder = weights((64, 32, 1))
        vector = weights((32, 64, 1))
        for first_tier, second_tier in (("q8_0", "q8_0"), ("q4_0", "q4_0"), ("q8_0", "q4_0")):
            with self.subTest(first_tier=first_tier, second_tier=second_tier):
                stem = first_tier + "-then-" + second_tier
                src = self.directory / (stem + "-source.gguf")
                first = self.directory / (stem + "-first.gguf")
                second = self.directory / (stem + "-second.gguf")
                write_gguf(src, "supertonic3", {vocoder_name: vocoder, vector_name: vector})
                reader = self.requantize_path(src, first, first_tier, "--name-filter", "/vocoder/")
                before, first_roster = self.snapshot(reader)
                del reader
                reader = self.requantize_path(first, second, second_tier, "--name-filter", "/vector_estimator/")
                after, final_roster = self.snapshot(reader)
                del reader

                self.assertEqual(first_roster, [vocoder_name])
                self.assertEqual(before[vocoder_name], after[vocoder_name])
                self.assertEqual(after[vocoder_name][:2], ((64, 32), POLICY._QUANT_TYPE[first_tier]))
                self.assertEqual(after[vocoder_name][2],
                                 gguf.quants.quantize(vocoder[:, :, 0], POLICY._QUANT_TYPE[first_tier]).tobytes())
                self.assertEqual(after[vector_name][:2], ((32, 64), POLICY._QUANT_TYPE[second_tier]))
                self.assertEqual(after[vector_name][2],
                                 gguf.quants.quantize(vector[:, :, 0], POLICY._QUANT_TYPE[second_tier]).tobytes())
                self.assertEqual(final_roster, [vocoder_name, vector_name])

    def test_supertonic_squeeze_roster_survives_passthrough(self):
        name = "supertonic/vocoder/t0"
        for tier in ("q4_0", "q8_0"):
            with self.subTest(tier=tier):
                src = self.directory / (tier + "-source.gguf")
                first = self.directory / (tier + "-first.gguf")
                second = self.directory / (tier + "-passthrough.gguf")
                write_gguf(src, "supertonic3", {name: weights((64, 32, 1))})
                reader = self.requantize_path(src, first, tier)
                before, first_roster = self.snapshot(reader)
                del reader
                reader = self.requantize_path(first, second, tier, "--name-filter", "no-matching-tensor")
                after, final_roster = self.snapshot(reader)
                del reader
                self.assertEqual(before, after)
                self.assertEqual(first_roster, [name])
                self.assertEqual(final_roster, first_roster)

    @unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in CONVERTER_DEPS),
                         "direct converter imports require torch/safetensors/huggingface_hub")
    def test_direct_chatterbox_converters(self):
        modules = ((load_script("convert-s3gen-to-gguf.py"), "add_tensor_maybe_q"),
                   (load_script("convert-t3-mtl-to-gguf.py"), "add_maybe_quantized"))
        name = "cfm/down_blocks/0/0/res_conv/weight"
        matrix = "cfm/attn/q/w"
        original = weights((256, 320, 1))
        original_bytes = original.tobytes()
        for module, helper in modules:
            for tier in ("q4_0", "q8_0"):
                with self.subTest(module=module.__name__, tier=tier):
                    out = self.directory / (module.__name__ + "-" + tier + ".gguf")
                    writer = gguf.GGUFWriter(str(out), "chatterbox-s3gen")
                    write_tensor = getattr(module, helper)
                    write_tensor(writer, name, original, tier)
                    write_tensor(writer, matrix, weights((64, 64)), tier)
                    writer.write_header_to_file()
                    writer.write_kv_data_to_file()
                    writer.write_tensors_to_file()
                    writer.close()
                    reader = gguf.GGUFReader(str(out))
                    tensors = {t.name: t for t in reader.tensors}
                    self.assert_kept(tensors[name], original)
                    self.assertEqual(tensors[matrix].tensor_type, POLICY._QUANT_TYPE[tier])
                    self.assertNotIn("supertonic.pwconv_squeezed", reader.fields)
                    self.assertEqual(original.tobytes(), original_bytes)


if __name__ == "__main__":
    if POLICY is None:
        print("SKIP: quantized-shape regressions require numpy and gguf", file=sys.stderr)
        raise SystemExit(77)
    unittest.main(verbosity=2)
