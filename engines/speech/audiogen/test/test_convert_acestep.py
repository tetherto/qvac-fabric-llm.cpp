"""Tests for scripts/convert-acestep-to-gguf.py.

The checkpoint-selection, safetensors-header, tokenizer, and naming logic run
against hand-crafted files with no third-party dependencies. The end-to-end
conversion test builds a tiny fake DiT checkpoint and reads the emitted GGUF
back; it runs only when real numpy + gguf are importable, matching how
test-minimax-converter treats optional Python dependencies.

Method names are kept short on purpose: the TruffleHog CI lane's Lob detector
flags `test_` followed by a long lowercase run as an API key.
"""

import importlib.util
import json
import os
import pathlib
import struct
import sys
import tempfile
import unittest
import zipfile


def load_converter():
    path = pathlib.Path(__file__).parents[1] / "scripts" / "convert-acestep-to-gguf.py"
    spec = importlib.util.spec_from_file_location("convert_acestep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def optional_deps_available():
    try:
        import gguf  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


converter = load_converter()


def write_safetensors(path, tensors):
    header = {}
    blobs = []
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    meta = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(meta)))
        f.write(meta)
        for blob in blobs:
            f.write(blob)


def write_silence_latent(path):
    values = struct.pack(
        "<%df" % (converter.SILENCE_LATENT_CHANNELS * converter.SILENCE_LATENT_FRAMES),
        *([0.5] * converter.SILENCE_LATENT_CHANNELS * converter.SILENCE_LATENT_FRAMES))
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data/0", values)


class ClassifyTest(unittest.TestCase):
    def test_stems(self):
        self.assertEqual(converter.classify("acestep-5Hz-lm-0.6B"), "lm")
        self.assertEqual(converter.classify("acestep-v15-turbo"), "dit")
        self.assertEqual(converter.classify("acestep-v15-sft"), "dit")
        self.assertEqual(converter.classify("Qwen3-Embedding-0.6B"), "text-enc")
        self.assertEqual(converter.classify("vae"), "vae")
        self.assertIsNone(converter.classify("silence_latent"))

    def test_lm_prefix(self):
        self.assertEqual(converter.normalize_tensor_name("layers.0.w", "lm"),
                         "model.layers.0.w")
        self.assertEqual(converter.normalize_tensor_name("model.layers.0.w", "lm"),
                         "model.layers.0.w")
        self.assertEqual(converter.normalize_tensor_name("layers.0.w", "dit"),
                         "layers.0.w")


class FileDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_single(self):
        (self.root / "model.safetensors").write_bytes(b"")
        self.assertEqual(converter.find_sf_files(str(self.root)),
                         [str(self.root / "model.safetensors")])

    def test_sharded(self):
        index = {"weight_map": {"a": "model-00002.safetensors",
                                "b": "model-00001.safetensors",
                                "c": "model-00001.safetensors"}}
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index))
        self.assertEqual(converter.find_sf_files(str(self.root)),
                         [str(self.root / "model-00001.safetensors"),
                          str(self.root / "model-00002.safetensors")])

    def test_diffusers(self):
        (self.root / "diffusion_pytorch_model.safetensors").write_bytes(b"")
        self.assertEqual(converter.find_sf_files(str(self.root)),
                         [str(self.root / "diffusion_pytorch_model.safetensors")])

    def test_none(self):
        self.assertEqual(converter.find_sf_files(str(self.root)), [])

    def test_header(self):
        path = self.root / "model.safetensors"
        raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        write_safetensors(str(path), {"w": ("F32", [2, 2], raw)})
        meta, hdr_size = converter.read_sf_header(str(path))
        self.assertEqual(meta["w"]["shape"], [2, 2])
        self.assertEqual(meta["w"]["data_offsets"], [0, 16])
        self.assertEqual(os.path.getsize(path), hdr_size + 16)

    def test_selection(self):
        for name in ("vae", "acestep-v15-turbo", "unrelated"):
            (self.root / name).mkdir()
        (self.root / "stray.txt").write_text("")
        selected, skipped = converter.select_checkpoints(str(self.root), None)
        self.assertEqual([(n, t) for n, _, t in selected],
                         [("acestep-v15-turbo", "dit"), ("vae", "vae")])
        self.assertEqual(skipped, ["unrelated"])

    def test_only(self):
        for name in ("vae", "acestep-v15-turbo"):
            (self.root / name).mkdir()
        selected, skipped = converter.select_checkpoints(str(self.root), ["vae"])
        self.assertEqual([n for n, _, _ in selected], ["vae"])
        self.assertEqual(skipped, [])


class TokenizerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_merges(self):
        path = self.root / "merges.txt"
        path.write_text("#version: 0.2\na b\n\nc d\n")
        self.assertEqual(converter.read_bpe_merges(str(path)), ["a b", "c d"])

    def test_tokens(self):
        path = self.root / "vocab.json"
        path.write_text(json.dumps({"b": 1, "a": 0, "oob": 5}))
        self.assertEqual(converter.read_bpe_tokens(str(path)), ["a", "b", ""])


class MainTest(unittest.TestCase):
    def test_missing_dir(self):
        self.assertEqual(converter.main(["--checkpoints", "/nonexistent-acestep"]), 1)

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(converter.main(["--checkpoints", tmp]), 1)


@unittest.skipUnless(optional_deps_available(), "requires numpy and gguf")
class ConvertTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.checkpoints = self.root / "checkpoints"
        self.out = self.root / "models"
        self.checkpoints.mkdir()
        self.write_dit_checkpoint(self.checkpoints / "acestep-v15-turbo")

    def tearDown(self):
        self._tmp.cleanup()

    def write_dit_checkpoint(self, model_dir):
        model_dir.mkdir()
        cfg = {"num_hidden_layers": 2, "hidden_size": 8, "in_channels": 4,
               "is_turbo": True, "rms_norm_eps": 1e-6}
        (model_dir / "config.json").write_text(json.dumps(cfg))
        f32 = struct.pack("<4f", 1.0, -1.0, 0.5, 2.0)
        f16 = struct.pack("<4H", 0x3C00, 0x3C00, 0x3C00, 0x3C00)
        write_safetensors(str(model_dir / "model.safetensors"), {
            "decoder.norm_out.weight": ("F16", [4], f16),
            "decoder.proj_in.1.weight": ("F32", [2, 2], f32),
        })
        write_silence_latent(str(model_dir / converter.SILENCE_LATENT_FILE))

    def test_convert(self):
        import gguf
        import numpy as np

        rc = converter.main(["--checkpoints", str(self.checkpoints),
                             "--out", str(self.out)])
        self.assertEqual(rc, 0)

        out_path = self.out / "acestep-v15-turbo-BF16.gguf"
        self.assertTrue(out_path.exists())

        reader = gguf.GGUFReader(str(out_path))
        fields = reader.fields
        arch = bytes(fields["general.architecture"].parts[-1]).decode()
        self.assertEqual(arch, "acestep-dit")
        self.assertEqual(int(fields["acestep.in_channels"].parts[-1][0]), 4)
        self.assertEqual(bool(fields["acestep.is_turbo"].parts[-1][0]), True)

        tensors = {t.name: t for t in reader.tensors}
        self.assertEqual(tensors["decoder.proj_in.1.weight"].tensor_type,
                         gguf.GGMLQuantizationType.BF16)
        self.assertEqual(tensors["decoder.norm_out.weight"].tensor_type,
                         gguf.GGMLQuantizationType.F16)

        latent = tensors["silence_latent"]
        self.assertEqual(latent.tensor_type, gguf.GGMLQuantizationType.F32)
        self.assertEqual(list(latent.shape),
                         [converter.SILENCE_LATENT_CHANNELS,
                          converter.SILENCE_LATENT_FRAMES])
        self.assertTrue(np.allclose(latent.data[:8], 0.5))

    def test_rerun_skips(self):
        args = ["--checkpoints", str(self.checkpoints), "--out", str(self.out)]
        self.assertEqual(converter.main(args), 0)
        stamp = os.path.getmtime(self.out / "acestep-v15-turbo-BF16.gguf")
        self.assertEqual(converter.main(args), 0)
        self.assertEqual(os.path.getmtime(self.out / "acestep-v15-turbo-BF16.gguf"), stamp)

    def test_missing_silence_latent_fails(self):
        model_dir = self.checkpoints / "acestep-v15-turbo"
        (model_dir / converter.SILENCE_LATENT_FILE).unlink()
        rc = converter.main(["--checkpoints", str(self.checkpoints), "--out", str(self.out)])
        self.assertEqual(rc, 1)
        self.assertFalse((self.out / "acestep-v15-turbo-BF16.gguf").exists())

    def test_missing_tokenizer_fails_lm(self):
        lm_dir = self.checkpoints / "acestep-5Hz-lm-0.6B"
        self.write_lm_checkpoint(lm_dir)
        rc = converter.main(["--checkpoints", str(self.checkpoints), "--out", str(self.out),
                             "--only", "acestep-5Hz-lm-0.6B"])
        self.assertEqual(rc, 1)
        self.assertFalse((self.out / "acestep-5Hz-lm-0.6B-BF16.gguf").exists())

    def write_lm_checkpoint(self, model_dir):
        model_dir.mkdir(exist_ok=True)
        cfg = {"num_hidden_layers": 2, "hidden_size": 8, "vocab_size": 16,
               "tie_word_embeddings": True, "rms_norm_eps": 1e-6}
        (model_dir / "config.json").write_text(json.dumps(cfg))
        f32 = struct.pack("<4f", 1.0, -1.0, 0.5, 2.0)
        write_safetensors(str(model_dir / "model.safetensors"), {
            "model.layers.0.self_attn.q_proj.weight": ("F32", [2, 2], f32),
        })


if __name__ == "__main__":
    sys.exit(unittest.main())
