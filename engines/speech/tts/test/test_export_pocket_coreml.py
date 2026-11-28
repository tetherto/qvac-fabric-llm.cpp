#!/usr/bin/env python3
"""Pocket Core ML sidecar rebuild checks without model downloads.

FlowLM step, on random weights with the checkpoint's tensor names and small
dimensions: decoding frame by frame over a prefix written into its state
matches a full causal forward of the same rows, RoPE tables match ggml's mode
0, the sidecar naming rule and shared constants match the runtime, and a
Core ML export of a tiny GGUF (macOS with coremltools) carries the
checkpoint hash and predicts the same frames with the prefix written into its
Core ML state.

Mimi decoder, on random full-size weights: the phase-form upsample and
transposed convolutions match PyTorch's transposed convolutions, decoding in
calls of any frame count (across the attention window) gives the same PCM,
the geometry matches the runtime, and a Core ML export reproduces the eager
stream frame by frame with its state fed back."""

import importlib.util
import math
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

HAVE_DEPS = all(importlib.util.find_spec(name) for name in ("numpy", "torch"))
HAVE_GGUF = HAVE_DEPS and importlib.util.find_spec("gguf") is not None
HAVE_COREML = HAVE_GGUF and importlib.util.find_spec("coremltools") is not None and sys.platform == "darwin"
if HAVE_DEPS:
    import numpy as np
    import torch
    import torch.nn.functional as F
    import pocket_coreml_mimi as mimi
    import pocket_coreml_models as pm
    SPEC = importlib.util.spec_from_file_location("export_pocket_coreml", ROOT / "scripts/export-pocket-coreml.py")
    EXPORTER = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(EXPORTER)

CONTEXT = 12
PREFIX = 5
FRAMES = 4


def runtime_constant(name):
    match = re.search(name + r"\s*=\s*([^;]+);", (ROOT / "src/pocket/coreml.h").read_text())
    return match.group(1).strip()


def reference(tensors, metadata, seed=1):
    """Prefix embeddings, frame latents/noises, and the causal-forward hidden/next_latent of each frame."""
    rng = np.random.default_rng(seed)
    d = pm.dimensions(metadata)
    model = pm.FlowLM(tensors, metadata).eval()
    prefix = torch.from_numpy(rng.normal(0, 1, (PREFIX, d["dim"])).astype(np.float32))
    latents = torch.from_numpy(rng.normal(0, 1, (FRAMES, d["latent_dim"])).astype(np.float32))
    noises = torch.from_numpy(rng.normal(0, 0.5, (FRAMES, d["latent_dim"])).astype(np.float32))
    with torch.no_grad():
        _, keys, values = model.causal(prefix)
        rows = torch.cat((prefix, model.linear(latents, "input_linear", False)))
        hidden = model.causal(rows)[0][PREFIX:]
        nxt = noises + model.velocity(hidden, noises, model.time_condition(0.0, 1.0))
    return model, keys, values, latents, noises, hidden, nxt


def frame_inputs(metadata, position, latent, noise):
    inputs = pm.example_inputs(metadata, CONTEXT, position)
    inputs["latent"], inputs["noise"] = latent[None], noise[None]
    return inputs


@unittest.skipUnless(HAVE_DEPS, "numpy/torch unavailable")
class ExportTest(unittest.TestCase):
    def test_sidecar_naming(self):
        for gguf, stem in (("flow-lm.gguf", "flow-lm-step"), ("m/flow-lm-f16.gguf", "flow-lm-step"),
                           ("pocket-flow-lm-q8_0.gguf", "pocket-flow-lm-step")):
            self.assertEqual(EXPORTER.model_stem(gguf) + EXPORTER.SUFFIX, stem)
        self.assertEqual(runtime_constant("COREML_STEP_SUFFIX"), '"%s"' % EXPORTER.SUFFIX)
        self.assertEqual(runtime_constant("COREML_SCHEMA"), '"%s"' % EXPORTER.SCHEMA)
        self.assertEqual(float(runtime_constant("COREML_MASKED").rstrip("f")), pm.MASKED)

    def test_rope_matches_ggml_mode_0(self):
        head_dim, base = 64, 10000.0
        for position in (0, 1, 37, 500):
            cos, sin = pm.rope_tables(position, head_dim, base)
            theta, scale = np.float32(position), np.float32(base) ** np.float32(-2.0 / head_dim)
            for i in range(head_dim // 2):  # ggml_rope_cache_init: theta *= theta_scale per pair
                self.assertAlmostEqual(float(cos[i]), math.cos(float(theta)), delta=1e-3)
                self.assertAlmostEqual(float(sin[i]), math.sin(float(theta)), delta=1e-3)
                theta = np.float32(theta * scale)

    def test_step_matches_causal_forward(self):
        tensors, metadata = pm.random_checkpoint(7)
        model, keys, values, latents, noises, hidden, nxt = reference(tensors, metadata)
        step = pm.FlowStep(tensors, metadata, CONTEXT, state_dtype=torch.float32).eval()
        with torch.no_grad():
            for i in range(model.layers):
                step.keys[i, :, :PREFIX] = keys[i]
                step.values[i, :, :PREFIX] = values[i]
            for f in range(FRAMES):
                inputs = frame_inputs(metadata, PREFIX + f, latents[f], noises[f])
                h, n = step(*(inputs[name] for name in pm.INPUTS))
                np.testing.assert_allclose(h[0].numpy(), hidden[f].numpy(), atol=2e-5, rtol=1e-4)
                np.testing.assert_allclose(n[0].numpy(), nxt[f].numpy(), atol=2e-5, rtol=1e-4)
        self.assertTrue(torch.equal(step.keys[:, :, PREFIX + FRAMES:], torch.zeros_like(step.keys[:, :, PREFIX + FRAMES:])),
                        "a frame wrote outside its position")

    def test_time_condition_folds_s0_t1(self):
        tensors, metadata = pm.random_checkpoint(3)
        step = pm.FlowStep(tensors, metadata, CONTEXT)
        with torch.no_grad():
            np.testing.assert_allclose(step.time_bias.numpy(), step.time_condition(0.0, 1.0).numpy(), atol=0)

    @unittest.skipUnless(HAVE_GGUF, "gguf unavailable")
    def test_read_gguf_rejects_other_architectures(self):
        import gguf

        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "mimi.gguf"
            writer = gguf.GGUFWriter(str(path), "pocket-tts-mimi")
            writer.add_tensor("x", np.zeros(4, dtype=np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            with self.assertRaises(ValueError):
                pm.read_gguf(path)

    @unittest.skipUnless(HAVE_COREML, "coremltools/gguf unavailable or not macOS")
    def test_coreml_export_and_state_prefix(self):
        import coremltools as ct
        import gguf

        tensors, metadata = pm.random_checkpoint(11, source="tiny-checkpoint")
        model, keys, values, latents, noises, hidden, nxt = reference(tensors, metadata)
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "flow-lm-f16.gguf"
            writer = gguf.GGUFWriter(str(path), pm.ARCH)
            for name, value in metadata.items():
                if isinstance(value, float):
                    writer.add_float32(name, value)
                elif isinstance(value, str):
                    writer.add_string(name, value)
                else:
                    writer.add_uint32(name, value)
            for name, value in tensors.items():
                writer.add_tensor(name, value)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            argv, sys.argv = sys.argv, ["export-pocket-coreml.py", "--gguf", str(path), "--context", str(CONTEXT)]
            try:
                EXPORTER.main()
            finally:
                sys.argv = argv
            package = Path(scratch) / "flow-lm-step.mlpackage"
            self.assertTrue(package.exists())
            mlmodel = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
            meta = mlmodel.user_defined_metadata
            self.assertEqual(meta["pocket.source_sha256"], "tiny-checkpoint")
            self.assertEqual(meta["pocket.coreml.context"], str(CONTEXT))
            self.assertEqual(meta["pocket.coreml.schema"], EXPORTER.SCHEMA)
            state = mlmodel.make_state()
            for name, rows in (("keys", keys), ("values", values)):
                buffer = np.zeros((model.layers, model.heads, CONTEXT, model.head_dim), dtype=np.float32)
                for i in range(model.layers):
                    buffer[i, :, :PREFIX] = rows[i].numpy()
                state.write_state(name, buffer)
            for f in range(FRAMES):
                inputs = frame_inputs(metadata, PREFIX + f, latents[f], noises[f])
                out = mlmodel.predict({k: v.numpy() for k, v in inputs.items()}, state=state)
                for name, expected in (("hidden", hidden[f]), ("next_latent", nxt[f])):
                    got, want = out[name].reshape(-1), expected.numpy().reshape(-1)
                    scale = max(1.0, float(np.abs(want).max()))
                    self.assertLess(float(np.abs(got - want).max()) / scale, 3e-2, f"frame {f} {name}")


@unittest.skipUnless(HAVE_DEPS, "numpy/torch unavailable")
class MimiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tensors, cls.metadata = mimi.random_checkpoint(5, source="tiny-mimi")
        cls.model = mimi.Decoder(cls.tensors).eval()
        cls.latents = np.random.default_rng(2).normal(0, 1, (20, mimi.LATENT)).astype(np.float32)

    def test_naming_and_geometry_match_runtime(self):
        self.assertEqual(EXPORTER.model_stem("m/mimi.gguf") + EXPORTER.MIMI_SUFFIX, "mimi-decoder")
        self.assertEqual(runtime_constant("COREML_DECODER_SUFFIX"), '"%s"' % EXPORTER.MIMI_SUFFIX)
        header = (ROOT / "src/pocket/coreml.h").read_text()
        for name, value in (("dim", mimi.DIM), ("latent", mimi.LATENT), ("heads", mimi.HEADS),
                            ("head_dim", mimi.HEAD_DIM), ("layers", mimi.LAYERS), ("keep", mimi.KEEP),
                            ("upsample", mimi.UP), ("frame_samples", mimi.FRAME_SAMPLES), ("frames", mimi.FRAMES)):
            self.assertRegex(header, rf"\b{name} = {value}\b", name)
        runtime = (ROOT / "src/pocket/coreml.cpp").read_text()
        for name, channels, length in mimi.HISTORIES:
            self.assertIn(f'{{"{name}", "new_{name}", {channels}, {length}}}', runtime)

    def test_phase_forms_match_transposed_convolutions(self):
        rng = np.random.default_rng(4)
        steps = 5
        x = torch.from_numpy(rng.normal(0, 1, (1, mimi.DIM, steps + 1)).astype(np.float32))  # [history, x]
        with torch.no_grad():
            # upsample: depthwise, stride 16, kernel 32
            y = F.conv1d(x, self.model.upsample, groups=mimi.DIM).reshape(mimi.DIM, mimi.UP, -1)
            y = y.permute(0, 2, 1).reshape(1, mimi.DIM, -1)
            w = torch.from_numpy(self.tensors["upsample.convtr.convtr.weight"].T.copy())[:, None, :]  # [DIM, 1, 32]
            want = F.conv_transpose1d(x, w, stride=mimi.UP, groups=mimi.DIM)[..., mimi.UP:mimi.UP * (steps + 1)]
            np.testing.assert_allclose(y.numpy(), want.numpy(), atol=1e-5)
            # first decoder transposed conv: ratio 6, kernel 12
            ratio, k = mimi.RATIOS[0], self.tensors["decoder.model.2.convtr.weight"]  # [2r, out, in]
            y = F.conv1d(x, self.model.p["convtr0_w"])
            out = self.model.out_channels[0]
            y = y.reshape(ratio, out, -1).permute(1, 2, 0).reshape(1, out, -1)
            w = torch.from_numpy(k.transpose(2, 1, 0).copy())  # [in, out, 2r]
            want = F.conv_transpose1d(x, w, stride=ratio)[..., ratio:ratio * (steps + 1)]
            np.testing.assert_allclose(y.numpy(), want.numpy(), atol=1e-4)

    def test_any_call_split_gives_the_same_stream(self):
        # 20 frames = 320 transformer rows, past the 250-row attention window.
        whole = mimi.run(self.model, self.latents, (20,))
        for calls in ((1,) * 20, (3, 7, 1, 9), (16, 4)):
            np.testing.assert_allclose(mimi.run(self.model, self.latents, calls), whole, atol=2e-5, err_msg=str(calls))
        self.assertGreater(float(np.abs(whole).max()), 1e-3)

    @unittest.skipUnless(HAVE_COREML, "coremltools/gguf unavailable or not macOS")
    def test_coreml_decoder_streams(self):
        import coremltools as ct

        inputs = mimi.Stream().inputs(np.zeros((mimi.FRAMES, mimi.LATENT), np.float32))
        converted = EXPORTER.convert_decoder(self.model, inputs)
        with tempfile.TemporaryDirectory() as scratch:
            package = Path(scratch) / "mimi-decoder.mlpackage"
            converted.save(str(package))
            mlmodel = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
            frames = 18
            want = mimi.run(self.model, self.latents[:frames], (1,) * frames)
            stream, got = mimi.Stream(), []
            for f in range(frames):
                out = mlmodel.predict(stream.inputs(self.latents[f:f + 1]))
                got.append(stream.update(out, 1))
            got = np.concatenate(got)
            snr = 10 * np.log10((want ** 2).sum() / ((got - want) ** 2).sum())
            self.assertGreater(snr, 30, f"Core ML Mimi SNR {snr:.1f} dB")


if __name__ == "__main__":
    if not HAVE_DEPS:
        print("SKIP: numpy/torch unavailable")
        sys.exit(0)
    unittest.main()
