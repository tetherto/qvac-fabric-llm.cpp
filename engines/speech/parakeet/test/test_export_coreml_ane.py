#!/usr/bin/env python3
"""Model-free checks for the Neural Engine Core ML exporters.

Covers the pure helpers of export-encoder-coreml-ane.py (frame arithmetic and
the channel-axis LayerNorm the ANE graph uses) and the TDT decoder graph of
export-decoder-coreml-ane.py against its scalar reference on random weights.
"""

import importlib.util
import platform
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None

if torch is None:
    print("SKIP: torch is not installed")
    raise SystemExit(0)

if "coremltools" not in sys.modules:
    try:
        import coremltools  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["coremltools"] = types.ModuleType("coremltools")

try:
    import gguf  # noqa: F401
except ModuleNotFoundError:
    print("SKIP: gguf is not installed")
    raise SystemExit(0)

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load(name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENCODER = load("export-encoder-coreml-ane.py", "export_encoder_coreml_ane")
DECODER = load("export-decoder-coreml-ane.py", "export_decoder_coreml_ane")
REFERENCE = load("ref-encoder-from-gguf.py", "ref_encoder_from_gguf")
GENERIC = load("export-encoder-coreml.py", "export_encoder_coreml")
EXPERIMENT = load("export-transcribepsy-eu-coreml-experiment.py", "export_eu_coreml_experiment")


def toy_encoder_weights(d_model=8, n_layers=2, n_heads=2, n_mels=8, channels=4, ff_dim=16,
                        conv_kernel=5):
    """Random FastConformer weights in the GGUF layout both scripts read."""
    torch.manual_seed(2)
    w = {}

    def rand(*shape, scale=0.3):
        return torch.randn(*shape) * scale

    w["preproc.mel_filterbank"] = torch.ones(n_mels, 9)
    w["encoder.subsampling.conv0.weight"] = rand(channels, 1, 3, 3)
    w["encoder.subsampling.conv0.bias"] = rand(channels, scale=0.1)
    for stage in ("conv1", "conv2"):
        w[f"encoder.subsampling.{stage}_dw.weight"] = rand(channels, 1, 3, 3)
        w[f"encoder.subsampling.{stage}_dw.bias"] = rand(channels, scale=0.1)
        w[f"encoder.subsampling.{stage}_pw.weight"] = rand(channels, channels, 1, 1)
        w[f"encoder.subsampling.{stage}_pw.bias"] = rand(channels, scale=0.1)
    freq_out = ENCODER.encoder_frames_for_mel(n_mels, causal=False)
    w["encoder.subsampling.out.weight"] = rand(d_model, channels * freq_out)
    w["encoder.subsampling.out.bias"] = rand(d_model, scale=0.1)
    for i in range(n_layers):
        p = f"encoder.blk.{i}"
        for norm in ("norm_ff1", "norm_attn", "norm_conv", "norm_ff2", "norm_out", "conv.norm"):
            w[f"{p}.{norm}.weight"] = 1.0 + rand(d_model, scale=0.1)
            w[f"{p}.{norm}.bias"] = rand(d_model, scale=0.1)
        for ff in ("ff1", "ff2"):
            w[f"{p}.{ff}.linear1.weight"] = rand(ff_dim, d_model)
            w[f"{p}.{ff}.linear1.bias"] = rand(ff_dim, scale=0.1)
            w[f"{p}.{ff}.linear2.weight"] = rand(d_model, ff_dim)
            w[f"{p}.{ff}.linear2.bias"] = rand(d_model, scale=0.1)
        for proj in ("q", "k", "v", "out", "pos"):
            w[f"{p}.attn.{proj}.weight"] = rand(d_model, d_model)
            if proj != "pos":
                w[f"{p}.attn.{proj}.bias"] = rand(d_model, scale=0.1)
        w[f"{p}.attn.pos_bias_u"] = rand(n_heads, d_model // n_heads, scale=0.1)
        w[f"{p}.attn.pos_bias_v"] = rand(n_heads, d_model // n_heads, scale=0.1)
        w[f"{p}.conv.pw1.weight"] = rand(2 * d_model, d_model)
        w[f"{p}.conv.pw1.bias"] = rand(2 * d_model, scale=0.1)
        w[f"{p}.conv.dw.weight"] = rand(d_model, 1, conv_kernel)
        w[f"{p}.conv.dw.bias"] = rand(d_model, scale=0.1)
        w[f"{p}.conv.pw2.weight"] = rand(d_model, d_model)
        w[f"{p}.conv.pw2.bias"] = rand(d_model, scale=0.1)
    meta = {
        "parakeet.encoder.d_model": d_model,
        "parakeet.encoder.n_layers": n_layers,
        "parakeet.encoder.n_heads": n_heads,
        "parakeet.encoder.xscaling": True,
        "parakeet.encoder.conv_norm_type": "layer_norm",
        "parakeet.encoder.causal_downsampling": False,
        "parakeet.encoder.pos_emb_max_len": 64,
    }
    return w, meta


class EuEncoderTests(unittest.TestCase):
    def test_routed_encoder_matches_independent_routes_and_unadapted_gate(self):
        model = EXPERIMENT.RoutedEncoder(REFERENCE, self.weights, self.meta, 64).eval()
        mel = torch.randn(8, 64)
        for valid in (1, 7, 8, 17, 63, 64):
            inputs, count = ENCODER.window_inputs(model, mel, valid)
            for scale, route in ((0, "en"), (1, "other")):
                with self.subTest(valid=valid, route=route), torch.inference_mode():
                    output, gate = model(*inputs, torch.full((1, 1, 1, 1), float(scale)))
                    want = ENCODER.reference_frames(REFERENCE, self.weights, self.meta,
                                                     mel, valid, route)
                    self.assertLess(ENCODER.relative_error(output[0, :, 0, :count].T, want),
                                    ENCODER.PARITY_REL_TOLERANCE)
                    if scale == 0:
                        want_gate = ENCODER.reference_frames(REFERENCE, self.weights, self.meta,
                                                              mel, valid, "gate")
                        self.assertLess(ENCODER.relative_error(gate[0, :, 0, :count].T, want_gate),
                                        ENCODER.PARITY_REL_TOLERANCE)

    def test_tracing_preserves_runtime_adapter_selection(self):
        model = EXPERIMENT.RoutedEncoder(REFERENCE, self.weights, self.meta, 64).eval()
        inputs, _ = ENCODER.window_inputs(model, torch.randn(8, 64))
        zero = torch.zeros(1, 1, 1, 1)
        traced = torch.jit.trace(model, (*inputs, zero), check_trace=False)
        with torch.inference_mode():
            english, _ = traced(*inputs, zero)
            other, _ = traced(*inputs, torch.ones_like(zero))
            expected, _ = model(*inputs, torch.ones_like(zero))
        torch.testing.assert_close(other, expected)
        self.assertGreater(float((other - english).abs().max()), 0.01)

    def test_multifunction_predictions_skip_on_macos14(self):
        spec = importlib.util.spec_from_file_location("ane_tests_macos14", __file__)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "platform", "darwin"), \
                mock.patch.object(platform, "mac_ver", return_value=("14.7.0", ("", "", ""), "")):
            spec.loader.exec_module(module)
        case = module.EuEncoderTests("test_multifunction_shares_weights_and_preserves_route_outputs")
        result = unittest.TestResult()
        case.run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.failures, [])

    @unittest.skipUnless(sys.platform == "darwin" and int(platform.mac_ver()[0].split(".")[0]) >= 15,
                         "Multifunction Core ML predictions require macOS 15+")
    def test_multifunction_shares_weights_and_preserves_route_outputs(self):
        import coremltools as ct

        if not hasattr(ct, "convert"):
            self.skipTest("coremltools is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor = ct.utils.MultiFunctionDescriptor()
            mel = torch.randn(8, 64)
            for route in EXPERIMENT.ROUTES:
                model = ENCODER.AneEncoder(REFERENCE, self.weights, self.meta, 64,
                                           eu_route=route).eval()
                converted = EXPERIMENT.convert_encoder(model, unified=False)
                path = root / f"{route}.mlpackage"
                converted.save(str(path))
                descriptor.add_function(str(path), "main", route)
            descriptor.default_function_name = "en"
            shared = root / "shared.mlpackage"
            ct.utils.save_multifunction(descriptor, str(shared))
            shared_weights = sum(path.stat().st_size for path in shared.rglob("*.bin"))
            separate_weights = sum(path.stat().st_size for route in EXPERIMENT.ROUTES
                                   for path in (root / f"{route}.mlpackage").rglob("*.bin"))
            self.assertLess(shared_weights, separate_weights)
            for route in EXPERIMENT.ROUTES:
                model = ENCODER.AneEncoder(REFERENCE, self.weights, self.meta, 64,
                                           eu_route=route).eval()
                inputs, count = ENCODER.window_inputs(model, mel)
                compiled = ct.models.MLModel(str(shared), function_name=route,
                                              compute_units=ct.ComputeUnit.CPU_ONLY)
                output = compiled.predict({name: value.numpy().astype(np.float16)
                                           for name, value in zip(EXPERIMENT.INPUT_NAMES, inputs)})
                got = torch.from_numpy(output["encoder_out"][0, :, 0, :count].T.copy())
                want = ENCODER.reference_frames(REFERENCE, self.weights, self.meta, mel,
                                                 eu_route=route)
                self.assertLess(ENCODER.relative_error(got, want), 0.01)

    def test_experiments_require_gate_capacity_and_preserve_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "routed.mlpackage"
            arguments = ["export", "--gguf", "source.gguf", "--route", "unified",
                         "--out", str(output)]
            with mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    EXPERIMENT.parse_arguments([*arguments, "--window-mel-frames", "2000"])
                with self.assertRaises(SystemExit):
                    EXPERIMENT.parse_arguments([*arguments, "--palettize-workers", "0"])
            self.assertEqual(EXPERIMENT.parse_arguments(arguments).window_mel_frames, 2001)
            output.mkdir()
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                EXPERIMENT.parse_arguments(arguments)

    def test_multifunction_palettization_requires_compressed_sources(self):
        with self.assertRaisesRegex(ValueError, "Palettize each source route"):
            EXPERIMENT.merge_encoders(types.SimpleNamespace(nbits=6))

    def test_fp32_exports_preserve_uncompressed_precision(self):
        ENCODER.validate_fixed_precision("fp32", 0)
        ENCODER.validate_fixed_precision("fp16", 6)
        with self.assertRaisesRegex(ValueError, "--nbits 0"):
            ENCODER.validate_fixed_precision("fp32", 6)

    def setUp(self):
        self.weights, self.meta = toy_encoder_weights()
        self.meta.update({"parakeet.model_variant": ENCODER.EU_VARIANT,
                          "parakeet.eu.adapter_layers": [0, 1],
                          "parakeet.eu.gate_layer": 0})
        for index in range(2):
            for name, shape in (("down.weight", (3, 8)), ("down.bias", (3,)),
                                ("up.weight", (8, 3)), ("up.bias", (8,))):
                self.weights[f"eu.adapter.{index}.{name}"] = torch.randn(*shape) * 0.2

    def test_full_eu_requires_explicit_export_route(self):
        with self.assertRaisesRegex(ValueError, "requires --eu-route"):
            ENCODER.AneEncoder(REFERENCE, self.weights, self.meta, 64)
        with self.assertRaisesRegex(ValueError, "another model"):
            ENCODER.validate_eu_route({}, "other")
        with self.assertRaisesRegex(ValueError, "Invalid"):
            ENCODER.validate_eu_route(self.meta, "invalid")

    def test_all_routes_match_reference_with_partial_and_full_windows(self):
        mel = torch.randn(8, 64)
        outputs = {}
        for route in ("en", "other", "gate"):
            model = ENCODER.AneEncoder(REFERENCE, self.weights, self.meta, 64,
                                       eu_route=route).eval()
            self.assertEqual(model.n_layers, 1 if route == "gate" else 2)
            for valid in (1, 7, 8, 17, 63, 64):
                with self.subTest(route=route, valid=valid):
                    want = ENCODER.reference_frames(REFERENCE, self.weights, self.meta,
                                                     mel, valid, route)
                    got = ENCODER.ane_encoder_frames(model, mel, valid)
                    self.assertLess(ENCODER.relative_error(got, want),
                                    ENCODER.PARITY_REL_TOLERANCE)
            outputs[route] = ENCODER.ane_encoder_frames(model, mel)
        self.assertGreater(float((outputs["other"] - outputs["en"]).abs().max()), 0.01)
        self.assertGreater(float((outputs["gate"] - outputs["en"]).abs().max()), 0.01)

    def test_unwindowed_flexible_routes_match_reference_at_different_lengths(self):
        self.meta["parakeet.model.type"] = "tdt"
        for route in ("en", "other"):
            meta = dict(self.meta, **{"coreml.eu_route": route})
            self.assertEqual(GENERIC.validate_export_contract(meta, flexible=True, eu_route=route), "tdt")
            model = GENERIC.EncoderModule(REFERENCE, self.weights, meta, flexible=True,
                                           pe_table=REFERENCE.sinusoidal_rel_pe(64, 8)).eval()
            for frames in (9, 17, 33, 64, 129):
                with self.subTest(route=route, frames=frames), torch.inference_mode():
                    mel = torch.randn(8, frames)
                    want = REFERENCE.encoder_forward(mel, self.weights, meta,
                                                       eu_use_adapters=route == "other")[0]
                    self.assertLess(ENCODER.relative_error(model(mel), want),
                                    ENCODER.PARITY_REL_TOLERANCE)

    def test_flexible_export_preserves_numerics_and_input_range(self):
        meta = dict(self.meta, **{"parakeet.model.type": "tdt", "coreml.eu_route": "other"})
        model = GENERIC.EncoderModule(REFERENCE, self.weights, meta, flexible=True,
                                       pe_table=REFERENCE.sinusoidal_rel_pe(64, 8)).eval()
        exported = GENERIC.trace_flexible_encoder(model, torch.zeros(8, 32), 9, 257)
        converted = exported.module()
        for length in (9, 17, 33, 64, 129, 257):
            mel = torch.randn(8, length)
            with torch.inference_mode():
                torch.testing.assert_close(converted(mel), model(mel))
        with self.assertRaises(RuntimeError):
            converted(torch.zeros(8, 8))


class EncoderHelperTests(unittest.TestCase):
    def test_default_window_is_fifteen_seconds(self):
        self.assertEqual(ENCODER.DEFAULT_WINDOW_MEL_FRAMES, 1500)

    def test_encoder_frames_for_offline_window(self):
        # Three stride-2 convs, kernel 3, pad 1: 1500 -> 750 -> 375 -> 188.
        self.assertEqual(ENCODER.encoder_frames_for_mel(1500, causal=False), 188)
        self.assertEqual(ENCODER.encoder_frames_for_mel(1501, causal=False), 188)
        self.assertEqual(ENCODER.encoder_frames_for_mel(326, causal=False), 41)

    def test_encoder_frames_for_causal_window(self):
        self.assertEqual(ENCODER.encoder_frames_for_mel(1500, causal=True), 189)

    def test_channel_norm_matches_layer_norm_over_channels(self):
        torch.manual_seed(0)
        x = torch.randn(1, 6, 1, 5)
        gamma = torch.randn(1, 6, 1, 1)
        beta = torch.randn(1, 6, 1, 1)
        got = ENCODER.channel_norm(x, gamma, beta, eps=1e-5)
        want = F.layer_norm(x.permute(0, 2, 3, 1), (6,), gamma.reshape(6), beta.reshape(6),
                            eps=1e-5).permute(0, 3, 1, 2)
        self.assertTrue(torch.allclose(got, want, atol=1e-5))

    def test_key_bias_pad_is_finite(self):
        self.assertTrue(np.isfinite(np.float16(ENCODER.KEY_BIAS_PAD)))

    def test_halved_mask_keeps_exactly_the_valid_frames(self):
        # Every valid length v: the conv keeps floor((v - 1) / 2) + 1 frames,
        # and the even-index slice of a v-frame mask has that many ones.
        for valid in range(1, 40):
            mask = torch.zeros(1, 1, 40, 1)
            mask[:, :, :valid] = 1.0
            halved = ENCODER.AneEncoder.halve_mask(mask)
            self.assertEqual(int(halved.sum()), ENCODER.conv_out_len(valid), valid)
            self.assertEqual(halved.shape[2], ENCODER.conv_out_len(40))


class EncoderGraphTests(unittest.TestCase):
    WINDOW = 64

    def setUp(self):
        self.weights, self.meta = toy_encoder_weights()
        self.model = ENCODER.AneEncoder(REFERENCE, self.weights, self.meta, self.WINDOW).eval()

    def test_full_window_matches_reference(self):
        torch.manual_seed(0)
        mel = torch.randn(self.model.n_mels, self.WINDOW)
        self.assertTrue(ENCODER.compare_with_reference(
            self.model, REFERENCE, self.weights, self.meta, mel, "full"))

    def test_partial_window_valid_frames_match_short_input(self):
        # The valid frames of a padded, masked window must equal the reference
        # run on the short input alone; without masks at every subsampling
        # resolution the last valid frames absorb the padding.
        torch.manual_seed(1)
        for valid in (13, 22, 37, 51):
            mel = torch.randn(self.model.n_mels, valid)
            self.assertTrue(ENCODER.compare_with_reference(
                self.model, REFERENCE, self.weights, self.meta, mel, f"partial {valid}"), valid)

    def test_trailing_padding_frame_is_masked_at_the_valid_length(self):
        # The mel front-end pads the last frame with zeros; the engine passes
        # the audio length to the wrapper, which must mask from there. Marking
        # the padding frame valid lets its conv activations reach the last
        # audio frames.
        torch.manual_seed(3)
        padded = torch.randn(self.model.n_mels, 37)
        padded[:, -1] = 0.0
        want = ENCODER.reference_frames(REFERENCE, self.weights, self.meta, padded, valid_mel=36)
        masked_at_audio = ENCODER.ane_encoder_frames(self.model, padded, valid_mel=36)
        padding_marked_valid = ENCODER.ane_encoder_frames(self.model, padded, valid_mel=37)
        self.assertLess(ENCODER.relative_error(masked_at_audio, want), ENCODER.PARITY_REL_TOLERANCE)
        self.assertGreater(ENCODER.relative_error(padding_marked_valid[: want.shape[0]], want),
                           ENCODER.PARITY_REL_TOLERANCE)

    def test_check_parity_covers_full_and_partial_windows(self):
        self.assertTrue(ENCODER.check_parity(self.model, REFERENCE, self.weights, self.meta))


def toy_tdt_weights(h_pred=4, h_joint=3, d_enc=5, vocab=6, n_durations=3, layers=2):
    torch.manual_seed(1)
    weights = {}
    for layer in range(layers):
        p = f"tdt.predict.lstm.{layer}"
        weights[f"{p}.w_ih"] = torch.randn(4 * h_pred, h_pred) * 0.3
        weights[f"{p}.w_hh"] = torch.randn(4 * h_pred, h_pred) * 0.3
        weights[f"{p}.b_ih"] = torch.randn(4 * h_pred) * 0.1
        weights[f"{p}.b_hh"] = torch.randn(4 * h_pred) * 0.1
    weights["tdt.predict.embed.weight"] = torch.randn(vocab + 1, h_pred)
    weights["tdt.joint.enc.weight"] = torch.randn(h_joint, d_enc) * 0.3
    weights["tdt.joint.enc.bias"] = torch.randn(h_joint) * 0.1
    weights["tdt.joint.pred.weight"] = torch.randn(h_joint, h_pred) * 0.3
    weights["tdt.joint.pred.bias"] = torch.randn(h_joint) * 0.1
    weights["tdt.joint.out.weight"] = torch.randn(vocab + 1 + n_durations, h_joint) * 0.3
    weights["tdt.joint.out.bias"] = torch.randn(vocab + 1 + n_durations) * 0.1
    meta = {
        "parakeet.tdt.pred_rnn_layers": layers,
        "parakeet.tdt.pred_hidden": h_pred,
        "parakeet.tdt.joint_hidden": h_joint,
        "parakeet.tdt.vocab_size": vocab,
        "parakeet.tdt.blank_id": vocab,
        "parakeet.tdt.num_durations": n_durations,
    }
    return weights, meta


class DecoderExporterTests(unittest.TestCase):
    def test_geometry_reads_the_checkpoint(self):
        weights, meta = toy_tdt_weights()
        geom = DECODER.TdtGeometry(weights, meta)
        self.assertEqual((geom.n_layers, geom.h_pred, geom.d_enc, geom.v_out), (2, 4, 5, 10))
        self.assertEqual(geom.blank_id, 6)

    def test_geometry_rejects_wrong_joint_output(self):
        weights, meta = toy_tdt_weights()
        meta["parakeet.tdt.num_durations"] = 2
        with self.assertRaises(RuntimeError):
            DECODER.TdtGeometry(weights, meta)

    def test_geometry_rejects_missing_lstm_layer(self):
        weights, meta = toy_tdt_weights()
        meta["parakeet.tdt.pred_rnn_layers"] = 3
        with self.assertRaises(RuntimeError):
            DECODER.TdtGeometry(weights, meta)

    def test_graph_matches_scalar_reference(self):
        weights, meta = toy_tdt_weights()
        geom = DECODER.TdtGeometry(weights, meta)
        model = DECODER.AneTdtDecoder(weights, geom, lanes=3, width=1).eval()
        self.assertTrue(DECODER.check_parity(model, weights, geom))

    def test_default_lanes_and_width(self):
        self.assertEqual(DECODER.DEFAULT_LANES, 16)
        self.assertEqual(DECODER.DEFAULT_WIDTH, 1)


class DecoderPalettizationTests(unittest.TestCase):
    def test_joint_output_group_divides_real_and_other_channel_counts(self):
        for channels, preferred, expected in ((8198, 32, 2), (8198, 16, 2),
                                               (640, 16, 16), (8199, 16, 1)):
            self.assertEqual(DECODER.joint_out_group_size(channels, preferred), expected)
        for channels, preferred in ((0, 16), (8198, 0), (-1, 16)):
            with self.assertRaises(ValueError):
                DECODER.joint_out_group_size(channels, preferred)

    @staticmethod
    def make_model():
        torch.manual_seed(10)
        model = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 1), torch.nn.ReLU(),
                                    torch.nn.Conv2d(64, 70, 1)).eval()
        traced = torch.jit.trace(model, torch.zeros(1, 64, 1, 1))
        return DECODER.ct.convert(
            traced, inputs=[DECODER.ct.TensorType(shape=(1, 64, 1, 1))],
            compute_precision=DECODER.ct.precision.FLOAT16,
            minimum_deployment_target=DECODER.ct.target.iOS18, skip_model_load=True)

    @unittest.skipUnless(hasattr(DECODER.ct, "convert"), "coremltools is not installed")
    def test_grouped_quantization_reaches_joint_output(self):
        model = self.make_model()
        result, names, group = DECODER.palettize_decoder_weights(model, 70, 6, 6, 16)
        self.assertEqual(group, 2)
        DECODER.verify_joint_out_palettization(result, names, 6, 2)
        with self.assertRaisesRegex(RuntimeError, "requested 4-bit"):
            DECODER.verify_joint_out_palettization(result, names, 4, 2)
        with self.assertRaisesRegex(RuntimeError, "requested 6-bit/group 4"):
            DECODER.verify_joint_out_palettization(result, names, 6, 4)

    @unittest.skipUnless(hasattr(DECODER.ct, "convert"), "coremltools is not installed")
    def test_output_only_and_fp16_output_modes(self):
        for nbits, nbits_out in ((0, 6), (6, 0)):
            with self.subTest(nbits=nbits, nbits_out=nbits_out):
                result, names, group = DECODER.palettize_decoder_weights(
                    self.make_model(), 70, nbits, nbits_out, 16)
                DECODER.verify_joint_out_palettization(result, names, nbits_out, group)
                other = [op for func in result._mil_program.functions.values()
                         for op in func.operations if op.op_type == "conv" and op.name not in names]
                self.assertEqual(len(other), 1)
                self.assertEqual(other[0].weight.op.op_type,
                                 "const" if nbits == 0 else "constexpr_lut_to_dense")

    @unittest.skipUnless(hasattr(DECODER.ct, "convert"), "coremltools is not installed")
    def test_cli_output_only_option_emits_a_compressed_package(self):
        weights, meta = toy_tdt_weights(h_pred=64, h_joint=64, d_enc=64,
                                       vocab=66, n_durations=3, layers=1)
        reference = types.SimpleNamespace(load_gguf=lambda path: (weights, meta))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "decoder.mlpackage"
            argv = ["export-decoder", "--gguf", "unused.gguf", "--out", str(output),
                    "--lanes", "1", "--nbits", "0", "--nbits-out", "6"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(DECODER, "_load_sibling", side_effect=[ENCODER, reference]):
                self.assertEqual(DECODER.main(), 0)
            model = DECODER.ct.models.MLModel(str(output), skip_model_load=True)
            block = next(iter(model.get_spec().mlProgram.functions["main"].block_specializations.values()))
            self.assertEqual(sum(op.type == "constexpr_lut_to_dense" for op in block.operations), 1)

    @unittest.skipUnless(hasattr(DECODER.ct, "convert"), "coremltools is not installed")
    def test_silently_skipped_or_missing_projection_fails(self):
        model = self.make_model()
        with mock.patch("coremltools.optimize.coreml.palettize_weights", return_value=model):
            with self.assertRaisesRegex(RuntimeError, "was not palettized"):
                DECODER.palettize_decoder_weights(model, 70, 6, 6, 16)
        with self.assertRaisesRegex(RuntimeError, "projection missing"):
            DECODER.palettize_decoder_weights(model, 71, 6, 6, 16)


if __name__ == "__main__":
    unittest.main()
