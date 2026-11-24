#!/usr/bin/env python3
"""Model-free checks for the Parakeet Core ML exporter helpers."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import gguf
except ModuleNotFoundError:
    gguf = None

try:
    import torch
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

SCRIPT = Path(__file__).parents[1] / "scripts" / "export-encoder-coreml.py"
SPEC = importlib.util.spec_from_file_location("export_encoder_coreml", SCRIPT)
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


def eou_meta():
    return {
        "parakeet.model.type": "eou",
        "parakeet.encoder.causal_downsampling": True,
        "parakeet.encoder.conv_context_size": "causal",
        "parakeet.encoder.conv_norm_type": "layer_norm",
        "parakeet.encoder.att_context_style": "chunked_limited",
        "parakeet.encoder.att_context_size_left": 2,
        "parakeet.encoder.att_context_size_right": 1,
    }


def rnnt_meta():
    return {
        "parakeet.model.type": "rnnt",
        "parakeet.encoder.causal_downsampling": False,
        "parakeet.encoder.conv_context_size": "default",
        "parakeet.encoder.conv_norm_type": "batch_norm",
        # Unified preserves its dynamic-chunk training style in metadata, but
        # the runtime executes the unbounded offline inference contract.
        "parakeet.encoder.att_context_style": "chunked_limited_with_rc",
        "parakeet.encoder.att_context_size_left": -1,
        "parakeet.encoder.att_context_size_right": -1,
    }


def indic_ctc_meta():
    return {
        "parakeet.model.type": "ctc",
        "parakeet.encoder.causal_downsampling": False,
        "parakeet.encoder.conv_context_size": "default",
        "parakeet.encoder.conv_norm_type": "batch_norm",
        "parakeet.encoder.att_context_style": "regular",
        "parakeet.encoder.att_context_size_left": -1,
        "parakeet.encoder.att_context_size_right": -1,
    }


def sortformer_v2_1_meta():
    return {
        "parakeet.model.type": "sortformer",
        "parakeet.model_variant": "sortformer-streaming-v2.1-aosc",
        "parakeet.encoder.d_model": 4,
        "parakeet.encoder.n_layers": 0,
        "parakeet.encoder.n_heads": 1,
        "parakeet.encoder.pos_emb_max_len": 32,
        "parakeet.encoder.xscaling": True,
        "parakeet.encoder.causal_downsampling": False,
        "parakeet.encoder.conv_context_size": "default",
        "parakeet.encoder.att_context_style": "regular",
        "parakeet.encoder.att_context_size_left": -1,
        "parakeet.encoder.att_context_size_right": -1,
    }


def nemotron_meta():
    return {
        "parakeet.model.type": "nemotron",
        "parakeet.encoder.causal_downsampling": True,
        "parakeet.encoder.conv_context_size": "causal",
        "parakeet.encoder.conv_norm_type": "layer_norm",
        "parakeet.encoder.att_context_style": "chunked_limited",
        "parakeet.encoder.att_context_size_left": 56,
        "parakeet.encoder.att_context_size_right": 3,
    }


class ExportContractTests(unittest.TestCase):
    def test_full_eu_flexible_precision_avoids_nonfinite_fp16_execution(self):
        meta = {"parakeet.model_variant": "transcribepsy-250m-eu"}
        self.assertEqual(EXPORTER.resolve_export_precision(meta, None, True), "fp32")
        self.assertEqual(EXPORTER.resolve_export_precision(meta, "fp32", True), "fp32")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            EXPORTER.resolve_export_precision(meta, "fp16", True)
        self.assertEqual(EXPORTER.resolve_export_precision({}, None, True), "fp16")

    def test_flexible_input_minimum_includes_single_encoder_row(self):
        if not hasattr(EXPORTER.ct, "proto"):
            self.skipTest("Core ML protobuf definitions unavailable")
        spec = EXPORTER.ct.proto.Model_pb2.Model()
        feature = spec.description.input.add()
        feature.name = "mel"
        for lower, upper in ((128, 128), (9, 40000)):
            size_range = feature.type.multiArrayType.shapeRange.sizeRanges.add()
            size_range.lowerBound, size_range.upperBound = lower, upper
        mask = spec.description.input.add()
        mask.name = "mel_valid_mask"
        mask_range = mask.type.multiArrayType.shapeRange.sizeRanges.add()
        mask_range.lowerBound, mask_range.upperBound = 9, 40000
        EXPORTER.set_flexible_input_minimum(spec, 1)
        self.assertEqual(mask_range.lowerBound, 1)
        ranges = feature.type.multiArrayType.shapeRange.sizeRanges
        self.assertEqual((ranges[0].lowerBound, ranges[1].lowerBound, ranges[1].upperBound),
                         (128, 1, 40000))
        with self.assertRaises(ValueError):
            EXPORTER.set_flexible_input_minimum(spec, 0)

    def test_full_eu_flexible_accepts_all_routes(self):
        meta = {"parakeet.model_variant": "transcribepsy-250m-eu", "parakeet.model.type": "tdt",
                "parakeet.encoder.conv_norm_type": "layer_norm"}
        for route in ("en", "other", "gate"):
            self.assertEqual(EXPORTER.validate_export_contract(meta, flexible=True, eu_route=route), "tdt")

    def test_full_eu_requires_route_aware_exporter(self):
        with self.assertRaisesRegex(ValueError, "--eu-route"):
            EXPORTER.validate_export_contract({"parakeet.model_variant": "transcribepsy-250m-eu"})

    def test_accepts_indic_ctc_fixed_and_flexible(self):
        self.assertEqual(
            EXPORTER.validate_export_contract(indic_ctc_meta()), "ctc")
        self.assertEqual(
            EXPORTER.validate_export_contract(indic_ctc_meta(), flexible=True),
            "ctc")

    def test_rejects_incompatible_indic_ctc_metadata(self):
        invalid_cases = (
            ("parakeet.encoder.causal_downsampling", True,
             "non-causal downsampling"),
            ("parakeet.encoder.conv_context_size", "causal",
             "non-causal convolution"),
            ("parakeet.encoder.conv_norm_type", "layer_norm",
             "batch-normalized convolution"),
            ("parakeet.encoder.att_context_size_left", 70,
             "unbounded attention context"),
            ("parakeet.encoder.att_context_size_right", 1,
             "unbounded attention context"),
        )
        for key, value, message in invalid_cases:
            with self.subTest(key=key, value=value):
                meta = indic_ctc_meta()
                meta[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    EXPORTER.validate_export_contract(meta)

    def test_accepts_offline_rnnt_fixed_and_flexible(self):
        self.assertEqual(
            EXPORTER.validate_export_contract(rnnt_meta()), "rnnt")
        self.assertEqual(
            EXPORTER.validate_export_contract(rnnt_meta(), flexible=True),
            "rnnt")

    def test_rejects_incompatible_rnnt_metadata(self):
        invalid_cases = (
            ("parakeet.encoder.causal_downsampling", True,
             "non-causal downsampling"),
            ("parakeet.encoder.conv_context_size", "causal",
             "non-causal convolution"),
            ("parakeet.encoder.conv_norm_type", "layer_norm",
             "batch-normalized convolution"),
            ("parakeet.encoder.att_context_size_left", 70,
             "unbounded attention context"),
            ("parakeet.encoder.att_context_size_right", 1,
             "unbounded attention context"),
        )
        for key, value, message in invalid_cases:
            with self.subTest(key=key, value=value):
                meta = rnnt_meta()
                meta[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    EXPORTER.validate_export_contract(meta)

    def test_accepts_fixed_eou_and_rejects_flexible(self):
        self.assertEqual(EXPORTER.validate_export_contract(eou_meta()), "eou")
        with self.assertRaisesRegex(ValueError, "fixed shape"):
            EXPORTER.validate_export_contract(eou_meta(), flexible=True)

    def test_accepts_fixed_nemotron_and_rejects_flexible(self):
        self.assertEqual(
            EXPORTER.validate_export_contract(nemotron_meta()), "nemotron")
        with self.assertRaisesRegex(ValueError, "fixed shape"):
            EXPORTER.validate_export_contract(nemotron_meta(), flexible=True)

    def test_rejects_incompatible_nemotron_metadata(self):
        meta = nemotron_meta()
        meta["parakeet.encoder.conv_norm_type"] = "batch_norm"
        with self.assertRaisesRegex(ValueError, "conv_norm_type"):
            EXPORTER.validate_export_contract(meta)

    def test_accepts_sortformer_v2_1_full_and_bypass_exports(self):
        meta = sortformer_v2_1_meta()
        self.assertEqual(EXPORTER.validate_export_contract(meta), "sortformer")
        self.assertEqual(
            EXPORTER.validate_export_contract(meta, bypass_pre_encode=True),
            "sortformer")

    def test_rejects_flexible_sortformer_export(self):
        with self.assertRaisesRegex(ValueError, "fixed shape"):
            EXPORTER.validate_export_contract(sortformer_v2_1_meta(), flexible=True)

    def test_rejects_other_sortformer_variants(self):
        for variant in ("", "sortformer-v1", "sortformer-streaming-v2"):
            with self.subTest(variant=variant):
                meta = sortformer_v2_1_meta()
                meta["parakeet.model_variant"] = variant
                with self.assertRaisesRegex(ValueError, "sortformer-streaming-v2.1-aosc"):
                    EXPORTER.validate_export_contract(meta)

    def test_rejects_non_sortformer_bypass_export(self):
        for meta in (rnnt_meta(), eou_meta()):
            with self.subTest(model_type=meta["parakeet.model.type"]):
                with self.assertRaisesRegex(ValueError, "only for Sortformer"):
                    EXPORTER.validate_export_contract(
                        meta, bypass_pre_encode=True)

    def test_rejects_incompatible_sortformer_encoder_metadata(self):
        invalid_cases = (
            ("parakeet.encoder.causal_downsampling", True, "non-causal downsampling"),
            ("parakeet.encoder.conv_context_size", "causal", "non-causal convolution"),
            ("parakeet.encoder.att_context_style", "chunked_limited", "full-context attention"),
            ("parakeet.encoder.att_context_size_right", 1, "unbounded attention context"),
        )
        for key, value, message in invalid_cases:
            with self.subTest(key=key, value=value):
                meta = sortformer_v2_1_meta()
                meta[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    EXPORTER.validate_export_contract(meta)

    def test_rejects_incompatible_eou_metadata(self):
        meta = eou_meta()
        meta["parakeet.encoder.conv_norm_type"] = "batch_norm"
        with self.assertRaisesRegex(ValueError, "conv_norm_type"):
            EXPORTER.validate_export_contract(meta)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_prepare_bypass_encoder_inputs_builds_masks(self):
        pre_encode = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        valid_mask = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)
        encoder_input, sequence_mask, attention_mask = (
            EXPORTER.prepare_bypass_encoder_inputs(pre_encode, valid_mask))

        self.assertEqual(tuple(encoder_input.shape), (1, 3, 4))
        torch.testing.assert_close(encoder_input[0], pre_encode.transpose(0, 1))
        self.assertEqual(tuple(sequence_mask.shape), (1, 3, 1))
        torch.testing.assert_close(sequence_mask[0, :, 0], valid_mask)
        torch.testing.assert_close(
            attention_mask[0, 0, 0], torch.tensor([0.0, 0.0, -1.0e4]))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_bypass_module_preserves_time_and_feature_orientation(self):
        class Ref:
            @staticmethod
            def sinusoidal_rel_pe(max_len, d_model, dtype):
                return torch.zeros(1, 2 * max_len - 1, d_model, dtype=dtype)

        module = EXPORTER.BypassEncoderModule(
            ref=Ref(), weights={}, meta=sortformer_v2_1_meta()).eval()
        pre_encode = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        output = module(pre_encode, torch.ones(3, dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (3, 4))
        torch.testing.assert_close(output, pre_encode.transpose(0, 1) * 2.0)

        padded = module(
            pre_encode, torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32))
        torch.testing.assert_close(padded[:2], output[:2])
        torch.testing.assert_close(padded[2], torch.zeros(4))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_chunked_attention_visibility(self):
        mask = EXPORTER.chunked_attention_mask(6, 2, 1, torch.float32, "cpu")[0, 0]
        self.assertEqual(mask[0, 0].item(), 0.0)
        self.assertEqual(mask[0, 1].item(), 0.0)
        self.assertLess(mask[0, 2].item(), -1e20)
        self.assertEqual(mask[3, 2].item(), 0.0)
        self.assertEqual(mask[3, 3].item(), 0.0)
        self.assertLess(mask[3, 4].item(), -1e20)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_flexible_subsampling_preserves_native_padding_boundaries(self):
        reference = EXPORTER.load_reference_encoder(Path(__file__).parents[1] / "scripts")
        weights = {
            "encoder.subsampling.conv0.weight": torch.randn(2, 1, 3, 3),
            "encoder.subsampling.conv0.bias": torch.randn(2),
            "encoder.subsampling.conv1_dw.weight": torch.randn(2, 1, 3, 3),
            "encoder.subsampling.conv1_dw.bias": torch.randn(2),
            "encoder.subsampling.conv1_pw.weight": torch.randn(2, 2, 1, 1),
            "encoder.subsampling.conv1_pw.bias": torch.randn(2),
            "encoder.subsampling.conv2_dw.weight": torch.randn(2, 1, 3, 3),
            "encoder.subsampling.conv2_dw.bias": torch.randn(2),
            "encoder.subsampling.conv2_pw.weight": torch.randn(2, 2, 1, 1),
            "encoder.subsampling.conv2_pw.bias": torch.randn(2),
            "encoder.subsampling.out.weight": torch.randn(4, 2),
            "encoder.subsampling.out.bias": torch.randn(4),
        }
        for frames in (399, 400, 401, 402, 407, 408, 409):
            for valid in (frames, frames - 1):
                with self.subTest(frames=frames, valid=valid):
                    mel = torch.randn(8, frames)
                    mask = (torch.arange(frames) < valid).float()
                    actual = EXPORTER.subsampling_shape_generic(mel, weights, mask)
                    expected, _ = reference.subsampling(mel, weights, valid)
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(actual.shape[1], (frames + 7) // 8)

    def test_causal_subsampling_dimensions(self):
        weights = {
            "encoder.subsampling.conv0.weight": torch.zeros(2, 1, 3, 3),
            "encoder.subsampling.conv0.bias": torch.zeros(2),
            "encoder.subsampling.conv1_dw.weight": torch.zeros(2, 1, 3, 3),
            "encoder.subsampling.conv1_dw.bias": torch.zeros(2),
            "encoder.subsampling.conv1_pw.weight": torch.zeros(2, 2, 1, 1),
            "encoder.subsampling.conv1_pw.bias": torch.zeros(2),
            "encoder.subsampling.conv2_dw.weight": torch.zeros(2, 1, 3, 3),
            "encoder.subsampling.conv2_dw.bias": torch.zeros(2),
            "encoder.subsampling.conv2_pw.weight": torch.zeros(2, 2, 1, 1),
            "encoder.subsampling.conv2_pw.bias": torch.zeros(2),
            "encoder.subsampling.out.weight": torch.zeros(4, 4),
            "encoder.subsampling.out.bias": torch.zeros(4),
        }
        out = EXPORTER.causal_subsampling(torch.zeros(8, 9), weights)
        self.assertEqual(tuple(out.shape), (1, 2, 4))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_causal_layer_norm_convolution_shape(self):
        class Ref:
            @staticmethod
            def layer_norm(x, weight, bias):
                return torch.nn.functional.layer_norm(
                    x, (x.size(-1),), weight=weight, bias=bias)

        weights = {
            "conv.pw1.weight": torch.zeros(4, 2, 1),
            "conv.pw1.bias": torch.zeros(4),
            "conv.dw.weight": torch.zeros(2, 1, 3),
            "conv.dw.bias": torch.zeros(2),
            "conv.norm.weight": torch.ones(2),
            "conv.norm.bias": torch.zeros(2),
            "conv.pw2.weight": torch.zeros(2, 2, 1),
            "conv.pw2.bias": torch.zeros(2),
        }
        out = EXPORTER.conformer_conv(
            Ref(), torch.zeros(1, 5, 2), weights, "conv",
            causal=True, layer_norm=True)
        self.assertEqual(tuple(out.shape), (1, 5, 2))


@unittest.skipIf(gguf is None, "gguf is not installed")
class NativePaddingExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_path = Path(__file__).with_name("test_export_coreml_ane.py")
        spec = importlib.util.spec_from_file_location("coreml_ane_fixtures", fixture_path)
        cls.fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fixtures)
        cls.reference = cls.fixtures.REFERENCE

    def setUp(self):
        self.weights, self.meta = self.fixtures.toy_encoder_weights()
        self.meta.update({
            "parakeet.model.type": "tdt",
            "parakeet.model_variant": "transcribepsy-250m-eu",
            "parakeet.eu.adapter_layers": [0, 1],
            "parakeet.eu.gate_layer": 0,
        })
        for index in self.meta["parakeet.eu.adapter_layers"]:
            for name, shape in (("down.weight", (3, 8)), ("down.bias", (3,)),
                                ("up.weight", (8, 3)), ("up.bias", (8,))):
                self.weights[f"eu.adapter.{index}.{name}"] = torch.randn(*shape) * 0.2

    def make_model(self, meta):
        return EXPORTER.NativePaddingFlexibleEncoder(
            self.reference, self.weights, meta, flexible=True,
            pe_table=self.reference.sinusoidal_rel_pe(64, 8)).eval()

    def test_two_input_dynamic_export_matches_native_padding_reference(self):
        for route in ("en", "other"):
            meta = dict(self.meta, **{"coreml.eu_route": route})
            model = self.make_model(meta)
            exported = EXPORTER.trace_flexible_encoder(model, torch.zeros(8, 32), 9, 409)
            program = exported.module()
            for frames, valid in ((9, 1), (17, 8), (33, 17), (399, 398),
                                  (400, 399), (401, 400), (409, 408)):
                with self.subTest(route=route, frames=frames, valid=valid), torch.no_grad():
                    mel = torch.randn(8, frames)
                    mask = (torch.arange(frames) < valid).float()
                    expected = self.reference.encoder_forward(
                        mel, self.weights, meta, mel_valid_len=valid,
                        eu_use_adapters=route == "other")[0]
                    torch.testing.assert_close(model(mel, mask), expected)
                    actual = program(mel, mask)
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(tuple(actual.shape), ((frames + 7) // 8, 8))
                    changed_padding = mel.clone()
                    changed_padding[:, valid:] += 100.0
                    torch.testing.assert_close(program(changed_padding, mask), actual)
            with self.assertRaises(RuntimeError):
                program(torch.zeros(8, 17), torch.ones(16))
            with self.assertRaises(RuntimeError):
                program(torch.zeros(8, 410), torch.ones(410))

    def test_frontend_extra_row_keeps_projection_bias_and_attends(self):
        meta = dict(self.meta, **{"coreml.eu_route": "en"})
        mel = torch.randn(8, 401)
        mask = (torch.arange(401) < 400).float()
        projected = EXPORTER.subsampling_shape_generic(mel, self.weights, mask)
        self.assertEqual(tuple(projected.shape), (1, 51, 8))
        torch.testing.assert_close(projected[0, -1], self.weights["encoder.subsampling.out.bias"])
        with torch.no_grad():
            padded = self.make_model(meta)(mel, mask)
            expected = self.reference.encoder_forward(mel, self.weights, meta, mel_valid_len=400)[0]
            trimmed = self.reference.encoder_forward(mel[:, :400], self.weights, meta)[0]
        torch.testing.assert_close(padded, expected)
        self.assertEqual(tuple(padded.shape), (51, 8))
        self.assertGreater(float((padded[:50] - trimmed).abs().max()), 1.0e-5)
        self.assertGreater(float(padded[-1].abs().max()), 0.1)

    def test_main_gate_route_truncates_graph_and_saved_metadata(self):
        full_meta = dict(self.meta)
        expected_meta = dict(self.meta, **{"parakeet.encoder.n_layers": 1})
        saved_model = types.SimpleNamespace(user_defined_metadata={}, save=mock.Mock())

        def convert_gate(ref, weights, meta, example, *args):
            self.assertEqual(meta["coreml.eu_route"], "gate")
            self.assertEqual(meta["parakeet.encoder.n_layers"], 1)
            with torch.no_grad():
                actual = self.make_model(meta)(example, torch.ones(example.shape[1]))
                expected = ref.encoder_forward(example, weights, expected_meta)[0]
                full = ref.encoder_forward(example, weights, full_meta)[0]
            torch.testing.assert_close(actual, expected)
            self.assertGreater(float((actual - full).abs().max()), 0.01)
            return saved_model

        argv = [str(SCRIPT), "--gguf", "synthetic.gguf", "--out", "gate.mlpackage",
                "--n-mel-frames", "33", "--flexible", "--eu-route", "gate"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(EXPORTER, "load_reference_encoder", return_value=self.reference), \
                mock.patch.object(self.reference, "load_gguf", return_value=(self.weights, self.meta)), \
                mock.patch.object(EXPORTER, "convert_flexible", side_effect=convert_gate), \
                mock.patch.object(EXPORTER.ct, "precision", types.SimpleNamespace(
                    FLOAT16="fp16", FLOAT32="fp32"), create=True), \
                mock.patch.object(EXPORTER.ct, "target", types.SimpleNamespace(
                    macOS13=13, macOS15=15), create=True):
            EXPORTER.main()
        metadata = saved_model.user_defined_metadata
        self.assertEqual(metadata["parakeet.eu.route"], "gate")
        self.assertEqual(metadata["parakeet.encoder.n_layers"], "1")
        self.assertEqual(metadata["parakeet.encoder.precision"], "fp32")
        self.assertEqual(metadata["parakeet.eu.native_padding"], "true")
        saved_model.save.assert_called_once_with("gate.mlpackage")

    def test_gguf_integer_adapter_arrays_round_trip(self):
        cases = ((gguf.GGUFValueType.INT32, [-1, 0, 1, 2147483647]),
                 (gguf.GGUFValueType.UINT32, [0, 1, 2147483648, 4294967295]))
        for element_type, values in cases:
            with self.subTest(element_type=element_type), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "metadata.gguf"
                writer = gguf.GGUFWriter(path, "parakeet")
                writer.add_key_value("parakeet.eu.adapter_layers", values,
                                     gguf.GGUFValueType.ARRAY, sub_type=element_type)
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                field = gguf.GGUFReader(path).fields["parakeet.eu.adapter_layers"]
                self.assertEqual(field.types, [gguf.GGUFValueType.ARRAY, element_type])
                weights, metadata = self.reference.load_gguf(path)
                self.assertEqual(weights, {})
                self.assertEqual(metadata["parakeet.eu.adapter_layers"], values)
                self.assertTrue(all(type(value) is int for value in metadata["parakeet.eu.adapter_layers"]))


if __name__ == "__main__":
    unittest.main()
