#!/usr/bin/env python3
"""Unit checks of coreml_template.py.

The pure parts (MILBlob parsing, blob matching, recipe drafting, header
refresh, exporter arguments, template.json merging) need only numpy. The spec
comparison needs coremltools' protobuf classes, and the end-to-end `suggest`
and `extract` cases on the toy exporter also need gguf-py and
qvac-coreml-tool ($QVAC_COREML_TOOL).
Those cases are skipped when their requirements are missing, unless
QVAC_COREML_REQUIRE_COREMLTOOLS=1 (the template CI job), where a missing
requirement fails instead.
"""

import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coreml_template as ctpl  # noqa: E402

REQUIRE = os.environ.get("QVAC_COREML_REQUIRE_COREMLTOOLS") == "1"
HAVE_CT = importlib.util.find_spec("coremltools") is not None
HAVE_GGUF = importlib.util.find_spec("gguf") is not None
TOOL = os.environ.get("QVAC_COREML_TOOL", "")


def needs(cond, what):
    if cond:
        return lambda f: f
    if REQUIRE:
        def fail(self):
            self.fail(f"requirement missing: {what}")
        return lambda f: fail
    return unittest.skip(f"requirement missing: {what}")


def weight_bin(arrays):
    """A MILBlob v2 file holding `arrays` (fp16 or fp32), coremltools' layout."""
    out = bytearray(64)
    struct.pack_into("<II", out, 0, len(arrays), 2)
    for a in arrays:
        dtype = ctpl.MIL_FP16 if a.dtype == np.float16 else ctpl.MIL_FP32
        data = a.astype("<f2" if dtype == ctpl.MIL_FP16 else "<f4").tobytes()
        meta = len(out)
        hdr = bytearray(64)
        struct.pack_into("<IIQQ", hdr, 0, 0xDEADBEEF, dtype, len(data), meta + 64)
        out += hdr + data
        out += bytes((-len(out)) % 64)
    return bytes(out)


class WeightBinTest(unittest.TestCase):
    def test_round_trip(self):
        a = np.arange(5, dtype=np.float16)
        b = np.linspace(-1, 1, 7).astype(np.float32)
        blobs = ctpl.read_weight_bin(weight_bin([a, b]))
        self.assertEqual([x.offset for x in blobs], [64, 192])
        self.assertEqual([x.dtype_name for x in blobs], ["fp16", "fp32"])
        self.assertEqual([x.count for x in blobs], [5, 7])
        np.testing.assert_array_equal(blobs[0].array(), a)
        np.testing.assert_array_equal(blobs[1].array(), b)

    def test_malformed(self):
        good = bytearray(weight_bin([np.ones(4, np.float16)]))
        for mutate, msg in [
            (lambda d: struct.pack_into("<I", d, 4, 3), "version 3"),
            (lambda d: struct.pack_into("<I", d, 64, 0xCAFEBABE), "bad sentinel"),
            (lambda d: struct.pack_into("<Q", d, 64 + 8, 1 << 20), "has data offset"),
            (lambda d: struct.pack_into("<I", d, 0, 2), "past the end"),
            (lambda d: struct.pack_into("<I", d, 64 + 4, 7), "MIL dtype 7"),
        ]:
            d = bytearray(good)
            mutate(d)
            with self.assertRaisesRegex(ctpl.TemplateError, msg):
                ctpl.read_weight_bin(bytes(d))
        with self.assertRaises(ctpl.TemplateError):
            ctpl.read_weight_bin(b"\0" * 10)


def blob_of(arr, dtype=np.float16):
    return ctpl.read_weight_bin(weight_bin([np.asarray(arr).reshape(-1).astype(dtype)]))[0]


class ExplainTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.w = rng.standard_normal((4, 6)).astype(np.float32)
        self.v = rng.standard_normal(24).astype(np.float32)
        self.tensors = {"w": ("f32", self.w), "v": ("f32", self.v), "q": ("q8_0", None)}

    def test_cast(self):
        self.assertEqual(ctpl.explain_blob(blob_of(self.w), self.tensors)[0], 'f16(t("w"))')
        self.assertEqual(ctpl.explain_blob(blob_of(self.w, np.float32), self.tensors)[0], 'f32(t("w"))')

    def test_transpose(self):
        self.assertEqual(ctpl.explain_blob(blob_of(self.w.T), self.tensors)[0], 'f16(permute(t("w"), [1, 0]))')

    def test_scalar_multiple_and_negation(self):
        self.assertEqual(ctpl.explain_blob(blob_of(self.v * np.float32(0.5)), self.tensors)[0],
                         'f16(mul(t("v"), 0.5))')
        self.assertEqual(ctpl.explain_blob(blob_of(-self.w), self.tensors)[0], 'f16(neg(t("w")))')
        k = np.float32(0.31830987)  # 1/pi in f32, as an exporter's constant
        got = ctpl.explain_blob(blob_of(self.v * k), self.tensors)[0]
        self.assertTrue(got.startswith('f16(mul(t("v"), '), got)
        lit = float(got.split(", ")[1].rstrip("))"))
        np.testing.assert_array_equal((self.v * np.float32(lit)).astype(np.float16),
                                      (self.v * k).astype(np.float16))

    def test_constants(self):
        self.assertTrue(ctpl.explain_blob(blob_of(np.zeros(9)), self.tensors)[0].startswith("f16(zeros([9]))"))
        self.assertTrue(ctpl.explain_blob(blob_of(np.ones(9)), self.tensors)[0].startswith("f16(ones([9]))"))

    def test_unexplained(self):
        self.assertEqual(ctpl.explain_blob(blob_of(np.sin(self.v)), self.tensors), [])

    def test_literal_round_trips(self):
        for k in (0.5, 1 / 3, 9.8672466, -2.5e-07, 60000.0):
            self.assertEqual(np.float32(float(ctpl._fmt_num(k))), np.float32(k))


class DraftTest(unittest.TestCase):
    def test_draft(self):
        rng = np.random.default_rng(3)
        w = rng.standard_normal((4, 6)).astype(np.float32)
        tensors = {"enc.w": ("f32", w), "enc.w_copy": ("f32", w.copy())}
        data = weight_bin([w.reshape(-1).astype(np.float16), np.sin(w).reshape(-1).astype(np.float16)])
        blobs = ctpl.read_weight_bin(data)
        manifest = {"tensors": [{"name": "enc.w", "type": "q8_0"}]}
        text, flagged, ambiguous = ctpl.draft_recipe("toy", "ab" * 32, len(data), blobs, tensors, manifest)
        lines = text.splitlines()
        self.assertEqual(lines[:5], ["mlrecipe 1", "template toy", f"spec_sha256 {'ab' * 32}",
                                     f"weight_bin_size {len(data)}", "blobs 2"])
        self.assertIn('source "enc.w" f32,q8_0 [4,6]', lines)
        self.assertIn('blob 64 fp16 24 f16(t("enc.w"))', lines)
        self.assertEqual(ambiguous, [64])  # two equal tensors: twin B decides
        self.assertEqual(flagged, [blobs[1].offset])
        self.assertIn(f"blob {blobs[1].offset} fp16 24 ???", lines)
        self.assertEqual(lines[-1], "end")

    def test_refresh_header(self):
        text = ("mlrecipe 1\ntemplate old\nspec_sha256 00\nweight_bin_size 1\nblobs 1\n"
                'source "a" f32 [2]\n# blobs follow\nblob 64 fp16 2 f16(t("a"))\nend\n')
        out = ctpl.refresh_recipe_header(text, "new", "ff" * 32, 192, 1)
        self.assertIn("template new\n", out)
        self.assertIn(f"spec_sha256 {'ff' * 32}\n", out)
        self.assertIn("weight_bin_size 192\n", out)
        self.assertTrue(out.endswith('source "a" f32 [2]\n# blobs follow\nblob 64 fp16 2 f16(t("a"))\nend\n'))
        with self.assertRaisesRegex(ctpl.TemplateError, "lacks blobs"):
            ctpl.refresh_recipe_header("mlrecipe 1\ntemplate a\nspec_sha256 0\nweight_bin_size 1\nend\n", "a", "0", 1, 1)


class TemplateJsonTest(unittest.TestCase):
    def test_expand_args(self):
        self.assertEqual(ctpl.expand_args(["--gguf", "{gguf}", "--output", "{out}/x", "--w", "128"], "/m.gguf", "/o"),
                         ["--gguf", "/m.gguf", "--output", "/o/x", "--w", "128"])
        with self.assertRaises(ctpl.TemplateError):
            ctpl.expand_args([128], "a", "b")

    def test_merge_keeps_author_fields(self):
        existing = {"id": "old", "family": "parler", "stage": "dac", "match": {"architecture": "parler",
                    "metadata": {"parler.dac.hop": 512}}, "default_compute_units": "cpu_and_ane",
                    "weight_bin_size": 1}
        computed = {"schema": 1, "id": "parler-dac-w128", "weight_bin_size": 4096,
                    "functions": [{"name": "main", "inputs": [], "outputs": []}], "stateful": False}
        j = ctpl.merge_template_json(existing, computed)
        self.assertEqual(j["id"], "parler-dac-w128")
        self.assertEqual(j["weight_bin_size"], 4096)
        self.assertEqual(j["family"], "parler")
        self.assertEqual(j["match"]["metadata"], {"parler.dac.hop": 512})
        self.assertEqual(j["default_compute_units"], "cpu_and_ane")
        self.assertEqual(j["expected_warm_load_ms"], {"main": 0})
        self.assertEqual(j["patches"], [])
        self.assertTrue(j["shareable"])

    def test_merge_defaults_match_from_manifest(self):
        manifest = {"metadata": [{"key": "general.architecture", "value": "qvac-toy"}]}
        j = ctpl.merge_template_json({}, {"stateful": True, "functions": []}, manifest)
        self.assertEqual(j["match"], {"architecture": "qvac-toy", "metadata": {}})
        self.assertFalse(j["shareable"])  # a stateful model gets one instance per owner


class ManifestTest(unittest.TestCase):
    def test_equivalent_modulo_identifiers(self):
        def manifest(ids, path="com.apple.CoreML/model.mlmodel"):
            return json.dumps({"fileFormatVersion": "1.0.0", "rootModelIdentifier": ids[1], "itemInfoEntries": {
                ids[0]: {"author": "com.apple.CoreML", "name": "weights", "path": "com.apple.CoreML/weights"},
                ids[1]: {"author": "com.apple.CoreML", "name": "model.mlmodel", "path": path}}})
        a = manifest(["U1", "U2"])
        self.assertTrue(ctpl.manifest_equivalent(a, manifest(["B1", "B2"])))
        self.assertFalse(ctpl.manifest_equivalent(a, manifest(["B1", "B2"], "com.apple.CoreML/other.mlmodel")))
        wrong_root = json.loads(manifest(["B1", "B2"]))
        wrong_root["rootModelIdentifier"] = "B1"  # the weights item
        self.assertFalse(ctpl.manifest_equivalent(a, json.dumps(wrong_root)))
        self.assertFalse(ctpl.manifest_equivalent(a, "not json"))


class SpecDiffTest(unittest.TestCase):
    @needs(HAVE_CT, "coremltools")
    def test_spec_diff(self):
        from coremltools.proto import Model_pb2

        a = Model_pb2.Model()
        a.specificationVersion = 9
        a.description.metadata.userDefined["k"] = "1"
        a.description.metadata.userDefined["same"] = "x"
        b = Model_pb2.Model()
        b.CopyFrom(a)
        self.assertEqual(ctpl.spec_diff(a, b), [])
        b.description.metadata.userDefined["k"] = "2"
        b.description.metadata.userDefined["extra"] = "y"
        b.specificationVersion = 8
        d = ctpl.spec_diff(a, b)
        self.assertIn("specificationVersion", d)
        self.assertIn("description.metadata.userDefined['k']", d)
        self.assertIn("description.metadata.userDefined['extra'] (only in B)", d)
        # coremltools' conversion date is not a difference.
        b.CopyFrom(a)
        a.description.metadata.userDefined["com.github.apple.coremltools.conversion_date"] = "2026-11-13"
        b.description.metadata.userDefined["com.github.apple.coremltools.conversion_date"] = "2026-11-14"
        self.assertEqual(ctpl.spec_diff(a, b), [])
        del a.description.metadata.userDefined["com.github.apple.coremltools.conversion_date"]
        self.assertEqual(ctpl.spec_diff(a, b), [])
        del b.description.metadata.userDefined["com.github.apple.coremltools.conversion_date"]
        # Deterministic serialisation is independent of map insertion order.
        c = Model_pb2.Model()
        c.specificationVersion = 9
        c.description.metadata.userDefined["same"] = "x"
        c.description.metadata.userDefined["k"] = "1"
        self.assertEqual(a.SerializeToString(deterministic=True), c.SerializeToString(deterministic=True))


def mil_spec(consts):
    """A minimal ML program spec: one const op per (name, MIL dtype, payload kind, values)."""
    from coremltools.proto import Model_pb2

    m = Model_pb2.Model()
    m.specificationVersion = 9
    block = m.mlProgram.functions["main"].block_specializations["CoreML8"]
    for name, dtype, kind, values in consts:
        op = block.operations.add()
        op.type = "const"
        op.outputs.add().name = name
        a = op.attributes["val"]
        a.type.tensorType.dataType = dtype
        if kind == "bytes":
            a.immediateValue.tensor.bytes.values = np.asarray(values, dtype="<f2").tobytes()
        else:
            a.immediateValue.tensor.floats.values.extend(float(v) for v in values)
        name_attr = op.attributes["name"]
        name_attr.immediateValue.tensor.strings.values.append(name)
    return m


class PatchSiteTest(unittest.TestCase):
    @needs(HAVE_CT, "coremltools")
    def test_sites_for_weight_dependent_constants(self):
        fixed = ("eps", ctpl.MIL_TYPE_FP16, "bytes", [1e-5])
        a = mil_spec([fixed, ("bias", ctpl.MIL_TYPE_FP16, "bytes", [0.5, -1.0, 2.0]),
                      ("scale", ctpl.MIL_TYPE_FP32, "floats", [3.0, 4.0])])
        b = mil_spec([fixed, ("bias", ctpl.MIL_TYPE_FP16, "bytes", [0.25, 1.0, 2.0]),
                      ("scale", ctpl.MIL_TYPE_FP32, "floats", [5.0, 4.0])])
        sites = ctpl.differing_constants(a, b)
        self.assertEqual([(p.split("/")[-1], d) for p, _, d in sites], [("bias.val", "fp16"), ("scale.val", "fp32")])
        spec, out = ctpl.patch_spec(a, b)
        data = spec.SerializeToString(deterministic=True)
        self.assertEqual(len(data), len(a.SerializeToString(deterministic=True)))
        found = ctpl.find_sites(data, [(p, sent) for p, sent, _, _ in out])
        self.assertEqual([f["len"] for f in found], [6, 8])
        for f, (_, sent, dtype, payload) in zip(found, out):
            self.assertEqual(bytes.fromhex(f["sentinel_hex"]), sent)
            self.assertEqual(data[f["spec_offset"]:f["spec_offset"] + f["len"]], sent)
        # A's own values come back with the payload, for the patch-line drafts.
        self.assertEqual(out[0][3], np.asarray([0.5, -1.0, 2.0], dtype="<f2").tobytes())
        # Writing A's values at the sites gives A's deterministic bytes back.
        restored = bytearray(data)
        for f, (_, _, _, payload) in zip(found, out):
            restored[f["spec_offset"]:f["spec_offset"] + f["len"]] = payload
        self.assertEqual(bytes(restored), a.SerializeToString(deterministic=True))
        # Sentinels are deterministic, and finite for fp32.
        self.assertEqual(ctpl.sentinel_bytes("p", 8, "fp32"), ctpl.sentinel_bytes("p", 8, "fp32"))
        for i in range(64):
            v = np.frombuffer(ctpl.sentinel_bytes(f"p{i}", 64, "fp32"), dtype="<f4")
            self.assertTrue(np.all(np.isfinite(v)))

    @needs(HAVE_CT, "coremltools")
    def test_unpatchable_differences(self):
        a = mil_spec([("idx", 21, "bytes", [1.0])])  # int8: not a patchable dtype
        b = mil_spec([("idx", 21, "bytes", [2.0])])
        with self.assertRaisesRegex(ctpl.TemplateError, "cannot be patched"):
            ctpl.differing_constants(a, b)
        a = mil_spec([("x", ctpl.MIL_TYPE_FP16, "bytes", [1.0])])
        b = mil_spec([("y", ctpl.MIL_TYPE_FP16, "bytes", [1.0])])
        with self.assertRaisesRegex(ctpl.TemplateError, "same inline constants"):
            ctpl.differing_constants(a, b)
        b = mil_spec([("x", ctpl.MIL_TYPE_FP16, "bytes", [1.0])])
        b.specificationVersion = 8
        with self.assertRaisesRegex(ctpl.TemplateError, "specificationVersion"):
            ctpl.patch_spec(a, b)
        with self.assertRaisesRegex(ctpl.TemplateError, "occurs never"):
            ctpl.find_sites(b"abc", [("p", b"zz")])
        with self.assertRaisesRegex(ctpl.TemplateError, "occurs twice"):
            ctpl.find_sites(b"zz.zz", [("p", b"zz")])


class PatchDraftTest(unittest.TestCase):
    def test_add_patch_drafts(self):
        text = 'mlrecipe 1\nblob 64 fp16 2 f16(t("a"))\npatch 10 fp16 1 f16(t("b"))\nend\n'
        drafts = [(10, 'patch 10 fp16 1 f16(t("b"))', "p1"), (40, 'patch 40 fp16 2 f16(neg(t("c")))', "p2"),
                  (90, "patch 90 fp32 1 ???", "p3")]
        out, added = ctpl.add_patch_drafts(text, drafts)
        self.assertEqual([a[0] for a in added], [40])
        self.assertTrue(out.endswith('patch 40 fp16 2 f16(neg(t("c")))\nend\n'), out)
        self.assertIn("# drafted by extract from twin A (p2)", out)
        self.assertEqual(ctpl.add_patch_drafts(out, drafts)[1], [])


class PatchDraftEndTest(unittest.TestCase):
    def test_recipe_without_end_is_a_template_error(self):
        drafts = [(10, 'patch 10 fp16 1 f16(t("b"))', "p1")]
        with self.assertRaisesRegex(ctpl.TemplateError, "no `end` line"):
            ctpl.add_patch_drafts('mlrecipe 1\nblob 64 fp16 2 f16(t("a"))\n', drafts)


class LiteralTest(unittest.TestCase):
    def test_extract_literals(self):
        rope = np.cos(np.arange(12, dtype=np.float64)).astype(np.float16)
        w_a = np.ones(5, np.float16)
        a = ctpl.read_weight_bin(weight_bin([w_a, rope]))
        b = ctpl.read_weight_bin(weight_bin([w_a * 2, rope]))
        decl = [{"id": "rope", "dtype": "fp16", "shape": [3, 4], "exporter_blob": a[1].offset}]
        data, entries = ctpl.extract_literals(decl, a, b)
        self.assertEqual(entries[0]["offset"], 0)
        self.assertEqual(entries[0]["exporter_blob"], a[1].offset)
        self.assertEqual(data, rope.astype("<f2").tobytes())
        two = decl + [{"id": "again", "dtype": "fp16", "shape": [12], "exporter_blob": a[1].offset}]
        data, entries = ctpl.extract_literals(two, a, b)
        self.assertEqual([e["offset"] for e in entries], [0, 64])
        self.assertIsNone(ctpl.extract_literals([], a, b))
        for bad, msg in [
            ([{"id": "w", "dtype": "fp16", "shape": [5], "exporter_blob": a[0].offset}], "depends on the weights"),
            ([{"id": "r", "dtype": "fp32", "shape": [12], "exporter_blob": a[1].offset}], "dtype"),
            ([{"id": "r", "dtype": "fp16", "shape": [2, 4], "exporter_blob": a[1].offset}], "elements"),
            ([{"id": "r", "dtype": "fp16", "shape": [12], "exporter_blob": 7}], "no weight.bin blob"),
            ([{"id": "r", "dtype": "fp16", "shape": [12]}], "exporter_blob"),
        ]:
            with self.assertRaisesRegex(ctpl.TemplateError, msg):
                ctpl.extract_literals(bad, a, b)


TOY_DIR = HERE.parent / "test" / "toy"
TOY_STATELESS = HERE.parent / "test" / "toy-templates" / "toy-stateless"
TOY_ARGS = ["--kind", "stateless", "--gguf", "{gguf}", "--output", "{out}"]
NEEDS_TOY = needs(HAVE_CT and HAVE_GGUF and TOOL and Path(TOOL).is_file(), "coremltools, gguf and $QVAC_COREML_TOOL")


NEEDS_TOOL = needs(HAVE_GGUF and TOOL and Path(TOOL).is_file(), "gguf and $QVAC_COREML_TOOL")


def write_gguf(path, tensors):
    import gguf

    w = gguf.GGUFWriter(str(path), "toy")
    for name, arr in tensors.items():
        w.add_tensor(name, np.asarray(arr, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def with_constant_stats(manifest, values):
    """A copy of `manifest` whose tensors in `values` record that one value
    for every element (as a scalar bias or a constant vector does)."""
    m = json.loads(Path(manifest).read_text())
    for t in m["tensors"]:
        if t["name"] in values:
            v = values[t["name"]]
            t["stats"] = {"min": v, "max": v, "mean": v, "std": 0.0}
    return m


class TwinTest(unittest.TestCase):
    """Twins differ on every tensor, and the extraction gates refuse twins
    that hold the same values for a tensor."""

    @NEEDS_TOOL
    def test_constant_tensors_differ_between_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            man = tmp / "reference-gguf.json"
            man.write_text(json.dumps(with_constant_stats(TOY_STATELESS / "reference-gguf.json",
                                                          {"toy.b1": 0.25, "toy.w1": 0.0})))
            twins = []
            for seed in (1, 2):
                out = tmp / f"twin-{seed}.gguf"
                subprocess.run([TOOL, "twin", "--manifest", man, "--seed", str(seed), "--types", "f32", "-o", out],
                               check=True, capture_output=True)
                twins.append(out)
            self.assertEqual(ctpl.coinciding_tensors(*twins), [])
            for twin in twins:
                t = ctpl.gguf_tensors(twin)
                b1, w1 = t["toy.b1"][1], t["toy.w1"][1]
                self.assertTrue(np.all((b1 >= 0.1875) & (b1 <= 0.3125)), b1)  # within 25%, sign kept
                self.assertTrue(np.all(np.abs(w1) <= 0.00025), w1)
                self.assertGreater(np.ptp(b1), 0.0)

    @needs(HAVE_GGUF, "gguf")
    def test_equal_tensors_are_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.gguf", Path(tmp) / "b.gguf"
            write_gguf(a, {"same": [0.5], "other": [1.0, 2.0]})
            write_gguf(b, {"same": [0.5], "other": [1.0, 2.5]})
            self.assertEqual(ctpl.coinciding_tensors(a, b), ["same"])
            with self.assertRaisesRegex(ctpl.TemplateError, r"same values for 1 tensor\(s\).*: same"):
                ctpl.check_twins_differ(a, b, "extract")


# Templates whose patch lines read GGUF tensors by name directly, with those
# names. Other templates' patches may read metadata or resolve logical names,
# so they are not listed here.
PATCH_INPUT_TENSORS = {"parler-dac-w128": ["dac.dec.conv_out.bias"]}


class RepoTemplatePatchTest(unittest.TestCase):
    """For each template in PATCH_INPUT_TENSORS: two GGUFs that differ only in
    a tensor its patch lines read get different artifact keys, and generated
    programs that differ only at the patch sites. A program constant computed
    from a tensor the recipe does not read would make both the same."""

    def run_tool(self, *args):
        p = subprocess.run([TOOL] + [str(a) for a in args], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout

    @NEEDS_TOOL
    def test_patch_tensors_reach_key_and_program(self):
        import gguf

        by_id = {d.name: d for d in ctpl.find_templates(ctpl.repo_root())}
        for tid, names in PATCH_INPUT_TENSORS.items():
            self.assertIn(tid, by_id, f"template {tid} is not in the repository")
            tdir = by_id[tid]
            patches = json.loads((tdir / "template.json").read_text()).get("patches", [])
            self.assertTrue(patches, f"{tid}: no patch sites")
            lines = [l for l in (tdir / "recipe.txt").read_text().splitlines() if l.startswith("patch ")]
            for name in names:
                self.assertIn(f't("{name}")', "\n".join(lines), f"{tid}: no patch line reads {name}")
            sites = [(p["spec_offset"], p["spec_offset"] + p["len"]) for p in patches]
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                a = tmp / "a.gguf"
                self.run_tool("twin", "--manifest", tdir / "reference-gguf.json", "--seed", "1", "--types", "f32",
                              "-o", a)
                key_a = json.loads(self.run_tool("digest", "--template", tdir, "--gguf", a))["artifact_key"]
                self.run_tool("generate", "--template", tdir, "--gguf", a, "-o", tmp / "a.mlpackage")
                spec_a = ctpl.package_parts(tmp / "a.mlpackage")["spec"].read_bytes()
                for name in names:
                    b = tmp / "b.gguf"
                    shutil.copyfile(a, b)
                    r = gguf.GGUFReader(str(b), "r+")
                    t = next(x for x in r.tensors if x.name == name)
                    t.data[...] = np.asarray(t.data) * -1.5 - 0.125
                    del t, r
                    others = sorted(set(ctpl.gguf_tensors(a)) - {name})
                    self.assertEqual(ctpl.coinciding_tensors(a, b), others)  # only `name` differs
                    key_b = json.loads(self.run_tool("digest", "--template", tdir, "--gguf", b))["artifact_key"]
                    self.assertNotEqual(key_a, key_b, f"{tdir.name}: {name} is not in the artifact key")
                    gen_b = tmp / "b.mlpackage"
                    shutil.rmtree(gen_b, ignore_errors=True)
                    self.run_tool("generate", "--template", tdir, "--gguf", b, "-o", gen_b)
                    spec_b = ctpl.package_parts(gen_b)["spec"].read_bytes()
                    self.assertEqual(len(spec_a), len(spec_b))
                    diff = [i for i, (x, y) in enumerate(zip(spec_a, spec_b)) if x != y]
                    self.assertTrue(diff, f"{tdir.name}: {name} does not reach the program")
                    for i in diff:
                        self.assertTrue(any(lo <= i < hi for lo, hi in sites), f"{tdir.name}: byte {i} is no patch site")


def run_ctpl(*args):
    return subprocess.run([sys.executable, str(HERE / "coreml_template.py"), "--tool", TOOL] + [str(a) for a in args],
                          capture_output=True, text=True)


def exporter_args(args):
    return [f"--exporter-arg={a}" for a in args]


def recipe_body(text):
    """A recipe without its spec_sha256, which hashes the exporter's spec and
    so its conversion date."""
    return [line for line in text.splitlines() if not line.startswith("spec_sha256 ")]


def snapshot(d):
    return {f.name: f.read_bytes() for f in sorted(Path(d).iterdir()) if f.is_file()}


class ToySuggestTest(unittest.TestCase):
    """`suggest` on the toy exporter explains every blob, and the draft equals
    the committed toy recipe (whose blobs were all suggested)."""

    def suggest(self, tmp, exporter, *root):
        out = Path(tmp) / "recipe.txt"
        p = run_ctpl(*root, "suggest", "--manifest", TOY_STATELESS / "reference-gguf.json", "--exporter", exporter,
                     *exporter_args(TOY_ARGS), "--id", "toy-stateless", "-o", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("0 unexplained, 0 ambiguous", p.stderr)
        self.assertEqual(recipe_body(out.read_text()), recipe_body((TOY_STATELESS / "recipe.txt").read_text()))

    @NEEDS_TOY
    def test_suggest_stateless_toy(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.suggest(tmp, TOY_DIR / "export_toy.py")

    @NEEDS_TOY
    def test_suggest_resolves_exporter_under_root(self):
        # A relative exporter path is looked up under --root, not under the
        # checkout this script lives in (which has no such file).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "other-checkout"
            rel = Path("exporters") / "export_toy_elsewhere.py"
            (root / rel).parent.mkdir(parents=True)
            shutil.copyfile(TOY_DIR / "export_toy.py", root / rel)
            self.assertFalse((ctpl.repo_root() / rel).exists())
            self.suggest(tmp, rel, "--root", root)


# An exporter around the toy one, for the cases `extract` must refuse:
#   spec    a spec value computed from the weights (a user metadata entry),
#           so twin A's spec differs from twin B's;
#   layout  every export after the first carries one more fp16 blob in
#           weight.bin (counted in a state file), so the layouts differ.
VARIANT_EXPORTER = """
import struct
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import export_toy  # noqa: E402

mode, gguf, out, state = sys.argv[2:6]
pkg = export_toy.export("stateless", gguf, out)
data = pkg / "Data" / "com.apple.CoreML"
if mode == "spec":
    from coremltools.proto import Model_pb2

    spec = Model_pb2.Model()
    spec.ParseFromString((data / "model.mlmodel").read_bytes())
    w1 = export_toy.read_weights(gguf, "stateless")["toy.w1"]
    spec.description.metadata.userDefined["qvac.toy.w1_sum"] = repr(float(w1.sum()))
    (data / "model.mlmodel").write_bytes(spec.SerializeToString(deterministic=True))
elif mode == "patch":
    # An inline fp16 constant that depends on the weights (minus toy.b1): a
    # patch site that `extract` can explain and draft a patch line for.
    import numpy as np
    from coremltools.proto import Model_pb2

    spec = Model_pb2.Model()
    spec.ParseFromString((data / "model.mlmodel").read_bytes())
    b1 = export_toy.read_weights(gguf, "stateless")["toy.b1"]
    block = next(iter(spec.mlProgram.functions["main"].block_specializations.values()))
    op = block.operations.add()
    op.type = "const"
    op.outputs.add().name = "qvac_toy_neg_b1"
    val = op.attributes["val"]
    val.type.tensorType.dataType = 10  # fp16
    val.immediateValue.tensor.bytes.values = (-np.asarray(b1, dtype=np.float32)).astype("<f2").tobytes()
    op.attributes["name"].immediateValue.tensor.strings.values.append("qvac_toy_neg_b1")
    (data / "model.mlmodel").write_bytes(spec.SerializeToString(deterministic=True))
elif mode == "layout":
    n = int(Path(state).read_text()) if Path(state).is_file() else 0
    Path(state).write_text(str(n + 1))
    if n:
        wb = bytearray((data / "weights" / "weight.bin").read_bytes())
        wb += bytes((-len(wb)) % 64)
        blob = bytes(16)  # eight fp16 zeros
        meta = len(wb)
        wb += struct.pack("<IIQQ", 0xDEADBEEF, 1, len(blob), meta + 64) + bytes(40) + blob
        wb += bytes((-len(wb)) % 64)
        struct.pack_into("<I", wb, 0, struct.unpack_from("<I", wb, 0)[0] + 1)
        (data / "weights" / "weight.bin").write_bytes(bytes(wb))
print(pkg)
"""


class ToyExtractTest(unittest.TestCase):
    """`extract` on the toy exporter: it refuses an exporter whose spec or
    weight.bin layout depends on the weights, and an unchanged second
    extraction leaves every template file, and so the template's identity,
    as it was."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.tdir = self.tmp / "toy-stateless"
        shutil.copytree(TOY_STATELESS, self.tdir)

    def tearDown(self):
        self._tmp.cleanup()

    def template_sha256(self):
        twin = self.tmp / "identity-twin.gguf"
        subprocess.run([TOOL, "twin", "--manifest", self.tdir / "reference-gguf.json", "--seed", "1", "--types", "f32",
                        "-o", twin], check=True, capture_output=True)
        p = subprocess.run([TOOL, "generate", "--template", self.tdir, "--gguf", twin, "-o",
                            self.tmp / "identity.mlpackage"], check=True, capture_output=True, text=True)
        shutil.rmtree(self.tmp / "identity.mlpackage")
        return json.loads(p.stdout)["template_sha256"]

    def extract_variant(self, mode):
        script = self.tmp / "export_variant.py"
        script.write_text(VARIANT_EXPORTER)
        args = [TOY_DIR, mode, "{gguf}", "{out}", self.tmp / f"{mode}.state"]
        return run_ctpl("extract", "--template-dir", self.tdir, "--exporter", script, *exporter_args(args))

    @NEEDS_TOY
    def test_rejects_spec_that_differs_between_twins(self):
        before = snapshot(self.tdir)
        p = self.extract_variant("spec")
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("spec differs between twin A and twin B", p.stderr)
        self.assertIn("userDefined['qvac.toy.w1_sum']", p.stderr)
        self.assertEqual(snapshot(self.tdir), before)  # nothing written

    @NEEDS_TOY
    def test_rejects_weight_layout_that_differs_between_twins(self):
        before = snapshot(self.tdir)
        p = self.extract_variant("layout")
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("weight.bin layout differs between twin A and twin B", p.stderr)
        self.assertEqual((self.tmp / "layout.state").read_text(), "2")  # both twins were exported
        self.assertEqual(snapshot(self.tdir), before)

    @NEEDS_TOY
    def test_rejected_literal_writes_nothing(self):
        # A literal declared on a weight blob depends on the weights, so the
        # extraction is rejected after the specs compared equal. A stale
        # model.mlmodel (a spec the exporter no longer writes, which a passing
        # extraction would replace) must survive the rejection.
        from coremltools.proto import Model_pb2

        spec = Model_pb2.Model()
        spec.ParseFromString((self.tdir / "model.mlmodel").read_bytes())
        spec.description.metadata.userDefined["qvac.toy.stale"] = "1"
        (self.tdir / "model.mlmodel").write_bytes(spec.SerializeToString(deterministic=True))
        tj = json.loads((self.tdir / "template.json").read_text())
        tj["literals"] = [{"id": "w1", "dtype": "fp16", "shape": [32, 8], "exporter_blob": 64}]
        (self.tdir / "template.json").write_text(json.dumps(tj, indent=2) + "\n")
        before = snapshot(self.tdir)
        p = run_ctpl("extract", "--template-dir", self.tdir)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("depends on the weights", p.stderr)
        self.assertEqual(snapshot(self.tdir), before)

    def make_spec_stale(self):
        # A model.mlmodel the exporter no longer writes, so a passing
        # extraction would replace it: a rejection must leave it in place.
        from coremltools.proto import Model_pb2

        spec = Model_pb2.Model()
        spec.ParseFromString((self.tdir / "model.mlmodel").read_bytes())
        spec.description.metadata.userDefined["qvac.toy.stale"] = "1"
        (self.tdir / "model.mlmodel").write_bytes(spec.SerializeToString(deterministic=True))

    def edit_recipe(self, edit):
        recipe = self.tdir / "recipe.txt"
        recipe.write_text(edit(recipe.read_text()))

    @NEEDS_TOY
    def test_recipe_header_without_blobs_writes_nothing(self):
        self.make_spec_stale()
        self.edit_recipe(lambda t: "".join(l for l in t.splitlines(True) if not l.startswith("blobs ")))
        before = snapshot(self.tdir)
        p = run_ctpl("extract", "--template-dir", self.tdir)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("header lacks blobs", p.stderr)
        self.assertEqual(snapshot(self.tdir), before)

    @NEEDS_TOY
    def test_recipe_that_fails_to_generate_writes_nothing(self):
        # The recipe body names a tensor the GGUF lacks, so the twin-A
        # generate fails after every twin check passed.
        self.make_spec_stale()
        self.edit_recipe(lambda t: t.replace("toy.w1", "toy.nope", 1))
        before = snapshot(self.tdir)
        p = run_ctpl("extract", "--template-dir", self.tdir)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("generate (twin A) failed", p.stderr)
        self.assertEqual(snapshot(self.tdir), before)

    @NEEDS_TOY
    def test_manifest_elsewhere_with_a_rejected_extraction_writes_nothing(self):
        # --manifest outside the template directory would be copied in as
        # reference-gguf.json, but only by an extraction that passes.
        other = json.loads((self.tdir / "reference-gguf.json").read_text())
        other["review_probe"] = True
        (self.tmp / "other.json").write_text(json.dumps(other))
        tj = json.loads((self.tdir / "template.json").read_text())
        tj["literals"] = [{"id": "w1", "dtype": "fp16", "shape": [32, 8], "exporter_blob": 64}]
        (self.tdir / "template.json").write_text(json.dumps(tj, indent=2) + "\n")
        before = snapshot(self.tdir)
        p = run_ctpl("extract", "--template-dir", self.tdir, "--manifest", self.tmp / "other.json")
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("depends on the weights", p.stderr)
        self.assertEqual(snapshot(self.tdir), before)

    @NEEDS_TOY
    def test_recipe_without_end_writes_nothing(self):
        # A weight-dependent inline constant gives a patch line to draft into
        # the recipe; with no `end` line there is nowhere to put it.
        self.edit_recipe(lambda t: "".join(l for l in t.splitlines(True) if l.strip() != "end"))
        before = snapshot(self.tdir)
        p = self.extract_variant("patch")
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("patch sites", p.stderr)
        self.assertIn("no `end` line", p.stderr)
        self.assertEqual(snapshot(self.tdir), before)

    @NEEDS_TOY
    def test_constant_tensor_folded_into_the_program_becomes_a_patch_site(self):
        # toy.b1 recorded as one repeated value, as a scalar bias is. The
        # `patch` exporter folds -b1 into the program, so it must still become
        # a patch site, which it can only when the twins differ on it.
        man = with_constant_stats(self.tdir / "reference-gguf.json", {"toy.b1": 0.25})
        (self.tdir / "reference-gguf.json").write_text(json.dumps(man, indent=1) + "\n")
        p = self.extract_variant("patch")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("1 weight-dependent inline constant(s) become patch sites", p.stderr)
        self.assertEqual(len(json.loads((self.tdir / "template.json").read_text())["patches"]), 1)
        patch_lines = [l for l in (self.tdir / "recipe.txt").read_text().splitlines() if l.startswith("patch ")]
        self.assertEqual(len(patch_lines), 1)
        self.assertIn('t("toy.b1")', patch_lines[0])

    @NEEDS_TOY
    def test_unchanged_reextraction_keeps_template_bytes(self):
        from coremltools.proto import Model_pb2

        # A spec converted on another day: only the conversion date differs
        # from what the exporter writes today, so it must be kept as it is.
        spec = Model_pb2.Model()
        spec.ParseFromString((self.tdir / "model.mlmodel").read_bytes())
        self.assertIn("com.github.apple.coremltools.conversion_date", spec.description.metadata.userDefined)
        spec.description.metadata.userDefined["com.github.apple.coremltools.conversion_date"] = "2000-01-01"
        old_spec = spec.SerializeToString(deterministic=True)
        (self.tdir / "model.mlmodel").write_bytes(old_spec)
        old_manifest = (self.tdir / "Manifest.json").read_bytes()

        p = run_ctpl("extract", "--template-dir", self.tdir)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual((self.tdir / "model.mlmodel").read_bytes(), old_spec)
        self.assertEqual((self.tdir / "Manifest.json").read_bytes(), old_manifest)  # item identifiers kept
        first = snapshot(self.tdir)
        self.assertEqual(recipe_body(first["recipe.txt"].decode()),
                         recipe_body((TOY_STATELESS / "recipe.txt").read_text()))
        identity = self.template_sha256()

        p = run_ctpl("extract", "--template-dir", self.tdir)
        self.assertEqual(p.returncode, 0, p.stderr)
        second = snapshot(self.tdir)
        self.assertEqual(sorted(second), sorted(first))
        for name in first:
            self.assertEqual(second[name], first[name], f"{name} changed on an unchanged extraction")
        self.assertEqual(self.template_sha256(), identity)


if __name__ == "__main__":
    unittest.main(verbosity=2)
