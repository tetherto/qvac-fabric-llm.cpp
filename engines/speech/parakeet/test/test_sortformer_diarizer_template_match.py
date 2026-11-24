"""The Sortformer diarizer template matches only GGUFs the exporter accepts.

The template's network uses symmetric convolution padding, regular full-context
attention and non-causal 8x subsampling. A GGUF whose encoder metadata asks for
something else (causal convolution, dynamic chunked convolution, chunked
attention, causal downsampling) must get no_template, so the capacity stays on
ggml, which honours that metadata. Twins come from the template's
reference-gguf.json (the shipped GGUF's metadata) through `qvac-coreml-tool
twin`; `digest` refuses a GGUF the template does not match with
"no_template: <reason>". With --gguf the shipped GGUF itself is the positive
control. When torch is importable the exporter's own contract is checked on the
same metadata: every value it refuses, the template refuses too.

Usage: test_sortformer_diarizer_template_match.py --tool <qvac-coreml-tool>
           --template-dir <sortformer-diar-f410> [--gguf <shipped.gguf>]
"""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ARGS = None
VALIDATE = None
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# (metadata key, value written into the twin, whether the exporter refuses it).
# The template also refuses two values the exporter does not check: dynamic
# chunked convolution, which the ggml path honours, and a subsampling factor
# other than the 8 the 80 ms capacity assumes.
REFUSED = [
    ("parakeet.encoder.conv_context_size", "causal", True),
    ("parakeet.encoder.att_context_style", "chunked_limited", True),
    ("parakeet.encoder.causal_downsampling", "true", True),
    ("parakeet.encoder.conv_context_style", "dcc", False),
    ("parakeet.encoder.subsampling_factor", "4", False),
]


def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def exporter_contract():
    """The exporter's validate_export_contract, or None without torch."""
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("numpy") is None:
        return None
    spec = importlib.util.spec_from_file_location("parakeet_export_encoder", SCRIPTS / "export-encoder-coreml.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_export_contract


class SortformerTemplateMatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.manifest = Path(ARGS.template_dir) / "reference-gguf.json"
        cls.meta = {e["key"]: e["value"] for e in json.loads(cls.manifest.read_text())["metadata"]}

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def twin(self, name, sets=()):
        out = Path(self.work.name) / f"{name}.gguf"
        cmd = [ARGS.tool, "twin", "--manifest", str(self.manifest), "--seed", "1", "--types", "q8_0", "-o", str(out)]
        for s in sets:
            cmd += ["--set", s]
        r = run(cmd)
        self.assertEqual(r.returncode, 0, r.stderr)
        return out

    def digest(self, gguf):
        return run([ARGS.tool, "digest", "--template", ARGS.template_dir, "--gguf", str(gguf)])

    def test_reference_metadata_matches(self):
        r = self.digest(self.twin("reference"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)["template"], "sortformer-diar-f410")
        if VALIDATE is not None:
            VALIDATE(dict(self.meta), bypass_pre_encode=True)

    def test_refused_encoder_metadata_has_no_template(self):
        for key, value, exporter_refuses in REFUSED:
            with self.subTest(key=key):
                self.assertIn(key, self.meta, "the shipped GGUF records this key")
                r = self.digest(self.twin(f"refused-{key.rsplit('.', 1)[1]}", [f"{key}={value}"]))
                self.assertNotEqual(r.returncode, 0, f"{key}={value} matched the template")
                self.assertIn("no_template: metadata " + key, r.stderr)
                if VALIDATE is not None and exporter_refuses:
                    meta = dict(self.meta)
                    meta[key] = value == "true" if key.endswith("causal_downsampling") else value
                    with self.assertRaises(ValueError):
                        VALIDATE(meta, bypass_pre_encode=True)

    def test_shipped_gguf_matches(self):
        if not ARGS.gguf:
            self.skipTest("no shipped GGUF given")
        r = self.digest(ARGS.gguf)
        self.assertEqual(r.returncode, 0, r.stderr)


def main():
    global ARGS, VALIDATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--template-dir", required=True)
    ap.add_argument("--gguf", default="")
    ARGS, rest = ap.parse_known_args()
    VALIDATE = exporter_contract()
    print("exporter contract:", "checked" if VALIDATE else "not checked (no torch)", file=sys.stderr)
    unittest.main(argv=[sys.argv[0]] + rest, verbosity=2)


if __name__ == "__main__":
    main()
