#!/usr/bin/env python3
"""Exercise precision-report CLI output without requiring model fixtures."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
    import coremltools
except ImportError:
    print('SKIP: precision CLI test requires torch and coremltools')
    raise SystemExit(0)

PATH = Path(__file__).parents[1] / 'examples/ios-coreml-bench/compiled_precision.py'
SPEC = importlib.util.spec_from_file_location('compiled_precision_cli', PATH)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class PrecisionCliTests(unittest.TestCase):
    def test_report_output_survives_prediction_collection(self):
        records = [{'route': 'en', 'relative_l2': 0.001}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'metrics.json'

            def compare(configuration):
                self.assertEqual(configuration.assets, root)
                self.assertEqual(configuration.binary, root / 'parakeet')
                self.assertEqual(configuration.output, output)
                return records

            with patch.object(CLI, 'compare_models', side_effect=compare) as collector:
                CLI.main(['--assets', str(root), '--binary', str(root / 'parakeet'),
                          '--output', str(output)])
            collector.assert_called_once()
            self.assertEqual(json.loads(output.read_text()), records)


if __name__ == '__main__':
    unittest.main()
