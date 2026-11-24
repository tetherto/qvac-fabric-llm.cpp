#!/usr/bin/env python3
"""Check iPhone bundle packaging without Apple signing tools or model fixtures."""
import importlib.util
import os
from pathlib import Path
import plistlib
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).parents[1] / 'examples/ios-coreml-bench/package.py'
SPEC = importlib.util.spec_from_file_location('ios_coreml_package', PATH)
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)

EXECUTABLE_MODE = 0o755
BUNDLE_ID = 'org.example.qvacbench'
BINARY_CONTENT = b'benchmark executable\n'


@unittest.skipUnless(os.name == 'posix', 'Requires POSIX executable permissions')
class IosCoremlPackageTests(unittest.TestCase):
    def test_fresh_bundle_preserves_executable_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'benchmark'
            binary.write_bytes(BINARY_CONTENT)
            binary.chmod(EXECUTABLE_MODE)
            profile = root / 'profile.mobileprovision'
            profile.write_bytes(b'test provisioning profile')
            samples = root / 'samples'
            samples.mkdir()
            args = SimpleNamespace(binary=binary, profile=profile,
                                   identity='test identity', bundle_id=BUNDLE_ID,
                                   model_dir=[], samples=samples,
                                   output=root / 'Bench.app')
            decoded_profile = plistlib.dumps({'Entitlements': {
                'application-identifier': f'TESTTEAM.{BUNDLE_ID}'}})
            self.assertFalse(args.output.exists())

            with patch.object(PACKAGE.subprocess, 'check_output', return_value=decoded_profile), \
                    patch.object(PACKAGE.subprocess, 'run') as signer, \
                    patch('builtins.print'):
                PACKAGE.package(args)

            signer.assert_called_once()
            bundled_binary = args.output / PACKAGE.EXECUTABLE_NAME
            self.assertEqual(bundled_binary.read_bytes(), BINARY_CONTENT)
            self.assertEqual(stat.S_IMODE(bundled_binary.stat().st_mode), EXECUTABLE_MODE)
            self.assertTrue(os.access(bundled_binary, os.X_OK))


if __name__ == '__main__':
    unittest.main()
