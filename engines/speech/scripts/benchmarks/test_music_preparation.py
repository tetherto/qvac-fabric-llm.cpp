#!/usr/bin/env python3
"""Model-free cache integrity regressions; no network calls."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('prepare_music', Path(__file__).with_name('prepare-music-alignment.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.content = b'pinned model fixture'
        self.manifest = {'model_id': 'test/model', 'revision': 'a' * 40,
                         'files': {'config.json': hashlib.sha256(self.content).hexdigest()}}

    def test_download_verifies_and_reuses_without_network(self):
        urls = []
        def opener(url, timeout):
            urls.append(url)
            return io.BytesIO(self.content)
        target = module.prepare(self.manifest, self.root, opener=opener)
        self.assertEqual((target / 'config.json').read_bytes(), self.content)
        self.assertIn('/resolve/' + 'a' * 40 + '/', urls[0])
        module.prepare(self.manifest, self.root, offline=True,
                       opener=lambda *a, **kw: self.fail('unexpected network'))

    def test_corrupt_transfer_does_not_publish(self):
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            module.prepare(self.manifest, self.root, opener=lambda *a, **kw: io.BytesIO(b'corrupt'))
        target = self.root / 'reference-clap' / ('a' * 40)
        self.assertEqual(list(target.iterdir()), [])

    def test_offline_corrupt_cache_is_reported(self):
        target = self.root / 'reference-clap' / ('a' * 40)
        target.mkdir(parents=True)
        (target / 'config.json').write_bytes(b'bad')
        with self.assertRaisesRegex(FileNotFoundError, 'checksum mismatch'):
            module.prepare(self.manifest, self.root, offline=True)

    def test_unpinned_and_path_escape_are_rejected(self):
        for revision, name in [('main', 'config.json'), ('a' * 40, '../outside'),
                               ('a' * 40, '/absolute'), ('a' * 40, 'nested\\escape')]:
            with self.subTest(revision=revision, name=name):
                fixture = dict(self.manifest, revision=revision,
                               files={name: hashlib.sha256(self.content).hexdigest()})
                with self.assertRaises(ValueError):
                    module.prepare(fixture, self.root, offline=True)

    def test_symlink_escape_is_rejected(self):
        target = self.root / 'reference-clap' / ('a' * 40)
        target.mkdir(parents=True)
        outside = self.root / 'outside'
        outside.write_bytes(self.content)
        (target / 'config.json').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            module.prepare(self.manifest, self.root, offline=True)


if __name__ == '__main__':
    unittest.main()
