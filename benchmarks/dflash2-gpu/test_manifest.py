"""Runtime identity must track backend code, not just a shared-library launcher."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from manifest import runtime_artifacts, runtime_fingerprint


class RuntimeIdentityTests(unittest.TestCase):
    def test_backend_replacement_changes_identity_with_unchanged_launcher(self):
        for library_name in ("libggml-cuda.so.0.24.0", "libggml-metal.0.24.0.dylib"):
            with self.subTest(library=library_name), TemporaryDirectory() as directory:
                root = Path(directory)
                binary = root / "llama-server"
                binary.write_bytes(b"unchanged launcher")
                library = root / library_name
                library.write_bytes(b"baseline backend")
                before = runtime_artifacts(binary)
                library.write_bytes(b"candidate backend")
                after = runtime_artifacts(binary)
                self.assertEqual(before[binary.name]["sha256"], after[binary.name]["sha256"])
                self.assertNotEqual(runtime_fingerprint(before), runtime_fingerprint(after))

    def test_aliases_of_one_library_do_not_change_runtime_identity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "llama-server"
            binary.write_bytes(b"launcher")
            library = root / "libggml-metal.0.24.0.dylib"
            library.write_bytes(b"backend")
            before = runtime_fingerprint(runtime_artifacts(binary))
            (root / "libggml-metal.dylib").symlink_to(library.name)
            self.assertEqual(before, runtime_fingerprint(runtime_artifacts(binary)))


if __name__ == "__main__":
    unittest.main()
