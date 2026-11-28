"""Model-free CLI routing contracts for native and Core ML LavaSR stages."""
import subprocess
import sys
import unittest


BINARY = sys.argv.pop(1)


class Routing(unittest.TestCase):
    def invoke(self, *arguments):
        return subprocess.run([BINARY, "--denoiser", "missing-denoiser.gguf",
                               "--enhancer", "missing-enhancer.gguf", *arguments],
                              text=True, capture_output=True, timeout=15)

    def rejected(self, message, *arguments):
        result = self.invoke(*arguments)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(message, result.stderr)

    def test_explicit_native_selection_rejects_coreml(self):
        for backend in ("cpu", "opencl", "hexagon"):
            for stage in ("denoiser", "enhancer", "pipeline"):
                with self.subTest(backend=backend, stage=stage):
                    flag = "--enhancer-coreml" if stage == "enhancer" else "--denoiser-coreml"
                    self.rejected("Core ML options require --backend auto", "--backend", backend,
                                  "--stage", stage, flag, "model.mlmodelc",
                                  "--coreml-compute-units", "CPU_ONLY")

    def test_options_require_model_and_policy(self):
        self.rejected("requires a model path", "--coreml-compute-units", "CPU_ONLY")
        self.rejected("requires a model path", "--denoiser-coreml", "model.mlmodelc")
        self.rejected("invalid Core ML compute policy", "--denoiser-coreml", "model.mlmodelc",
                      "--coreml-compute-units", "unknown")

    def test_legacy_coreml_mode_reaches_model_loading(self):
        for policy in ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"):
            with self.subTest(policy=policy):
                result = self.invoke("--denoiser-coreml", "dn.mlmodelc", "--enhancer-coreml",
                                     "enh.mlmodelc", "--coreml-compute-units", policy)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("load_denoiser_gguf: failed to open GGUF", result.stderr)
                self.assertNotIn("argument error", result.stderr)

    def test_native_gpu_and_coreml_cannot_share_a_stage(self):
        self.rejected("cannot also request native GPU layers", "--denoiser-coreml", "dn.mlmodelc",
                      "--coreml-compute-units", "CPU_ONLY", "--dn-gpu-layers", "1")
        self.rejected("cannot also request native GPU layers", "--enhancer-coreml", "enh.mlmodelc",
                      "--coreml-compute-units", "CPU_ONLY", "--n-gpu-layers", "1")

    def test_unused_stage_options_are_rejected(self):
        self.rejected("Core ML options name an unused stage", "--stage", "enhancer",
                      "--denoiser-coreml", "dn.mlmodelc", "--coreml-compute-units", "CPU_ONLY")
        self.rejected("Core ML options name an unused stage", "--stage", "denoiser",
                      "--enhancer-coreml", "enh.mlmodelc", "--coreml-compute-units", "CPU_ONLY")

    def test_help_keeps_coreml_flags(self):
        result = subprocess.run([BINARY, "--help"], text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0)
        for flag in ("--denoiser-coreml", "--enhancer-coreml", "--coreml-compute-units"):
            self.assertIn(flag, result.stderr)


if __name__ == "__main__":
    unittest.main()
