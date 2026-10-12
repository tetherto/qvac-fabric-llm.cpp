import json
from subprocess import CompletedProcess
import unittest

from benchmark import validate_run


class BenchmarkValidation(unittest.TestCase):
    def result(self, selected=0, reason=None, code=0, nmse=1e-7):
        metrics = dict(cold_ms=1.0, warm_ms=0.5, nmse=nmse, max_abs=0.01)
        log = "coreml: route=ANE-plan tensor=test\n" * selected
        if reason:
            log += f"coreml: route=Metal tensor=test reason={reason}\n" * 4
        return CompletedProcess([], code, json.dumps(metrics), log)

    def test_all_predictions_required(self):
        validate_run(self.result(selected=4), "coreml", 1, True)
        for count in (0, 1, 3, 5):
            with self.assertRaises(RuntimeError):
                validate_run(self.result(selected=count), "coreml", 1, True)

    def test_fallback_reasons(self):
        for mode, reason in (("injected-failure", "injected-failure"),
                             ("prediction-failure", "injected-prediction-failure"),
                             ("missing-model", "model-load-or-layout")):
            validate_run(self.result(reason=reason), mode, 1, True)
            with self.assertRaises(RuntimeError):
                validate_run(self.result(reason="disabled-or-capture-or-abort"), mode, 1, True)

    def test_nonfinite_and_error_exit(self):
        for proc in (self.result(nmse=float("nan")), self.result(nmse=1.0), self.result(code=1)):
            with self.assertRaises(RuntimeError):
                validate_run(proc, "metal", 1, False)

    def test_unexpected_selection(self):
        with self.assertRaises(RuntimeError):
            validate_run(self.result(selected=1), "metal", 1, False)

    def test_empty_output(self):
        with self.assertRaises(RuntimeError):
            validate_run(CompletedProcess([], 0, "", ""), "metal", 1, False)


if __name__ == "__main__":
    unittest.main()
