"""
On-device test-backend-ops runner for llama.cpp (HTP0 backend).

On Android: executed by QDC's Appium test framework on the QDC runner.
The runner has ADB access to the allocated device.
On Linux: runs test-backend-ops directly via run_linux.sh (BASH framework).
"""

import os
import re
import sys

import pytest

from utils import (
    BIN_PATH,
    push_bundle_if_needed,
    run_snapdragon,
    write_qdc_log,
)


@pytest.fixture(scope="session", autouse=True)
def install(driver):
    push_bundle_if_needed(f"{BIN_PATH}/test-backend-ops")


@pytest.mark.parametrize("type_a", ["mxfp4", "fp16", "q4_0"])
def test_backend_ops_htp0(type_a):
    if type_a == "q4_0":
        pattern = r'^(?=.*type_a=q4_0)(?!.*type_b=f32,m=576,n=512,k=576).*$'
    else:
        pattern = f"type_a={type_a}"

    result = run_snapdragon(
        ["test-backend-ops", "-b", "HTP0", "-o", "MUL_MAT", "-p", pattern],
    )
    write_qdc_log(f"backend_ops_{type_a}.log", result.stdout or "")
    assert result.returncode == 0, (
        f"test-backend-ops type_a={type_a} failed (exit {result.returncode})"
    )


def test_backend_ops_snake_htp0():
    result = run_snapdragon(["test-backend-ops", "-b", "HTP0", "-o", "SNAKE"])
    output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout or "")
    write_qdc_log("backend_ops_snake.log", result.stdout or "")
    assert result.returncode == 0, f"SNAKE failed (exit {result.returncode})"
    assert re.search(r"Backend HTP0: OK", output), "HTP0 did not complete successfully"
    assert re.search(r"^\s*SNAKE\(.*\): OK$", output, re.MULTILINE), "No SNAKE case passed on HTP0"
    assert not re.search(r"^\s*SNAKE\(.*\): not supported", output, re.MULTILINE), "SNAKE cases were skipped"


if __name__ == "__main__":
    ret = pytest.main(["-s", "--junitxml=results.xml", os.path.realpath(__file__)])
    if os.path.exists("results.xml"):
        with open("results.xml") as f:
            write_qdc_log("results.xml", f.read())
    sys.exit(ret)
