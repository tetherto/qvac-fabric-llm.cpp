#!/usr/bin/env python3
"""Run CTest fixture suites and require evidence of executed coverage (CTest 3.21+)."""

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 1200
REPORT_FILENAME = "fixture-results.xml"


def select_tests(tests, label, required):
    selected = []
    for test in tests:
        properties = {item["name"]: item["value"] for item in test.get("properties", [])}
        if label not in properties.get("LABELS", []):
            continue
        if properties.get("DISABLED", False):
            print(f"Disabled: {test['name']}")
        else:
            selected.append(test["name"])
    missing = set(required) - set(selected)
    if missing:
        raise ValueError(f"Required tests are missing, disabled or outside label {label}: {sorted(missing)}")
    if not selected:
        raise ValueError(f"No enabled tests have label {label}")
    return [name for name in selected if name in required] if required else selected


def validate_report(report, selected, required):
    cases = {case.attrib["name"]: case for case in ET.parse(report).iter("testcase")}
    missing = set(selected) - cases.keys()
    if missing:
        raise ValueError(f"CTest did not report selected tests: {sorted(missing)}")
    passed = set()
    for name in selected:
        case = cases[name]
        if case.find("failure") is not None or case.find("error") is not None:
            raise ValueError(f"Test failed: {name}")
        if case.find("skipped") is None and case.get("status", "run") == "run":
            passed.add(name)
    missing = set(required) - passed
    if missing:
        raise ValueError(f"Required tests did not execute successfully: {sorted(missing)}")
    if not passed:
        raise ValueError("All selected tests were skipped; no fixture coverage executed")
    print(f"Executed successfully: {len(passed)} of {len(selected)} selected tests")


def run_tests(arguments):
    build = arguments.test_dir.resolve()
    configuration = ["-C", arguments.configuration] if arguments.configuration else []
    command = ["ctest", "--test-dir", str(build), *configuration]
    listing = subprocess.run([*command, "--show-only=json-v1"],
                             capture_output=True, text=True, check=True)
    selected = select_tests(json.loads(listing.stdout)["tests"], arguments.label, arguments.require)
    report = build / REPORT_FILENAME
    report.unlink(missing_ok=True)
    pattern = "^(" + "|".join(re.escape(name) for name in selected) + ")$"
    subprocess.run([*command, "-R", pattern, "--output-on-failure",
                    "--timeout", str(arguments.timeout), "--output-junit", str(report)], check=True)
    validate_report(report, selected, arguments.require)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--label", default="fixture")
    parser.add_argument("--require", action="append", default=[], metavar="TEST_NAME")
    parser.add_argument("--configuration")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args()


def main():
    try:
        run_tests(parse_arguments())
    except (ValueError, OSError, ET.ParseError, subprocess.CalledProcessError) as error:
        print(f"Fixture coverage failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
