#!/usr/bin/env python3
"""Fail unless every test in a CTest JUnit report ran and passed.

A green CTest step does not prove a test ran: disabled tests (missing
fixtures), "Not Run" tests and SKIP_RETURN_CODE skips all leave it green. The
Core ML lanes must fail on any of them.

    check-ctest-junit.py REPORT.xml [--require NAME ...]
"""

import argparse
import sys
import xml.etree.ElementTree as ET


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--require", action="append", default=[],
                        help="test name that must be present and pass")
    args = parser.parse_args()

    cases = list(ET.parse(args.report).iter("testcase"))
    if not cases:
        print(f"{args.report}: no test cases", file=sys.stderr)
        return 1

    bad = []
    seen = set()
    for case in cases:
        name = case.get("name", "?")
        seen.add(name)
        status = case.get("status", "run")
        if status != "run":
            bad.append(f"{name}: status={status}")
        elif case.find("skipped") is not None:
            bad.append(f"{name}: skipped")
        elif case.find("failure") is not None or case.find("error") is not None:
            bad.append(f"{name}: failed")
    for name in args.require:
        if name not in seen:
            bad.append(f"{name}: missing from the report")

    for line in bad:
        print(f"not run or failed: {line}", file=sys.stderr)
    if bad:
        return 1
    print(f"{len(cases)} test(s) ran and passed: {', '.join(sorted(seen))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
