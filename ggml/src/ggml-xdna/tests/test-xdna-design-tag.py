# usage: test-xdna-design-tag.py <gated_fmt>
# Fails when xdna-design-tag.h does not hold the tag kernels/design_tag.py computes from the kernel sources at that gated
# layout, CMake's default: a build without kernel generation compiles the header and looks for artifacts under its tag.
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = os.path.join(HERE, "..", "xdna-design-tag.h")
EXTRA = "npu2"  # what CMakeLists.txt passes design_tag.py
TAG_LINE = re.compile(r'#define XDNA_DESIGN_TAG "([0-9a-f]+)"')

sys.path.insert(0, os.path.join(HERE, "..", "kernels"))
import design_tag  # noqa: E402


def expected_tag(gated_fmt: str) -> str:
    # the build's knobs only: one set in this shell would hash into the tag
    for knob in design_tag.DESIGN_ENV:
        os.environ.pop(knob, None)
    os.environ["GATED_FMT"] = gated_fmt
    return design_tag.tag(EXTRA)


def committed_tag() -> str:
    with open(HEADER) as fh:
        m = TAG_LINE.search(fh.read())
    return m.group(1) if m else ""


def main() -> int:
    want, have = expected_tag(sys.argv[1]), committed_tag()
    ok = want == have
    print(f"test-xdna-design-tag: GATED_FMT={sys.argv[1]} sources {want}, xdna-design-tag.h {have or 'none'}: "
          f"{'passed' if ok else 'FAILED (regenerate it with kernels/design_tag.py)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
