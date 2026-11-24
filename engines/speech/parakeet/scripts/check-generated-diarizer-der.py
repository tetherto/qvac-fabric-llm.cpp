#!/usr/bin/env python3
"""The DER gate of the generated Nemotron 3 Diarization diarizers.

Runs the CLI on each pinned clip twice, once with the diarizers generated from
a GGUF that has no adjacent bundle (a fresh cache) and once with the
exporter's bundles beside another copy of the GGUF, both with wait = construct
and prewarm = all. Every diarization call must report backend=coreml-ane and
pending=0 on both routes (otherwise the comparison measures ggml), and the
DER of the generated run may differ from the exporter's by at most 0.2
percentage points (collar 250 ms, scripts/benchmarks/compute-der.py).

  check-generated-diarizer-der.py --cli parakeet --generated-gguf G.gguf
      --prebuilt-gguf P.gguf --samples DIR --der-script compute-der.py --work DIR
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

CLIPS = ("abcba", "abcdba")
MAX_DELTA = 0.002  # 0.2 percentage points


def run_cli(cli, gguf, wav, cache, out_jsonl):
    cmd = [cli, "--model", str(gguf), "--wav", str(wav), "--n-gpu-layers", "999", "--coreml-cache-dir", str(cache),
           "--coreml-wait", "construct", "--coreml-prewarm", "all", "--emit", "jsonl", "--verbose"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out_jsonl.write_text(p.stdout)
    calls = [l for l in p.stderr.splitlines() if l.startswith("[diarize]") and "backend=" in l]
    return p.returncode, calls


def der(der_script, jsonl, rttm, out):
    subprocess.run([sys.executable, str(der_script), "--hypothesis-jsonl", str(jsonl), "--reference", str(rttm),
                    "--collar-ms", "250", "--json-out", str(out)], check=True, capture_output=True)
    return float(json.loads(out.read_text())["der"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for a in ("--cli", "--generated-gguf", "--prebuilt-gguf", "--samples", "--der-script", "--work"):
        ap.add_argument(a, required=True)
    a = ap.parse_args(argv)
    work = Path(a.work)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    failures = []
    for clip in CLIPS:
        wav, rttm = Path(a.samples) / f"{clip}.wav", Path(a.samples) / f"{clip}.rttm"
        results = {}
        for route, gguf in (("generated", a.generated_gguf), ("prebuilt", a.prebuilt_gguf)):
            jsonl = work / f"{clip}-{route}.jsonl"
            rc, calls = run_cli(a.cli, gguf, wav, work / f"cache-{route}", jsonl)
            if rc != 0:
                failures.append(f"{clip} {route}: the CLI exited {rc}")
                continue
            off = [c for c in calls if "backend=coreml-ane" not in c or "pending=0" not in c]
            if not calls or off:
                failures.append(f"{clip} {route}: {len(off)} of {len(calls)} calls not on coreml-ane with pending=0")
                continue
            results[route] = der(a.der_script, jsonl, rttm, work / f"{clip}-{route}.json")
        if len(results) == 2:
            delta = results["generated"] - results["prebuilt"]
            print(f"{clip}: generated {results['generated'] * 100:.2f} %, prebuilt {results['prebuilt'] * 100:.2f} %, "
                  f"delta {delta * 100:+.2f} pp")
            if abs(delta) > MAX_DELTA:
                failures.append(f"{clip}: DER delta {delta * 100:+.2f} pp, over {MAX_DELTA * 100:.1f} pp")
    for f in failures:
        print(f"FAIL {f}")
    print("PASS" if not failures else f"FAILED ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
