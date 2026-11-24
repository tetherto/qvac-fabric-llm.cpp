#!/usr/bin/env python3
"""Measure callback and push streaming in independent serial processes."""
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
import benchmark as batch

REPEATS = 3


def arguments(device, wav, backend, language, pushed):
    model = f"models/transcribepsy-250m-eu.f16.gguf"
    args = ["--model", str(batch.ROOT / model) if device == "mac" else model,
            "--wav", str(batch.ROOT / "inputs" / wav) if device == "mac" else wav,
            "--language", language, "--backend", "cpu" if backend == "coreml" else backend,
            "--threads", "4", "--stream", "--emit", "jsonl", "--verbose", "--bench"]
    if pushed:
        args += ["--stream-duplex", "--stream-chunk-ms", "2000", "--stream-feed-bytes", "16384"]
    if backend == "coreml":
        args += ["--require-coreml"]
    return args


def invoke(device, args, backend, log, require_nominal=False):
    settings = {} if backend == "coreml" else {"PARAKEET_COREML_DISABLE": "1"}
    if device == "mac":
        command = ["/usr/bin/time", "-l", str(batch.BINARY)] + args
        environment = {**batch.BASE_ENV, **settings}
    else:
        command = ["xcrun", "devicectl", "device", "process", "launch", "--device", batch.DEVICE,
                   "--terminate-existing", "--console"] + (["--environment-variables", json.dumps(settings)] if settings else []) + ["--", batch.BUNDLE] + args
        environment = batch.BASE_ENV
    if device == "iphone":
        batch.wait_for_phone_cooldown()
    result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=600)
    content = result.stdout + result.stderr
    log.write_text(content)
    if result.returncode or device == "iphone" and "PHONE_METRICS rc=0" not in content:
        raise RuntimeError(f"Streaming command failed; see {log}")
    thermal = re.search(r'thermal=(\d+)', content)
    if require_nominal and thermal and int(thermal.group(1)) != 0:
        historical = log.with_name(log.stem + '.thermal-' + str(time.time_ns()) + log.suffix)
        historical.write_text(content)
        print('COOLDOWN retry', log.name, flush=True)
        return invoke(device, args, backend, log, require_nominal=True)
    return content


def measure(device, wav, backend, language, pushed):
    name = f"{device}-{backend}-{Path(wav).stem}-{language}-{'push' if pushed else 'callback'}"
    elapsed, transcripts, peaks, footprints, placements = [], [], [], [], []
    for repeat in range(REPEATS):
        content = invoke(device, arguments(device, wav, backend, language, pushed), backend,
                         batch.OUT / f"{name}-{repeat}.log", require_nominal=True)
        matches = re.findall(r"total=([0-9.]+)ms", content)
        if not matches:
            raise RuntimeError(f"No streaming timing for {name}")
        elapsed.append(float(matches[-1]))
        text = []
        for line in content.splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if "text" in value:
                text.append(value["text"])
        transcripts.append("".join(text))
        peak = re.search(r"peak_rss_bytes=(\d+)", content) if device == "iphone" else re.search(r"(\d+)\s+maximum resident set size", content)
        peaks.append(int(peak.group(1)) if peak else None)
        footprint = re.search(r"peak_footprint_bytes=(\d+)", content) if device == "iphone" else re.search(r"(\d+)\s+peak memory footprint", content)
        footprints.append(int(footprint.group(1)) if footprint else None)
        placement = re.findall(r"encoder_backend=([^\s]+)", content)
        placements.append(placement[-1] if placement else backend)
    reference = batch.normalized(batch.REFERENCES[wav])
    got = batch.normalized(transcripts[-1])
    return dict(device=device, wav=wav, backend=backend, language=language, mode="push" if pushed else "callback",
                inference_ms=elapsed, median_ms=statistics.median(elapsed), peak_rss_bytes=peaks,
                peak_footprint_bytes=footprints, encoder_placements=placements,
                transcript=transcripts[-1], repeats_identical=len(set(transcripts)) == 1,
                wer=batch.distance(reference.split(), got.split()) / max(1, len(reference.split())),
                cer=batch.distance(reference, got) / max(1, len(reference)))


def main():
    settings = batch.configure(streaming=True)
    result_path = batch.OUT / "stream-results.json"
    rows = json.loads(result_path.read_text()) if settings.resume and result_path.exists() else []
    for device in batch.DEVICES:
        for wav, language in (("jfk.wav", "en"), ("fr-multipiece-16k.wav", "other"),
                              ("en-60s.wav", "auto"), ("fr-long.wav", "auto")):
            for pushed in (False, True):
                for backend in settings.backends:
                    if any(row["device"] == device and row["wav"] == wav and
                           row["language"] == language and row["backend"] == backend and
                           row["mode"] == ("push" if pushed else "callback") for row in rows):
                        continue
                    row = measure(device, wav, backend, language, pushed)
                    rows.append(row)
                    (batch.OUT / "stream-results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
                    print(row["device"], row["wav"], row["backend"], row["mode"], row["median_ms"], row["wer"], flush=True)


if __name__ == "__main__":
    main()
