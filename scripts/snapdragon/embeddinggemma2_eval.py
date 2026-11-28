#!/usr/bin/env python3
"""Validate text embeddings and measure repeated corpus passes on one context."""

import argparse
import datetime
import hashlib
import json
import logging
import math
import os
import platform
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any


PACKING_MIN_COSINE = 0.998
BATCH_ALIGNMENT = 256
SEPARATOR = "<#embedding-eval#>"
TIMING_PREFIX = "EMBEDDING_BENCH "
REPO = Path(__file__).resolve().parents[2]
BENCH_ENV = ("LLAMA_EMBED_BENCH_REPETITIONS", "LLAMA_EMBED_BENCH_WARMUPS")
COMPUTE_OPS = {"MUL_MAT", "MUL_MAT_ID", "MUL_MAT_NX", "MUL_MAT_ID_NX", "FLASH_ATTN_EXT",
               "GELU", "GELU_QUICK", "GEGLU", "GEGLU_ERF", "GEGLU_QUICK", "RMS_NORM", "SOFT_MAX", "ROPE"}


def host_device_info():
    info = platform.uname()
    return {"host": info.system + " " + info.release, "machine": info.machine}


def htp_compute_records(log):
    pattern = re.compile(r"\b(HTP\d+(?::\d+)?)\s+profile-op\s+([A-Z0-9_+]+)(?=[|:\s])"
                         r".*?\b(?:op-)?cycles\s+(\d+)(?=\s|$)")
    records = []
    for line in log.splitlines():
        match = pattern.search(line)
        if match and COMPUTE_OPS.intersection(match[2].split("+")) and int(match[3]) > 0:
            records.append({"device": match[1], "op": match[2], "cycles": int(match[3])})
    return records


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def aligned_batch_size(size):
    return ((size + BATCH_ALIGNMENT - 1) // BATCH_ALIGNMENT) * BATCH_ALIGNMENT


def partial_batch_indices(lengths):
    capacity = aligned_batch_size(max(max(lengths), (sum(lengths) + 1) // 2))
    indices = []
    batches, batch_tokens = 1, 0
    # Keep two corpus passes and one extra text, then extend past the rounded capacity if needed.
    while len(indices) < 2 * len(lengths) + 1 or batches == 1 or batch_tokens == capacity:
        index = len(indices) % len(lengths)
        if batch_tokens + lengths[index] > capacity:
            batches += 1
            batch_tokens = 0
        indices.append(index)
        batch_tokens += lengths[index]
    return capacity, indices


def cosine(left, right):
    dot = sum(x * y for x, y in zip(left, right))
    norm = math.sqrt(sum(x * x for x in left) * sum(y * y for y in right))
    return dot / norm if norm else 0.0


def compare(left, right):
    if len(left) != len(right) or any(len(x) != len(y) for x, y in zip(left, right)):
        raise ValueError("embedding shapes differ")
    differences = [x - y for row_x, row_y in zip(left, right) for x, y in zip(row_x, row_y)]
    similarities = [cosine(x, y) for x, y in zip(left, right)]
    return {
        "min_cosine": min(similarities),
        "mean_cosine": statistics.mean(similarities),
        "max_absolute_error": max(abs(x) for x in differences),
        "rmse": math.sqrt(sum(x * x for x in differences) / len(differences)),
    }


class Runner:
    def __init__(self, args, report):
        self.args = args
        self.report = report
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_") and k not in BENCH_ENV}
        self.adb = ["adb", "-P", str(args.adb_port), "-s", args.adb] if args.adb else None

    def capture(self, command):
        return subprocess.run(command, env=self.env, text=True, encoding="utf-8", errors="replace",
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.args.timeout)

    def version(self):
        if self.adb:
            library_path = shlex.quote(f"{self.args.remote_dir}/lib")
            binary = self.args.remote_binary or f"{self.args.remote_dir}/bin/{self.args.binary.name}"
            command = self.adb + ["shell", f"LD_LIBRARY_PATH={library_path} "
                                  + shlex.join([binary, "--version"])]
        else:
            command = [str(self.args.binary), "--version"]
        result = self.capture(command)
        if result.returncode:
            raise RuntimeError(f"--version failed: {result.stderr.strip()}")
        return (result.stdout + result.stderr).strip()

    def remote(self, command):
        if self.adb is None:
            raise ValueError("ADB target is not configured")
        result = self.capture(self.adb + ["shell", command])
        if result.returncode:
            raise RuntimeError(f"ADB command failed: {result.stderr.strip()}")
        return result.stdout.strip()

    def verify_remote(self, local_path, remote_path):
        actual = self.remote("sha256sum " + shlex.quote(str(remote_path))).split()[0]
        if actual != sha256(local_path):
            raise ValueError(f"local and device SHA-256 differ: {remote_path}")

    def run(self, name, texts, device=None, model=None, batch_size=None, repetitions=1, diagnostic=False):
        args = self.args
        device = device or args.device
        model = model or args.model
        requested_batch_size = batch_size or args.batch_size
        batch_size = aligned_batch_size(requested_batch_size)
        binary = args.binary
        if self.adb:
            binary = args.remote_binary or f"{args.remote_dir}/bin/{args.binary.name}"
            if model == args.model:
                model = args.remote_model or f"{args.remote_dir}/models/{args.model.name}"
            else:
                model = args.remote_reference_model or f"{args.remote_dir}/models/{model.name}"
        command = [str(binary)]
        command += [
            "-m", str(model), "--device", device, "--fit", "off", "--no-warmup",
            "-c", str(batch_size), "-b", str(batch_size), "-ub", str(batch_size),
            "-t", str(args.threads), "-fa", "off" if device == "none" else args.flash_attn,
            "--pooling", "mean", "--embd-normalize", "2", "--embd-output-format", "array",
            "--embd-separator", SEPARATOR, "--log-colors", "off", "-p", SEPARATOR.join(texts),
        ]
        command += ["-ngl", "0" if device == "none" else "99"]
        if device == "none":
            command += ["--no-op-offload", "--no-kv-offload"]
        if diagnostic:
            command += ["-v"]
        backend_env = {"GGML_HEXAGON_PROFILE": "1" if diagnostic else "0",
                       "GGML_HEXAGON_VERBOSE": "1" if diagnostic else "0",
                       "GGML_HEXAGON_OPTRACE": "0", "GGML_SCHED_DEBUG": "2" if diagnostic else "0",
                       "GGML_OPENCL_KERNEL_CACHE_DEBUG": "0", "GGML_OPENCL_FA_TUNE": "0"}
        if not diagnostic:
            backend_env.update(LLAMA_EMBED_BENCH_REPETITIONS=str(repetitions),
                               LLAMA_EMBED_BENCH_WARMUPS=str(args.warmups))
        if device.startswith("HTP"):
            backend_env["GGML_HEXAGON_DEVICES"] = device
        if self.adb:
            library_path = f"{args.remote_dir}/lib"
            backend_env.update(LD_LIBRARY_PATH=library_path, ADSP_LIBRARY_PATH=library_path)
            env_command = " ".join(f"{key}={shlex.quote(value)}" for key, value in backend_env.items())
            command = self.adb + ["shell", "unset " + " ".join(BENCH_ENV) + "; "
                                  + env_command + " " + shlex.join(command)]
        else:
            for name_env in BENCH_ENV:
                self.env.pop(name_env, None)
            self.env.update(backend_env)
        start = time.monotonic()
        result = self.capture(command)
        wall_ms = (time.monotonic() - start) * 1000
        (args.output / f"{name}.stdout").write_text(result.stdout, encoding="utf-8")
        (args.output / f"{name}.stderr").write_text(result.stderr, encoding="utf-8")
        invocation = {"name": name, "device": device, "model": str(model), "batch_size": batch_size,
                      "requested_batch_size": requested_batch_size,
                      "texts": len(texts), "process_wall_ms": wall_ms, "returncode": result.returncode,
                      "diagnostic": diagnostic, "environment_overrides": backend_env}
        self.report["invocations"].append(invocation)
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}; see {name}.stderr")
        vectors = json.loads(result.stdout)
        if (not isinstance(vectors, list) or len(vectors) != len(texts) or not vectors
                or any(not isinstance(row, list) or not row for row in vectors)
                or len({len(row) for row in vectors}) != 1
                or len(vectors[0]) != args.expected_dimensions
                or any(not isinstance(x, (int, float)) or not math.isfinite(x) for row in vectors for x in row)):
            raise ValueError(f"{name}: invalid/non-finite embedding array")
        norm_error = max(abs(math.sqrt(sum(x * x for x in row)) - 1.0) for row in vectors)
        if norm_error > 1e-4:
            raise ValueError(f"{name}: maximum unit-norm error {norm_error} exceeds 1e-4")
        invocation.update(dimensions=len(vectors[0]), max_unit_norm_error=norm_error)
        timings = [line[len(TIMING_PREFIX):] for line in result.stderr.splitlines()
                   if line.startswith(TIMING_PREFIX)]
        if not diagnostic:
            if len(timings) != 1:
                raise ValueError(f"{name}: missing unique EMBEDDING_BENCH record; rebuild llama-embedding")
            timing = json.loads(timings[0])
            samples = timing["pass_ms"]
            if len(samples) != repetitions or any(not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError(f"{name}: invalid timing samples")
            if (timing["outputs_finite"] is not True or not math.isfinite(timing["repeated_min_cosine"])
                    or timing["repeated_min_cosine"] < PACKING_MIN_COSINE):
                raise ValueError(f"{name}: repeated passes produced non-finite or inconsistent embeddings")
            lengths, batches = timing["tokens_per_text"], timing["batch_tokens"]
            if (timing["texts"] != len(texts) or len(lengths) != len(texts) or not batches
                    or timing["batch_size"] != batch_size or timing["ubatch_size"] != batch_size
                    or len(batches) != timing["batches"] or sum(lengths) != timing["tokens"]
                    or sum(batches) != timing["tokens"]
                    or any(not isinstance(x, int) or not 0 < x <= batch_size for x in lengths + batches)
                    or any(not math.isfinite(timing[key]) or timing[key] < 0
                           for key in ("model_context_init_ms", "first_pass_ms"))):
                raise ValueError(f"{name}: invalid timing workload metadata")
            timing["warm_texts_per_second"] = len(texts) * len(samples) * 1000 / sum(samples)
            timing["warm_corpus_pass_ms_median"] = statistics.median(samples)
            timing["warm_corpus_pass_ms_min"] = min(samples)
            timing["warm_corpus_pass_ms_max"] = max(samples)
            invocation["timing"] = timing
        return vectors, invocation, result.stderr


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="local GGUF, also used to verify device model hash")
    parser.add_argument("--device", default="none", help="none, HTP0 or an enumerated GPU device")
    parser.add_argument("--corpus", type=Path, default=Path(__file__).with_name("embeddinggemma2_corpus.json"))
    parser.add_argument("--output", type=Path, required=True, help="new results directory")
    parser.add_argument("--reference-model", type=Path, help="local high-precision GGUF, run on CPU separately")
    parser.add_argument("--model-revision", default="unspecified")
    parser.add_argument("--reference-model-revision", default="unspecified")
    parser.add_argument("--backend-min-cosine", type=float, default=0.998)
    parser.add_argument("--reference-min-cosine", type=float, help="optional separate high-precision acceptance gate")
    parser.add_argument("--batch-size", type=positive, default=2048)
    parser.add_argument("--expected-dimensions", type=positive, default=768)
    parser.add_argument("--threads", type=positive, default=6)
    parser.add_argument("--repetitions", type=positive, default=5)
    parser.add_argument("--warmups", type=positive, default=1)
    parser.add_argument("--flash-attn", choices=("on", "off"), default="off")
    parser.add_argument("--timeout", type=positive, default=600, help="seconds per process")
    parser.add_argument("--adb", metavar="SERIAL", help="use an already connected and provisioned Android device")
    parser.add_argument("--adb-port", type=positive, default=5037)
    parser.add_argument("--remote-dir", default="/data/local/tmp/embeddinggemma2")
    parser.add_argument("--remote-binary")
    parser.add_argument("--remote-model")
    parser.add_argument("--remote-reference-model")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    for name in ("repetitions", "warmups"):
        if getattr(args, name) > 1000:
            parser.error(f"--{name} must be <= 1000")
    for name in ("backend_min_cosine", "reference_min_cosine"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or not -1 <= value <= 1):
            parser.error(f"--{name.replace('_', '-')} must be finite and in [-1, 1]")
    args.binary, args.model = args.binary.resolve(strict=True), args.model.resolve(strict=True)
    if args.reference_model:
        args.reference_model = args.reference_model.resolve(strict=True)
    try:
        corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
        records = corpus["texts"]
        if not isinstance(records, list) or any(
                not isinstance(item["id"], str) or not item["id"] or not isinstance(item["text"], str)
                or type(item.get("repeat", 1)) is not int or not 1 <= item.get("repeat", 1) <= 1000
                for item in records):
            raise ValueError("texts need string IDs/text and integer repeat counts in [1, 1000]")
        texts = [item["text"] * item.get("repeat", 1) for item in records]
        ids = [item["id"] for item in records]
        for query in corpus.get("retrieval", []):
            if (query["query"] not in ids or not query["documents"]
                    or any(key not in ids for key in query["documents"])
                    or query["relevant"] not in query["documents"]):
                raise ValueError("retrieval checks must refer to corpus IDs and include the relevant document")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"invalid corpus: {error}")
    if (len(texts) < 3 or len(set(texts)) < 2 or len(set(ids)) != len(ids)
            or any(not text or SEPARATOR in text for text in texts)):
        parser.error("corpus needs >=3 nonempty texts, >=2 distinct contents, unique IDs and no reserved separator")
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": 1, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "hashes": {"binary_sha256": sha256(args.binary), "model_sha256": sha256(args.model),
                   "corpus_sha256": sha256(args.corpus), "driver_sha256": sha256(__file__)},
        "invocations": [], "checks": [],
        "measurement": "Sequential corpus passes on one loaded context; tokenization/printing excluded. "
                       "No shared-model async concurrency, power, peak memory or thermal measurements.",
    }
    runner = Runner(args, report)

    def check(name, actual, expected, threshold=PACKING_MIN_COSINE):
        metrics = compare(actual, expected)
        report["checks"].append({"name": name, **metrics, "minimum_required_cosine": threshold,
                                 "passed": threshold is None or metrics["min_cosine"] >= threshold})

    try:
        git = runner.capture(["git", "-C", str(REPO), "rev-parse", "HEAD"])
        report["source_commit"] = git.stdout.strip() if git.returncode == 0 else "unavailable"
        git = runner.capture(["git", "-C", str(REPO), "status", "--porcelain"])
        report["source_status"] = git.stdout.splitlines() if git.returncode == 0 else []
        if args.reference_model:
            report["hashes"]["reference_model_sha256"] = sha256(args.reference_model)
        if runner.adb:
            runner.verify_remote(args.binary, args.remote_binary or f"{args.remote_dir}/bin/{args.binary.name}")
            runner.verify_remote(args.model, args.remote_model or f"{args.remote_dir}/models/{args.model.name}")
            if args.reference_model:
                runner.verify_remote(args.reference_model, args.remote_reference_model
                                     or f"{args.remote_dir}/models/{args.reference_model.name}")
            report["device_info"] = {prop: runner.remote("getprop " + prop) for prop in
                                     ("ro.product.model", "ro.soc.model", "ro.build.version.release", "ro.build.fingerprint")}
        else:
            report["device_info"] = host_device_info()
        report["build_version"] = runner.version()
        packed, timing, _ = runner.run("packed", texts, repetitions=args.repetitions)
        singles = [runner.run(f"single-{index:02}", [text])[0][0] for index, text in enumerate(texts)]
        check("packed_vs_single", packed, singles)
        reordered = runner.run("reordered", list(reversed(texts)))[0]
        check("output_order", list(reversed(reordered)), packed)
        changed = texts.copy()
        changed[1] = next(text for text in texts[2:] + texts[:1] if text != texts[1])
        neighbors = runner.run("changed-neighbor", changed)[0]
        check("neighbor_isolation", [row for i, row in enumerate(neighbors) if i != 1],
              [row for i, row in enumerate(packed) if i != 1])
        lengths = timing["timing"]["tokens_per_text"]
        partial_size, partial_indices = partial_batch_indices(lengths)
        partial_texts = [texts[index] for index in partial_indices]
        partial, partial_timing, _ = runner.run("partial-batches", partial_texts, batch_size=partial_size)
        check("partial_batches", partial, [packed[index] for index in partial_indices])
        batch_tokens = partial_timing["timing"]["batch_tokens"]
        report["checks"].append({"name": "partial_batch_exercised", "batch_tokens": batch_tokens,
                                 "batch_capacity": partial_timing["batch_size"],
                                 "passed": len(batch_tokens) > 1 and batch_tokens[-1] < partial_timing["batch_size"]})
        if args.device == "none":
            cpu = packed
        else:
            cpu, cpu_timing, _ = runner.run("cpu-same-quantization", texts, device="none")
            if cpu_timing["timing"]["tokens_per_text"] != lengths:
                raise ValueError("CPU and candidate token counts differ")
        check("backend_vs_cpu_same_quantization", packed, cpu, args.backend_min_cosine)
        if args.reference_model:
            reference, reference_timing, _ = runner.run("cpu-high-precision", texts, device="none", model=args.reference_model)
            if reference_timing["timing"]["tokens_per_text"] != lengths:
                raise ValueError("high-precision reference token counts differ")
            check("candidate_vs_high_precision", packed, reference, args.reference_min_cosine)
            check("cpu_quantized_vs_high_precision", cpu, reference, args.reference_min_cosine)
        by_id = dict(zip(ids, packed))
        for query in corpus.get("retrieval", []):
            scores = {key: cosine(by_id[query["query"]], by_id[key]) for key in query["documents"]}
            ranked = sorted(scores, key=lambda key: scores[key], reverse=True)
            report["checks"].append({"name": "retrieval_" + query["query"], "scores": scores,
                                     "passed": ranked[0] == query["relevant"]
                                     and sum(score == scores[ranked[0]] for score in scores.values()) == 1})
        if args.device.startswith("HTP"):
            _, _, diagnostic = runner.run("htp-diagnostic", texts, diagnostic=True)
            evidence = htp_compute_records(diagnostic)
            report["checks"].append({"name": "htp_compute_evidence", "compute_profile_records": len(evidence),
                                     "examples": evidence[:10], "passed": bool(evidence)})
        report["passed"] = all(item["passed"] for item in report["checks"])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired) as error:
        report.update(passed=False, error=str(error))
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    logging.getLogger("embeddinggemma2-eval").info("%s: %s", "PASS" if report["passed"] else "FAIL", args.output / "results.json")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
