#!/usr/bin/env python3
"""Run the Qwen3.8 27B Vulkan speculative decoding campaign."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import gzip
import hashlib
import json
import os
import re
import signal
import statistics
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

SOURCE_COMMIT = "d21a547da25d50bee71c2ea941df56414030d0c5"
DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
DATASET_ROW = 350
DATASET_ID = "66fa6702bb02136c067c6abb"
TOKENS_PER_SLOT = 131072
OUTPUT_TOKENS = 1024
TAG_WIDTH = 16
PROFILE_ENV = {
    "GGML_VK_PERF_LOGGER": "1",
    "GGML_VK_PERF_LOGGER_FREQUENCY": "1",
    "GGML_VK_SYNC_LOGGER": "1",
    "GGML_SCHED_DEBUG": "2",
}
QUANTS = ("q4", "q8")
MODES = ("off", "mtp", "dspark", "dflash")
CONCURRENCIES = (1, 2, 4)
SHAPES = ("cold-10k", "cold-100k", "warm-100k-plus-10k")
SPEC_TYPES = {
    "off": "none",
    "mtp": "draft-mtp",
    "dspark": "draft-dspark",
    "dflash": "draft-dflash",
}
TARGET_NAMES = {
    "q4": "Qwen3.8-27B-UD-Q4_K_XL.gguf",
    "q8": "Qwen3.8-27B-Q8_0.gguf",
}
DRAFT_NAMES = {
    ("q4", "mtp"): "mtp-Qwen3.8-27B-Q4_0.gguf",
    ("q8", "mtp"): "mtp-Qwen3.8-27B-Q8_0.gguf",
    ("q4", "dflash"): "dflash-Qwen3.8-27B-Q4_0.gguf",
    ("q8", "dflash"): "dflash-Qwen3.8-27B-Q8_0.gguf",
    ("q4", "dspark"): "dspark-Qwen3.8-27B-Q4_0.gguf",
    ("q8", "dspark"): "dspark-Qwen3.8-27B-Q8_0.gguf",
}
SHAPE_LENGTHS = {
    "cold-10k": (10000, 0, 10000),
    "cold-100k": (100000, 0, 100000),
    "warm-100k-plus-10k": (110000, 100000, 10000),
}
METRICS = (
    "llamacpp:spec_decode_num_draft_tokens_total",
    "llamacpp:spec_decode_num_accepted_tokens_total",
    "llamacpp:spec_decode_num_drafts_total",
)


@dataclasses.dataclass(frozen=True)
class Layout:
    root: Path

    @property
    def source(self) -> Path:
        return self.root / "source"

    @property
    def build(self) -> Path:
        return self.root / "build"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def fixture_dir(self) -> Path:
        return self.root / "fixtures"

    @property
    def venv_python(self) -> Path:
        return self.root / "venv" / "bin" / "python"

    def target(self, quant: str) -> Path:
        return self.root / "models" / "target" / quant / TARGET_NAMES[quant]

    def draft(self, quant: str, mode: str) -> Path:
        return self.root / "models" / "draft" / quant / DRAFT_NAMES[(quant, mode)]


@dataclasses.dataclass
class Server:
    layout: Layout
    quant: str
    mode: str
    concurrency: int
    port: int
    log_path: Path
    profile: bool = False
    process: subprocess.Popen[str] | None = None
    command: list[str] | None = None
    _log_file: Any = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "Server":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.command = server_command(
            self.layout, self.quant, self.mode, self.concurrency, self.port
        )
        env = os.environ.copy()
        env["GGML_VK_VISIBLE_DEVICES"] = "0"
        if self.profile:
            env.update(PROFILE_ENV)
        self.log_path.write_text("", encoding="ascii")
        self._log_file = self.log_path.open("a", encoding="utf-8", errors="backslashreplace")
        self.process = subprocess.Popen(
            self.command,
            cwd=self.layout.source,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            self._wait_ready()
            validate_startup(self.log_path, self.concurrency, self.mode)
        except Exception:
            self.stop()
            raise
        return self

    def _wait_ready(self, timeout: float = 1200.0) -> None:
        deadline = time.monotonic() + timeout
        last_error = ""
        while time.monotonic() < deadline:
            assert self.process is not None
            if self.process.poll() is not None:
                text = read_text(self.log_path)
                raise RuntimeError(
                    f"server exited with {self.process.returncode}: {text[-12000:]}"
                )
            try:
                data = get_json(self.base_url + "/health", timeout=5.0)
                if data.get("status") == "ok":
                    self._log_file.flush()
                    return
            except Exception as exc:
                last_error = str(exc)
            time.sleep(1.0)
        raise TimeoutError(f"server did not become ready: {last_error}")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self._log_file is not None:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


def server_command(
    layout: Layout,
    quant: str,
    mode: str,
    concurrency: int,
    port: int,
) -> list[str]:
    command = [
        str(layout.build / "bin" / "llama-server"),
        "-m",
        str(layout.target(quant)),
        "--device",
        "Vulkan0",
        "-ngl",
        "all",
        "-fa",
        "on",
        "-b",
        "2048",
        "-ub",
        "512",
        "--parallel",
        str(concurrency),
        "--ctx-size",
        str(TOKENS_PER_SLOT * concurrency),
        "--no-kv-unified",
        "-ctk",
        "f16",
        "-ctv",
        "f16",
        "-ctkd",
        "f16",
        "-ctvd",
        "f16",
        "--cache-ram",
        "8192",
        "--cache-idle-slots",
        "--metrics",
        "--perf",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--spec-type",
        SPEC_TYPES[mode],
        "--log-verbosity",
        "5" if profile else "4",
    ]
    if mode != "off":
        command.extend(
            [
                "--spec-draft-model",
                str(layout.draft(quant, mode)),
                "--spec-draft-device",
                "Vulkan0",
                "--spec-draft-ngl",
                "all",
            ]
        )
    return command


def validate_startup(log_path: Path, concurrency: int, mode: str) -> None:
    text = read_text(log_path)
    slot_line = (
        f"initializing, n_slots = {concurrency}, n_ctx_slot = {TOKENS_PER_SLOT}, "
        "kv_unified = 'false'"
    )
    required = [slot_line, "n_ctx_seq             = 131072", "Vulkan0 model buffer size"]
    if mode != "off":
        required.append(f"adding speculative implementation '{SPEC_TYPES[mode]}'")
    missing = [value for value in required if value not in text]
    offload_count = len(re.findall(r"offloaded\s+\d+/\d+\s+layers to GPU", text))
    expected_offloads = 1 if mode == "off" else 2
    if offload_count < expected_offloads:
        missing.append(f"{expected_offloads} complete GPU offload records")
    bad = (
        "VK_ERROR_DEVICE_LOST",
        "failed to allocate",
        "falling back to CPU",
        "using device CPU",
    )
    found_bad = [value for value in bad if value.lower() in text.lower()]
    if missing or found_bad:
        raise RuntimeError(f"startup gate failed: missing={missing}, errors={found_bad}")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def post_json(url: str, payload: dict[str, Any], timeout: float = 3600.0) -> Any:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def get_json(url: str, timeout: float = 30.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def get_text(url: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def stream_completion(
    base_url: str,
    prompt: list[int],
    slot: int,
    cache_prompt: bool,
    n_predict: int,
    barrier: threading.Barrier,
    start_ns: list[int],
    on_first: Callable[[], None] | None = None,
) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": 1,
        "ignore_eos": True,
        "stream": True,
        "return_tokens": True,
        "id_slot": slot,
        "cache_prompt": cache_prompt,
    }
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        base_url + "/completion",
        data=data,
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    barrier.wait()
    token_events_ns: list[int] = []
    generated_tokens: list[int] = []
    final: dict[str, Any] | None = None
    try:
        with urllib.request.urlopen(request, timeout=7200) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if not body or body == "[DONE]":
                    continue
                chunk = json.loads(body)
                tokens = chunk.get("tokens") or []
                if tokens:
                    now_ns = time.perf_counter_ns()
                    if not token_events_ns and on_first is not None:
                        on_first()
                    token_events_ns.extend([now_ns] * len(tokens))
                    generated_tokens.extend(int(token) for token in tokens)
                if chunk.get("stop"):
                    final = chunk
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"completion HTTP {exc.code}: {body}") from exc
    end_ns = time.perf_counter_ns()
    if final is None:
        raise RuntimeError("stream ended without a final completion chunk")
    if not token_events_ns:
        raise RuntimeError("stream returned no token events")
    timings = final.get("timings") or {}
    return {
        "slot": slot,
        "start_ns": start_ns[0],
        "first_ns": token_events_ns[0],
        "last_ns": token_events_ns[-1],
        "end_ns": end_ns,
        "token_events_ns": token_events_ns,
        "generated_tokens": generated_tokens,
        "timings": timings,
        "stop_type": final.get("stop_type"),
        "truncated": bool(final.get("truncated", False)),
    }


def run_barrier(
    base_url: str,
    prompts: list[list[int]],
    cache_prompt: bool,
    n_predict: int,
    on_all_first: Callable[[], None] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    concurrency = len(prompts)
    barrier = threading.Barrier(concurrency + 1)
    start_ns = [0]
    first_lock = threading.Lock()
    first_count = [0]

    def note_first() -> None:
        with first_lock:
            first_count[0] += 1
            if first_count[0] == concurrency and on_all_first is not None:
                on_all_first()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                stream_completion,
                base_url,
                prompt,
                slot,
                cache_prompt,
                n_predict,
                barrier,
                start_ns,
                note_first,
            )
            for slot, prompt in enumerate(prompts)
        ]
        start_ns[0] = time.perf_counter_ns()
        barrier.wait()
        results = [future.result() for future in futures]
    return start_ns[0], sorted(results, key=lambda item: item["slot"])


def parse_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        if not value or "{" in name:
            continue
        try:
            values[name] = float(value.strip())
        except ValueError:
            continue
    return values


def fetch_metrics(base_url: str) -> dict[str, float]:
    return parse_metrics(get_text(base_url + "/metrics"))


def metric_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {name: after.get(name, 0.0) - before.get(name, 0.0) for name in METRICS}


def acceptance_from_deltas(mode: str, deltas: dict[str, float]) -> float | None:
    if mode == "off":
        return None
    drafted = deltas[METRICS[0]]
    accepted = deltas[METRICS[1]]
    steps = deltas[METRICS[2]]
    if drafted <= 0 or steps <= 0:
        raise RuntimeError(f"no speculative verification activity: {deltas}")
    return 1.0 + accepted / steps


def tag_id(quant: str, mode: str, concurrency: int, rep: int, slot: int, group: int) -> int:
    return (
        1000
        + QUANTS.index(quant) * 50000
        + MODES.index(mode) * 10000
        + CONCURRENCIES.index(concurrency) * 2000
        + rep * 20
        + slot * 3
        + group
    )


def tagged_prompt(base_tokens: list[int], length: int, value: int) -> list[int]:
    if len(base_tokens) < length:
        raise RuntimeError(f"fixture has {len(base_tokens)} tokens, need {length}")
    result = list(base_tokens[:length])
    result[:TAG_WIDTH] = [value] + [0] * (TAG_WIDTH - 1)
    if len(result) != length:
        raise AssertionError("tagging changed prompt length")
    return result


def request_prompts(
    base_tokens: list[int],
    quant: str,
    mode: str,
    concurrency: int,
    rep: int,
    shape: str,
) -> tuple[list[list[int]], list[int]]:
    length = SHAPE_LENGTHS[shape][0]
    group = 0 if shape == "cold-10k" else 1
    tags = [tag_id(quant, mode, concurrency, rep, slot, group) for slot in range(concurrency)]
    return [tagged_prompt(base_tokens, length, value) for value in tags], tags


def validate_requests(
    requests: list[dict[str, Any]],
    shape: str,
    concurrency: int,
    n_predict: int,
) -> None:
    length, expected_cache, expected_fresh = SHAPE_LENGTHS[shape]
    if len(requests) != concurrency:
        raise RuntimeError(f"expected {concurrency} streams, got {len(requests)}")
    for request in requests:
        timings = request["timings"]
        actual = (int(timings.get("cache_n", -1)), int(timings.get("prompt_n", -1)))
        expected = (expected_cache, expected_fresh)
        if actual != expected:
            raise RuntimeError(f"{shape} slot {request['slot']} cache/fresh {actual}, expected {expected}")
        if actual[0] + actual[1] != length:
            raise RuntimeError(f"{shape} prompt accounting mismatch: {actual} vs {length}")
        predicted = int(timings.get("predicted_n", -1))
        if predicted != n_predict:
            raise RuntimeError(f"slot {request['slot']} predicted {predicted}, expected {n_predict}")
        if len(request["generated_tokens"]) != n_predict:
            raise RuntimeError(
                f"slot {request['slot']} streamed {len(request['generated_tokens'])} tokens, expected {n_predict}"
            )
        if request["truncated"]:
            raise RuntimeError(f"{shape} slot {request['slot']} reported context truncation")


def aggregate_shape(
    barrier_ns: int,
    requests: list[dict[str, Any]],
    fresh_per_request: int,
) -> dict[str, float | int]:
    all_first_ns = max(request["first_ns"] for request in requests)
    final_ns = max(request["end_ns"] for request in requests)
    ttft_s = (all_first_ns - barrier_ns) / 1e9
    decode_s = (final_ns - all_first_ns) / 1e9
    if ttft_s <= 0 or decode_s <= 0:
        raise RuntimeError(f"invalid phase durations: ttft={ttft_s}, decode={decode_s}")
    post_first_tokens = sum(
        sum(event_ns > all_first_ns for event_ns in request["token_events_ns"])
        for request in requests
    )
    if post_first_tokens <= 0:
        raise RuntimeError("no phase-pure decode tokens after all streams reached first token")
    itl_ms = statistics.mean(
        (request["last_ns"] - request["first_ns"])
        / 1e6
        / (len(request["token_events_ns"]) - 1)
        for request in requests
    )
    return {
        "fresh_tokens": fresh_per_request * len(requests),
        "post_first_tokens": post_first_tokens,
        "aggregate_prefill_tok_s": fresh_per_request * len(requests) / ttft_s,
        "aggregate_decode_tok_s": post_first_tokens / decode_s,
        "ttft_ms": ttft_s * 1000.0,
        "average_itl_ms": itl_ms,
        "all_first_ns": all_first_ns,
        "final_ns": final_ns,
    }


def run_shape(
    server: Server,
    base_tokens: list[int],
    quant: str,
    mode: str,
    concurrency: int,
    rep: int,
    shape: str,
    n_predict: int = OUTPUT_TOKENS,
    on_all_first: Callable[[], None] | None = None,
) -> dict[str, Any]:
    prompts, tags = request_prompts(base_tokens, quant, mode, concurrency, rep, shape)
    before = fetch_metrics(server.base_url)
    barrier_ns, requests = run_barrier(
        server.base_url,
        prompts,
        cache_prompt=shape == "warm-100k-plus-10k",
        n_predict=n_predict,
        on_all_first=on_all_first,
    )
    after = fetch_metrics(server.base_url)
    validate_requests(requests, shape, concurrency, n_predict)
    deltas = metric_deltas(before, after)
    if n_predict == 1:
        all_first_ns = max(request["first_ns"] for request in requests)
        ttft_s = (all_first_ns - barrier_ns) / 1e9
        if ttft_s <= 0:
            raise RuntimeError(f"invalid prefill duration: {ttft_s}")
        aggregate = {
            "fresh_tokens": SHAPE_LENGTHS[shape][2] * concurrency,
            "aggregate_prefill_tok_s": SHAPE_LENGTHS[shape][2] * concurrency / ttft_s,
            "ttft_ms": ttft_s * 1000.0,
            "all_first_ns": all_first_ns,
        }
    else:
        aggregate = aggregate_shape(barrier_ns, requests, SHAPE_LENGTHS[shape][2])
    return {
        "shape": shape,
        "rep": rep,
        "tags": tags,
        "barrier_ns": barrier_ns,
        "requests": requests,
        "metric_deltas": deltas,
        "average_acceptance_length": acceptance_from_deltas(mode, deltas) if n_predict > 1 else None,
        **aggregate,
    }


def run_protocol(
    server: Server,
    base_tokens: list[int],
    quant: str,
    mode: str,
    concurrency: int,
    rep: int,
) -> list[dict[str, Any]]:
    cold_10k = run_shape(server, base_tokens, quant, mode, concurrency, rep, "cold-10k")
    cold_100k = run_shape(server, base_tokens, quant, mode, concurrency, rep, "cold-100k")
    primer = prime_warm_checkpoint(server, base_tokens, quant, mode, concurrency, rep)
    warm = run_shape(server, base_tokens, quant, mode, concurrency, rep, "warm-100k-plus-10k")
    warm["primer"] = primer
    return [cold_10k, cold_100k, warm]


def relative_spread(values: list[float]) -> float:
    median = statistics.median(values)
    return (max(values) - min(values)) / median if median else 0.0


def needs_third_rep(repetitions: list[list[dict[str, Any]]]) -> bool:
    for shape_index in range(len(SHAPES)):
        pp = [float(rep[shape_index]["aggregate_prefill_tok_s"]) for rep in repetitions]
        tg = [float(rep[shape_index]["aggregate_decode_tok_s"]) for rep in repetitions]
        if relative_spread(pp) > 0.03 or relative_spread(tg) > 0.03:
            return True
    return False


def vocab_hash(path: Path) -> str:
    import gguf


    reader = gguf.GGUFReader(str(path), "r")
    keys = sorted(key for key in reader.fields if key.startswith("tokenizer.ggml."))
    if not keys:
        raise RuntimeError(f"{path} has no tokenizer.ggml fields")
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if all(hasattr(value, attr) for attr in ("dtype", "ndim", "shape", "flat", "tobytes")):
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(struct.pack("<I", value.ndim))
            for size in value.shape:
                digest.update(struct.pack("<Q", int(size)))
            if value.dtype.kind in "OUS":
                for item in value.flat:
                    update(item.item() if hasattr(item, "item") else item)
            else:
                digest.update(value.tobytes())
        elif isinstance(value, (list, tuple)):
            digest.update(struct.pack("<Q", len(value)))
            for item in value:
                update(item)
        elif isinstance(value, bytes):
            digest.update(struct.pack("<Q", len(value)))
            digest.update(value)
        elif isinstance(value, str):
            encoded = value.encode("utf-8")
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
        elif hasattr(value, "item"):
            update(value.item())
        else:
            update(str(value))

    for key in keys:
        update(key)
        update(reader.fields[key].contents())
    return digest.hexdigest()


def token_hash(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack("<I", int(token)))
    return digest.hexdigest()


def render_dataset_prompt(row: dict[str, Any]) -> str:
    return (
        "Repository context:\n"
        + row["context"]
        + "\n\nQuestion:\n"
        + row["question"]
        + "\n\nChoices:\nA. "
        + row["choice_A"]
        + "\nB. "
        + row["choice_B"]
        + "\nC. "
        + row["choice_C"]
        + "\nD. "
        + row["choice_D"]
        + "\n\nAnswer:\n"
    )


def prepare_fixture(layout: Layout) -> None:
    data_path = layout.fixture_dir / "longbench-v2-data.json"
    with data_path.open(encoding="utf-8") as handle:
        dataset = json.load(handle)
    row = dataset[DATASET_ROW]
    if row.get("_id") != DATASET_ID:
        raise RuntimeError(f"dataset row mismatch: {row.get('_id')} != {DATASET_ID}")
    prompt = render_dataset_prompt(row)
    fixture_results: dict[str, Any] = {}
    token_lists: dict[str, list[int]] = {}
    for quant in QUANTS:
        log_path = layout.results / "fixture" / f"{quant}-server.log"
        port = 19080 + QUANTS.index(quant)
        with Server(layout, quant, "off", 1, port, log_path) as server:
            response = post_json(
                server.base_url + "/tokenize",
                {"content": prompt, "add_special": False, "parse_special": True},
            )
            tokens = [int(token) for token in response["tokens"]]
            token_lists[quant] = tokens
            fixture_results[quant] = {
                "target": str(layout.target(quant)),
                "vocab_hash": vocab_hash(layout.target(quant)),
                "token_hash": token_hash(tokens),
                "token_count": len(tokens),
                "server_command": server.command,
                "server_log": str(log_path),
            }
        archive_and_normalize(log_path)
    if fixture_results["q4"]["vocab_hash"] != fixture_results["q8"]["vocab_hash"]:
        raise RuntimeError(f"target vocabulary hashes differ: {fixture_results}")
    if token_lists["q4"] != token_lists["q8"]:
        raise RuntimeError("target token arrays differ")
    base_tokens = token_lists["q4"]
    if len(base_tokens) < 110000:
        raise RuntimeError(f"selected row has only {len(base_tokens)} tokens")
    base_tokens = base_tokens[:110000]
    if base_tokens[:100000] != base_tokens[:110000][:100000]:
        raise AssertionError("100k prompt is not the exact 110k prefix")
    layout.fixture_dir.mkdir(parents=True, exist_ok=True)
    write_json(layout.fixture_dir / "base-tokens.json", base_tokens)
    manifest = {
        "source_commit": SOURCE_COMMIT,
        "dataset_revision": DATASET_REVISION,
        "dataset_row": DATASET_ROW,
        "dataset_id": DATASET_ID,
        "domain": row.get("domain"),
        "sub_domain": row.get("sub_domain"),
        "rendered_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "base_110k_token_hash": token_hash(base_tokens),
        "target_results": fixture_results,
        "lengths": {shape: values[0] for shape, values in SHAPE_LENGTHS.items()},
        "tag_width": TAG_WIDTH,
        "numeric_prompt_adds_bos": False,
    }
    write_json(layout.fixture_dir / "fixture-manifest.json", manifest)


def load_base_tokens(layout: Layout) -> list[int]:
    path = layout.fixture_dir / "base-tokens.json"
    if not path.exists():
        raise RuntimeError(f"missing fixture: {path}; run prepare-fixture first")
    with path.open(encoding="ascii") as handle:
        tokens = json.load(handle)
    if len(tokens) != 110000:
        raise RuntimeError(f"fixture has {len(tokens)} tokens, expected 110000")
    return [int(token) for token in tokens]


def smoke_pairs(layout: Layout) -> None:
    base_tokens = load_base_tokens(layout)
    gate_path = layout.results / "gates.json"
    gates: dict[str, Any] = {
        "source_commit": SOURCE_COMMIT,
        "pairs": {},
    }
    for quant in QUANTS:
        for mode in MODES:
            key = f"{quant}/{mode}"
            log_path = layout.results / "gates" / quant / f"{mode}.log"
            port = 19100 + QUANTS.index(quant) * 10 + MODES.index(mode)
            record: dict[str, Any] = {"quant": quant, "mode": mode, "supported": False}
            try:
                with Server(layout, quant, mode, 1, port, log_path) as server:
                    prompt = tagged_prompt(
                        base_tokens,
                        64,
                        tag_id(quant, mode, 1, 80, 0, 2),
                    )
                    before = fetch_metrics(server.base_url)
                    barrier_ns, requests = run_barrier(
                        server.base_url,
                        [prompt],
                        cache_prompt=False,
                        n_predict=OUTPUT_TOKENS,
                    )
                    after = fetch_metrics(server.base_url)
                    timings = requests[0]["timings"]
                    if (int(timings.get("cache_n", -1)), int(timings.get("prompt_n", -1))) != (0, 64):
                        raise RuntimeError(f"smoke prompt accounting failed: {timings}")
                    if int(timings.get("predicted_n", -1)) != OUTPUT_TOKENS:
                        raise RuntimeError(f"smoke output count failed: {timings}")
                    deltas = metric_deltas(before, after)
                    acceptance_from_deltas(mode, deltas)
                    record.update(
                        {
                            "supported": True,
                            "server_command": server.command,
                            "barrier_ns": barrier_ns,
                            "request": requests[0],
                            "metric_deltas": deltas,
                            "log": str(log_path),
                        }
                    )
            except Exception as exc:
                record["error"] = str(exc)
                record["log"] = str(log_path)
            finally:
                archive_and_normalize(log_path)
            gates["pairs"][key] = record
            write_json(gate_path, gates)


def load_gates(layout: Layout) -> dict[str, Any]:
    path = layout.results / "gates.json"
    if not path.exists():
        raise RuntimeError(f"missing gates: {path}; run smoke first")
    with path.open(encoding="ascii") as handle:
        return json.load(handle)


def run_unprofiled(layout: Layout) -> None:
    base_tokens = load_base_tokens(layout)
    gates = load_gates(layout)
    for quant in QUANTS:
        for mode in MODES:
            pair = gates["pairs"][f"{quant}/{mode}"]
            for concurrency in CONCURRENCIES:
                raw_path = raw_launch_path(layout, quant, mode, concurrency)
                if raw_path.exists():
                    with raw_path.open(encoding="ascii") as handle:
                        prior = json.load(handle)
                    if prior.get("status") == "complete" or (
                        prior.get("status") == "withheld" and not pair.get("supported")
                    ):
                        continue
                log_path = raw_path.with_suffix(".server.log")
                record: dict[str, Any] = {
                    "source_commit": SOURCE_COMMIT,
                    "quant": quant,
                    "mode": mode,
                    "spec_type": SPEC_TYPES[mode],
                    "concurrency": concurrency,
                    "status": "running",
                }
                if not pair.get("supported"):
                    error = str(pair.get("error", "unknown error"))
                    tensor_error = re.search(r"wrong number of tensors; expected \d+, got \d+", error)
                    reason = tensor_error.group(0) if tensor_error else error[-300:]
                    record.update(
                        {
                            "status": "withheld",
                            "reason": f"pair gate failed: {reason}; log: {pair['log']}",
                        }
                    )
                    write_json(raw_path, record)
                    print(f"withheld {quant}/{mode}/c{concurrency}: {reason}", flush=True)
                    continue
                port = 19200 + QUANTS.index(quant) * 100 + MODES.index(mode) * 10 + concurrency
                print(f"starting {quant}/{mode}/c{concurrency}", flush=True)
                try:
                    with Server(layout, quant, mode, concurrency, port, log_path) as server:
                        record["server_command"] = server.command
                        record["server_log"] = str(log_path)
                        record["warmup"] = run_protocol(
                            server, base_tokens, quant, mode, concurrency, 0
                        )
                        repetitions = [
                            run_protocol(server, base_tokens, quant, mode, concurrency, rep)
                            for rep in (1, 2)
                        ]
                        if needs_third_rep(repetitions):
                            repetitions.append(
                                run_protocol(server, base_tokens, quant, mode, concurrency, 3)
                            )
                        record["repetitions"] = repetitions
                        record["rows"] = summarize_launch(record)
                        record["status"] = "complete"
                except Exception as exc:
                    record["status"] = "withheld"
                    record["reason"] = str(exc)
                finally:
                    archive_and_normalize(log_path)
                    write_json(raw_path, record)
                print(f"finished {quant}/{mode}/c{concurrency}: {record['status']}", flush=True)
    run_batched_cross_checks(layout)
    render_all(layout)


def raw_launch_path(layout: Layout, quant: str, mode: str, concurrency: int) -> Path:
    return layout.results / "raw" / quant / mode / f"c{concurrency}.json"


def summarize_launch(record: dict[str, Any]) -> list[dict[str, Any]]:
    repetitions = record["repetitions"]
    rows: list[dict[str, Any]] = []
    for shape_index, shape in enumerate(SHAPES):
        shape_reps = [rep[shape_index] for rep in repetitions]
        pp = [float(item["aggregate_prefill_tok_s"]) for item in shape_reps]
        tg = [float(item["aggregate_decode_tok_s"]) for item in shape_reps]
        ttft = [float(item["ttft_ms"]) for item in shape_reps]
        itl = [float(item["average_itl_ms"]) for item in shape_reps]
        acceptance_values = [
            float(item["average_acceptance_length"])
            for item in shape_reps
            if item["average_acceptance_length"] is not None
        ]
        rows.append(
            {
                "quant": record["quant"],
                "shape": shape,
                "mode": record["mode"],
                "spec_type": record["spec_type"],
                "concurrency": record["concurrency"],
                "fresh_tokens_per_request": SHAPE_LENGTHS[shape][2],
                "cached_tokens_per_request": SHAPE_LENGTHS[shape][1],
                "aggregate_prefill_tok_s": statistics.median(pp),
                "aggregate_decode_tok_s": statistics.median(tg),
                "ttft_ms": statistics.median(ttft),
                "average_itl_ms": statistics.median(itl),
                "average_acceptance_length": (
                    statistics.median(acceptance_values) if acceptance_values else None
                ),
                "repetitions": len(shape_reps),
                "prefill_spread": relative_spread(pp),
                "decode_spread": relative_spread(tg),
            }
        )
    return rows


def run_batched_cross_checks(layout: Layout) -> None:
    output_dir = layout.results / "batched-bench"
    output_dir.mkdir(parents=True, exist_ok=True)
    for quant in QUANTS:
        for concurrency in CONCURRENCIES:
            output_path = output_dir / f"{quant}-c{concurrency}.jsonl"
            command_path = output_path.with_suffix(".command.json")
            if output_path.exists() and command_path.exists():
                with command_path.open(encoding="ascii") as handle:
                    previous = json.load(handle)
                if previous.get("returncode") == 0 and output_path.stat().st_size > 0:
                    continue
            command = [
                str(layout.build / "bin" / "llama-batched-bench"),
                "-m",
                str(layout.target(quant)),
                "--device",
                "Vulkan0",
                "-ngl",
                "all",
                "-fa",
                "on",
                "-c",
                str(TOKENS_PER_SLOT * concurrency),
                "-b",
                "2048",
                "-ub",
                "512",
                "--no-kv-unified",
                "-ctk",
                "f16",
                "-ctv",
                "f16",
                "-npp",
                "10000",
                "-ntg",
                str(OUTPUT_TOKENS),
                "-npl",
                str(concurrency),
                "--output-format",
                "jsonl",
            ]
            result = subprocess.run(
                command,
                cwd=layout.source,
                env={**os.environ, "GGML_VK_VISIBLE_DEVICES": "0"},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=7200,
            )
            write_ascii(output_path, result.stdout)
            write_ascii(output_path.with_suffix(".stderr.log"), result.stderr)
            write_json(
                output_path.with_suffix(".command.json"),
                {"command": command, "returncode": result.returncode},
            )
            if result.returncode != 0:
                raise RuntimeError(f"llama-batched-bench failed for {quant} c{concurrency}")


def append_marker(path: Path, marker: str) -> None:
    with path.open("a", encoding="ascii") as handle:
        handle.write(f"\n@@CAMPAIGN {marker}\n")
        handle.flush()
        os.fsync(handle.fileno())


def prime_warm_checkpoint(
    server: Server,
    base_tokens: list[int],
    quant: str,
    mode: str,
    concurrency: int,
    rep: int,
) -> dict[str, Any]:
    tags = [
        tag_id(quant, mode, concurrency, rep, slot, 1)
        for slot in range(concurrency)
    ]
    prompts = [tagged_prompt(base_tokens, 100004, tag) for tag in tags]
    for prompt in prompts:
        prompt[100000:100004] = [
            (token + 1) % 100000 for token in base_tokens[100000:100004]
        ]
    barrier_ns, requests = run_barrier(
        server.base_url, prompts, cache_prompt=True, n_predict=1
    )
    for request in requests:
        timings = request["timings"]
        if int(timings.get("cache_n", -1)) + int(timings.get("prompt_n", -1)) != 100004:
            raise RuntimeError(f"warm primer prompt accounting failed: {timings}")
        if int(timings.get("predicted_n", -1)) != 1:
            raise RuntimeError(f"warm primer output count failed: {timings}")
    return {"tags": tags, "barrier_ns": barrier_ns, "requests": requests}


def seed_warm_prefix(
    server: Server,
    base_tokens: list[int],
    quant: str,
    mode: str,
    concurrency: int,
    rep: int,
) -> None:
    prompts, _ = request_prompts(
        base_tokens, quant, mode, concurrency, rep, "cold-100k"
    )
    _, requests = run_barrier(
        server.base_url,
        prompts,
        cache_prompt=False,
        n_predict=1,
    )
    validate_requests(requests, "cold-100k", concurrency, 1)
    prime_warm_checkpoint(server, base_tokens, quant, mode, concurrency, rep)


def extract_profile_segment(log_path: Path, start_marker: str, end_marker: str, output: Path) -> None:
    text = read_text(log_path)
    start_token = f"@@CAMPAIGN {start_marker}"
    end_token = f"@@CAMPAIGN {end_marker}"
    start = text.find(start_token)
    end = text.find(end_token, start + len(start_token))
    if start < 0 or end < 0:
        raise RuntimeError(f"missing profile markers {start_marker}/{end_marker}")
    segment = text[start + len(start_token) : end]
    if "Vulkan Profiling Results" not in segment:
        raise RuntimeError(f"no Vulkan profiling sections between {start_marker} and {end_marker}")
    write_utf8(output, segment)


def profile_row(
    layout: Layout,
    base_tokens: list[int],
    row: dict[str, Any],
) -> None:
    quant = row["quant"]
    mode = row["mode"]
    concurrency = int(row["concurrency"])
    shape = row["shape"]
    row_dir = profile_dir(layout, quant, shape, mode, concurrency)
    combined = row_report_path(layout, quant, shape, mode, concurrency)
    if combined.exists():
        return
    row_dir.mkdir(parents=True, exist_ok=True)
    server_log = row_dir / "server.log"
    port = (
        20000
        + QUANTS.index(quant) * 1000
        + MODES.index(mode) * 200
        + CONCURRENCIES.index(concurrency) * 50
        + SHAPES.index(shape)
    )
    rep = 90 + SHAPES.index(shape)
    with Server(layout, quant, mode, concurrency, port, server_log, profile=True) as server:
        if shape == "warm-100k-plus-10k":
            seed_warm_prefix(server, base_tokens, quant, mode, concurrency, rep)
        append_marker(server_log, "PREFILL_START")
        prefill = run_shape(
            server,
            base_tokens,
            quant,
            mode,
            concurrency,
            rep,
            shape,
            n_predict=1,
        )
        append_marker(server_log, "PREFILL_END")

        if shape == "warm-100k-plus-10k":
            seed_warm_prefix(server, base_tokens, quant, mode, concurrency, rep + 1)
        append_marker(server_log, "DECODE_START")
        decode = run_shape(
            server,
            base_tokens,
            quant,
            mode,
            concurrency,
            rep + 1,
            shape,
            n_predict=OUTPUT_TOKENS,
            on_all_first=lambda: append_marker(server_log, "DECODE_ALL_FIRST"),
        )
        append_marker(server_log, "DECODE_END")
        command = server.command

    prefill_log = row_dir / "prefill.log"
    decode_log = row_dir / "decode.log"
    extract_profile_segment(server_log, "PREFILL_START", "PREFILL_END", prefill_log)
    extract_profile_segment(server_log, "DECODE_ALL_FIRST", "DECODE_END", decode_log)
    analyzer = layout.source / "vulkan_profiling_analyzer.py"
    profile_commands: list[list[str]] = []
    for phase, log_path in (("prefill", prefill_log), ("decode", decode_log)):
        markdown = row_dir / f"{phase}.md"
        analyzer_command = [
            str(layout.venv_python),
            str(analyzer),
            str(log_path),
            "--markdown-output",
            str(markdown),
        ]
        result = subprocess.run(
            analyzer_command,
            cwd=row_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3600,
        )
        write_ascii(row_dir / f"{phase}.analyzer.stdout.log", result.stdout)
        write_ascii(row_dir / f"{phase}.analyzer.stderr.log", result.stderr)
        if result.returncode != 0 or not markdown.exists():
            raise RuntimeError(f"analyzer failed for {row_key(row)} {phase}")
        if "Vulkan Profiling Results" not in read_text(log_path):
            raise RuntimeError(f"empty profile log for {row_key(row)} {phase}")
        chart = row_dir / "matmul_normalized_times.png"
        if chart.exists():
            chart.rename(row_dir / f"{phase}-matmul.png")
            write_utf8(
                markdown,
                read_text(markdown).replace(
                    "(matmul_normalized_times.png)", f"({phase}-matmul.png)"
                ),
            )
        gzip_original(log_path)
        normalize_ascii_file(log_path)
        normalize_ascii_file(markdown)
        profile_commands.append(analyzer_command)
    gzip_original(server_log)
    normalize_ascii_file(server_log)
    write_json(
        row_dir / "profile.json",
        {
            "row": row,
            "server_command": command,
            "analyzer_sha256": hashlib.sha256(analyzer.read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment": PROFILE_ENV,
            "prefill_replay": prefill,
            "decode_replay": decode,
            "analyzer_commands": profile_commands,
            "prefill_log": str(prefill_log),
            "decode_log": str(decode_log),
        },
    )
    write_row_report(layout, row)


def run_profiles(layout: Layout) -> None:
    base_tokens = load_base_tokens(layout)
    rows, _ = collect_rows(layout)
    for row in rows:
        print(f"profiling {row_key(row)}", flush=True)
        profile_row(layout, base_tokens, row)
        print(f"profiled {row_key(row)}", flush=True)
    render_all(layout)


def profile_dir(
    layout: Layout, quant: str, shape: str, mode: str, concurrency: int
) -> Path:
    return layout.results / "profiles" / quant / shape / mode / f"c{concurrency}"


def shape_file_token(shape: str) -> str:
    return {
        "cold-10k": "pp10000",
        "cold-100k": "pp100000",
        "warm-100k-plus-10k": "pp100000p10000",
    }[shape]


def row_report_path(
    layout: Layout, quant: str, shape: str, mode: str, concurrency: int
) -> Path:
    spec_type = "off" if mode == "off" else SPEC_TYPES[mode]
    return layout.results / quant / f"{shape_file_token(shape)}_{spec_type}_{concurrency}.md"


def row_key(row: dict[str, Any]) -> str:
    return f"{row['quant']}|{row['shape']}|{row['mode']}|{row['concurrency']}"


def collect_rows(layout: Layout) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    withheld: list[dict[str, Any]] = []
    for quant in QUANTS:
        for mode in MODES:
            for concurrency in CONCURRENCIES:
                path = raw_launch_path(layout, quant, mode, concurrency)
                if not path.exists():
                    withheld.append(
                        {
                            "quant": quant,
                            "mode": mode,
                            "concurrency": concurrency,
                            "reason": "missing raw launch",
                        }
                    )
                    continue
                with path.open(encoding="ascii") as handle:
                    launch = json.load(handle)
                if launch.get("status") == "complete":
                    rows.extend(launch["rows"])
                else:
                    for shape in SHAPES:
                        withheld.append(
                            {
                                "quant": quant,
                                "shape": shape,
                                "mode": mode,
                                "concurrency": concurrency,
                                "reason": launch.get("reason", "withheld"),
                            }
                        )
    return rows, withheld


def write_row_report(layout: Layout, row: dict[str, Any]) -> None:
    output = row_report_path(
        layout, row["quant"], row["shape"], row["mode"], int(row["concurrency"])
    )
    row_dir = profile_dir(
        layout, row["quant"], row["shape"], row["mode"], int(row["concurrency"])
    )
    raw_path = raw_launch_path(layout, row["quant"], row["mode"], int(row["concurrency"]))
    profile_json = row_dir / "profile.json"
    prefill_md = row_dir / "prefill.md"
    decode_md = row_dir / "decode.md"
    acceptance = row["average_acceptance_length"]
    acceptance_text = "-" if acceptance is None else f"{acceptance:.3f}"
    prefill_syncs = sum(line.strip() == "sync" for line in read_text(row_dir / "prefill.log").splitlines())
    decode_syncs = sum(line.strip() == "sync" for line in read_text(row_dir / "decode.log").splitlines())
    lines = [
        f"# {row['quant']} {row['shape']} {row['spec_type']} c{row['concurrency']}",
        "",
        f"- Source commit: `{SOURCE_COMMIT}`",
        f"- Target: `{layout.target(row['quant'])}`",
        f"- Raw production metrics: `{os.path.relpath(raw_path, output.parent)}`",
        f"- Profile evidence: `{os.path.relpath(profile_json, output.parent)}`",
        f"- Aggregate prefill: {row['aggregate_prefill_tok_s']:.3f} tok/s",
        f"- Aggregate decode: {row['aggregate_decode_tok_s']:.3f} tok/s",
        f"- TTFT: {row['ttft_ms']:.3f} ms",
        f"- Average ITL: {row['average_itl_ms']:.3f} ms",
        f"- Average acceptance length: {acceptance_text}",
        f"- Repetitions: {row['repetitions']}",
        f"- Spread: prefill {row['prefill_spread'] * 100:.2f}%, decode {row['decode_spread'] * 100:.2f}%",
        "",
        "## Prefill",
        "",
        read_text(prefill_md).replace(
            "(prefill-matmul.png)",
            f"({os.path.relpath(row_dir / 'prefill-matmul.png', output.parent)})",
        ),
        "",
        "## Decode",
        "",
        read_text(decode_md).replace(
            "(decode-matmul.png)",
            f"({os.path.relpath(row_dir / 'decode-matmul.png', output.parent)})",
        ),
        "",
        "## Synchronization",
        "",
        "The built-in logger reports synchronization occurrences only. It does not measure GPU wait duration.",
        f"- Prefill sync events: {prefill_syncs}",
        f"- Decode sync events: {decode_syncs}",
        "",
        f"- Prefill raw log: `{os.path.relpath(row_dir / 'prefill.log', output.parent)}`",
        f"- Decode raw log: `{os.path.relpath(row_dir / 'decode.log', output.parent)}`",
    ]
    write_ascii(output, "\n".join(lines) + "\n")


def render_summary(layout: Layout, rows: list[dict[str, Any]], withheld: list[dict[str, Any]]) -> str:
    header = (
        "| Quant | Prompt shape | Spec type | Concurrent | Fresh tokens/request | "
        "Cached tokens/request | Aggregate prefill tok/s | Aggregate decode tok/s | "
        "TTFT ms | Average ITL ms | Average acceptance length | Repetitions | Spread |"
    )
    separator = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    lines = [
        "# Qwen3.8 27B Vulkan speculative benchmark",
        "",
        f"Source commit: `{SOURCE_COMMIT}`",
        "",
        header,
        separator,
    ]
    order = {
        (quant, shape, mode, concurrency): index
        for index, (quant, shape, mode, concurrency) in enumerate(
            (q, s, m, c)
            for q in QUANTS
            for s in SHAPES
            for m in MODES
            for c in CONCURRENCIES
        )
    }
    for row in sorted(
        rows,
        key=lambda item: order[
            (item["quant"], item["shape"], item["mode"], int(item["concurrency"]))
        ],
    ):
        report = row_report_path(
            layout, row["quant"], row["shape"], row["mode"], int(row["concurrency"])
        )
        report_rel = os.path.relpath(report, layout.results)
        acceptance = row["average_acceptance_length"]
        acceptance_text = "-" if acceptance is None else f"{acceptance:.3f}"
        spread = (
            f"pp {row['prefill_spread'] * 100:.2f}%, "
            f"tg {row['decode_spread'] * 100:.2f}%"
        )
        lines.append(
            f"| {row['quant']} | [{row['shape']}]({report_rel}) | {row['spec_type']} | "
            f"{row['concurrency']} | {row['fresh_tokens_per_request']} | "
            f"{row['cached_tokens_per_request']} | {row['aggregate_prefill_tok_s']:.3f} | "
            f"{row['aggregate_decode_tok_s']:.3f} | {row['ttft_ms']:.3f} | "
            f"{row['average_itl_ms']:.3f} | {acceptance_text} | {row['repetitions']} | {spread} |"
        )
    if withheld:
        lines.extend(["", "## Withheld rows", ""])
        for item in withheld:
            lines.append(
                f"- `{item.get('quant')}/{item.get('shape', '*')}/{item.get('mode')}/c{item.get('concurrency')}`: "
                f"{item.get('reason')}"
            )
    lines.append("")
    return "\n".join(lines)


def render_all(layout: Layout) -> None:
    rows, withheld = collect_rows(layout)
    for row in rows:
        row_dir = profile_dir(
            layout, row["quant"], row["shape"], row["mode"], int(row["concurrency"])
        )
        if (row_dir / "prefill.md").exists() and (row_dir / "decode.md").exists():
            write_row_report(layout, row)
    write_ascii(layout.results / "summary.md", render_summary(layout, rows, withheld))


def recompute_shape(item: dict[str, Any]) -> dict[str, float | int]:
    return aggregate_shape(
        int(item["barrier_ns"]),
        item["requests"],
        int(item["fresh_tokens"]) // len(item["requests"]),
    )


def verify_campaign(layout: Layout) -> None:
    rows, withheld = collect_rows(layout)
    expected = {
        (quant, shape, mode, concurrency)
        for quant in QUANTS
        for shape in SHAPES
        for mode in MODES
        for concurrency in CONCURRENCIES
    }
    actual = {
        (row["quant"], row["shape"], row["mode"], int(row["concurrency"]))
        for row in rows
    }
    withheld_keys = {
        (item["quant"], item.get("shape"), item["mode"], int(item["concurrency"]))
        for item in withheld
        if item.get("shape") in SHAPES
    }
    if actual | withheld_keys != expected:
        missing = expected - actual - withheld_keys
        extra = (actual | withheld_keys) - expected
        raise RuntimeError(f"row key mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    if len(actual) != len(rows):
        raise RuntimeError("duplicate row keys in raw results")
    for quant in QUANTS:
        for mode in MODES:
            for concurrency in CONCURRENCIES:
                path = raw_launch_path(layout, quant, mode, concurrency)
                with path.open(encoding="ascii") as handle:
                    launch = json.load(handle)
                if launch.get("status") != "complete":
                    continue
                protocols = [launch["warmup"], *launch["repetitions"]]
                if len(launch["repetitions"]) not in (2, 3):
                    raise RuntimeError(f"wrong repetition count in {path}")
                for protocol in protocols:
                    if [item["shape"] for item in protocol] != list(SHAPES):
                        raise RuntimeError(f"wrong protocol order in {path}")
                    if protocol[1]["tags"] != protocol[2]["tags"]:
                        raise RuntimeError(f"warm prefix tags differ in {path}")
                    primer = protocol[2]["primer"]
                    if primer["tags"] != protocol[2]["tags"]:
                        raise RuntimeError(f"warm primer tags differ in {path}")
                    if len(primer["requests"]) != concurrency:
                        raise RuntimeError(f"warm primer slot count differs in {path}")
                    for request in primer["requests"]:
                        timings = request["timings"]
                        if int(timings["cache_n"]) + int(timings["prompt_n"]) != 100004:
                            raise RuntimeError(f"warm primer accounting differs in {path}")
                        if int(timings["predicted_n"]) != 1 or request["truncated"]:
                            raise RuntimeError(f"warm primer completion differs in {path}")
                    for item in protocol:
                        validate_requests(
                            item["requests"], item["shape"], concurrency, OUTPUT_TOKENS
                        )
                        acceptance_from_deltas(mode, item["metric_deltas"])
                recomputed_launch = dict(launch)
                for repetition in recomputed_launch["repetitions"]:
                    for item in repetition:
                        aggregate = recompute_shape(item)
                        for key in (
                            "aggregate_prefill_tok_s",
                            "aggregate_decode_tok_s",
                            "ttft_ms",
                            "average_itl_ms",
                        ):
                            if abs(float(item[key]) - float(aggregate[key])) > 1e-9:
                                raise RuntimeError(f"raw aggregate mismatch in {path}: {key}")
                recomputed_rows = summarize_launch(recomputed_launch)
                if json.dumps(recomputed_rows, sort_keys=True) != json.dumps(
                    launch["rows"], sort_keys=True
                ):
                    raise RuntimeError(f"summary row mismatch in {path}")
    rendered = render_summary(layout, rows, withheld)
    summary_path = layout.results / "summary.md"
    if read_text(summary_path) != rendered:
        raise RuntimeError("summary.md differs from fresh raw-result rendering")
    for row in rows:
        report = row_report_path(
            layout, row["quant"], row["shape"], row["mode"], int(row["concurrency"])
        )
        if not report.exists():
            raise RuntimeError(f"missing row report: {report}")
        profile = profile_dir(
            layout, row["quant"], row["shape"], row["mode"], int(row["concurrency"])
        )
        for required in (
            profile / "prefill.md",
            profile / "decode.md",
            profile / "prefill.log",
            profile / "decode.log",
            profile / "profile.json",
        ):
            if not required.exists():
                raise RuntimeError(f"missing row evidence: {required}")
    ascii_paths = [
        Path(__file__),
        *(
            path
            for path in layout.results.rglob("*")
            if path.is_file() and path.suffix in {".md", ".json", ".log", ".txt", ".tsv"}
        ),
        layout.fixture_dir / "fixture-manifest.json",
        layout.fixture_dir / "base-tokens.json",
    ]
    for path in ascii_paths:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                try:
                    chunk.decode("ascii")
                except UnicodeDecodeError as exc:
                    raise RuntimeError(f"non-ASCII byte in {path} near {exc.start}") from exc
    verification = {
        "source_commit": SOURCE_COMMIT,
        "expected_base_rows": 72,
        "published_rows": len(rows),
        "withheld_rows": len(withheld_keys),
        "unique_keys": len(actual),
        "ascii_files_checked": len(ascii_paths),
        "status": "passed",
    }
    write_json(layout.results / "verification.json", verification)


def archive_and_normalize(path: Path) -> None:
    if not path.exists():
        return
    gzip_original(path)
    normalize_ascii_file(path)


def gzip_original(path: Path) -> None:
    archive = path.with_suffix(path.suffix + ".raw.gz")
    with path.open("rb") as source, gzip.open(archive, "wb", compresslevel=6) as target:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)


def normalize_ascii(text: str) -> str:
    replacements = {
        "\u00b5": "u",
        "\u03bc": "u",
        "\u00b1": "+/-",
        "\u2192": "->",
        "\u00d7": "x",
        "\u2026": "...",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("ascii", "backslashreplace").decode("ascii")


def normalize_ascii_file(path: Path) -> None:
    write_ascii(path, normalize_ascii(read_text(path)))


def write_utf8(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_ascii(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalize_ascii(text), encoding="ascii")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def verify_source_commit(layout: Layout) -> None:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=layout.source,
        text=True,
    ).strip()
    if actual != SOURCE_COMMIT:
        raise RuntimeError(f"source commit {actual}, expected {SOURCE_COMMIT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "command",
        choices=("prepare-fixture", "smoke", "unprofiled", "profiles", "render", "verify", "all"),
    )
    args = parser.parse_args()
    layout = Layout(args.root.resolve())
    layout.results.mkdir(parents=True, exist_ok=True)
    verify_source_commit(layout)
    if args.command in ("prepare-fixture", "all"):
        prepare_fixture(layout)
    if args.command in ("smoke", "all"):
        smoke_pairs(layout)
    if args.command in ("unprofiled", "all"):
        run_unprofiled(layout)
    if args.command in ("profiles", "all"):
        run_profiles(layout)
    if args.command in ("render", "all"):
        render_all(layout)
    if args.command in ("verify", "all"):
        verify_campaign(layout)


if __name__ == "__main__":
    main()
