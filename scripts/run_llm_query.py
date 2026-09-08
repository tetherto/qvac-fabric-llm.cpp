#!/usr/bin/env python3
"""Run a local llama.cpp completion and print the answer + performance metrics.

Wraps ``llama-completion`` (CPU-friendly defaults for Qwen3.5). Parses stderr for
device/buffer lines and ``common_perf_print`` timings.

Each run writes under ``runs/{id}/``:

  raw/         verbatim qvac / llama / ggml output (stdout, stderr, llama.log)
  processed/   post-processed summary, layer-trace, result.json

Example:
  python3 scripts/run_llm_query.py "Explain what an NPU is in two sentences."
  python3 scripts/run_llm_query.py -n 128 --json "Hello"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN = REPO_ROOT / "build" / "bin" / "llama-completion"
DEFAULT_MODEL = REPO_ROOT / "models" / "weights" / "Qwen3.5-0.8B-Q4_K_M.gguf"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"


@dataclass
class PerfMetrics:
    sampling_ms: Optional[float] = None
    samplers_ms: Optional[float] = None
    samplers_tokens: Optional[int] = None
    load_ms: Optional[float] = None
    prompt_eval_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    prompt_ms_per_token: Optional[float] = None
    prompt_tokens_per_sec: Optional[float] = None
    eval_ms: Optional[float] = None
    eval_runs: Optional[int] = None
    eval_ms_per_token: Optional[float] = None
    eval_tokens_per_sec: Optional[float] = None
    total_ms: Optional[float] = None
    total_tokens: Optional[int] = None
    unaccounted_ms: Optional[float] = None
    unaccounted_pct: Optional[float] = None
    graphs_reused: Optional[int] = None


@dataclass
class DeviceInfo:
    arch: Optional[str] = None
    system_info: Optional[str] = None
    n_threads: Optional[int] = None
    buffers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    selected_backend: Optional[str] = None
    xdna_selected: bool = False
    xdna_messages: list[str] = field(default_factory=list)
    graph_compute_lines: list[str] = field(default_factory=list)
    device_info_lines: list[str] = field(default_factory=list)
    supports_op_lines: list[str] = field(default_factory=list)
    sched_split_lines: list[str] = field(default_factory=list)
    sched_node_lines: list[str] = field(default_factory=list)


@dataclass
class RunPaths:
    """Layout of one ``runs/{id}/`` directory."""
    run_id: str
    run_dir: Path
    raw_dir: Path
    processed_dir: Path
    stdout_path: Path
    stderr_path: Path
    llama_log_path: Path
    command_path: Path
    env_path: Path
    summary_path: Path
    layer_trace_path: Path
    result_json_path: Path
    meta_path: Path


@dataclass
class RunResult:
    prompt: str
    response: str
    thinking: Optional[str]
    wall_sec: float
    exit_code: int
    command: list[str]
    perf: PerfMetrics
    device: DeviceInfo
    log_path: Optional[str]
    raw_stdout: str
    raw_stderr: str  # process stderr only
    combined_log: str = ""  # preferred parse source (llama.log and/or stderr)
    layer_trace: list["LayerStep"] = field(default_factory=list)
    run_paths: Optional[RunPaths] = None


_PERF_LINE = re.compile(
    r"common_perf_print:\s*(?P<label>.+?)\s*=\s*(?P<body>.+)$"
)
_NUM = re.compile(r"([-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)")
_ARCH = re.compile(r"print_info:\s*arch\s*=\s*(\S+)")
_SYS = re.compile(r"system_info:\s*(.+)$")
_THREADS = re.compile(r"n_threads\s*=\s*(\d+)")
_BUFFER = re.compile(
    r"(?:print_backend_buffers_info|llama_context|llama_kv_cache|"
    r"llama_memory_recurrent|sched_reserve):\s*(.+)$"
)
_DEVICE_INFO_LINE = re.compile(r"^\s*-\s+(\S+)\s*:\s*(.+)$")
_XDNA_MSG = re.compile(r"(ggml_backend_xdna\w*|XDNA)", re.IGNORECASE)
# Scheduler assignment dump (GGML_SCHED_DEBUG>=1, needs -lv 5). Format from
# ggml_backend_sched_print_assignments in ggml/src/ggml-backend.cpp:
#   ## SPLIT #<n>: <backend> # <k> inputs
#   node #<i> (<op>): <name> (<size>) [<backend> <cause>] use=<u>,c=<c>:
_SCHED_SPLIT = re.compile(r"##\s*SPLIT\s*#(\d+):\s*(\S+)\s*#\s*(\d+)\s*inputs")
_SCHED_NODE = re.compile(
    r"node\s*#\s*(\d+)\s*\((.*?)\):\s*(.*?)\s*\(\s*(\S+)\s*\)\s*\[(.*?)\]"
    r"(?:\s*use=(\d+),c=(\d+))?"
)
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ASSISTANT = re.compile(
    r"(?:^|\n)assistant\s*\n(?P<body>.*)$",
    re.DOTALL | re.IGNORECASE,
)


def _f(nums: list[float], idx: int = 0) -> Optional[float]:
    return nums[idx] if len(nums) > idx else None


def _i(nums: list[float], idx: int = 0) -> Optional[int]:
    v = _f(nums, idx)
    return int(v) if v is not None else None


def parse_perf(stderr: str) -> PerfMetrics:
    perf = PerfMetrics()
    for raw in stderr.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw).strip()
        m = _PERF_LINE.search(line)
        if not m:
            continue
        label = m.group("label").strip()
        body = m.group("body").strip()
        nums = [float(x) for x in _NUM.findall(body)]
        if label == "sampling time":
            perf.sampling_ms = _f(nums)
        elif label == "samplers time":
            perf.samplers_ms = _f(nums, 0)
            perf.samplers_tokens = _i(nums, 1)
        elif label == "load time":
            perf.load_ms = _f(nums)
        elif label == "prompt eval time":
            perf.prompt_eval_ms = _f(nums, 0)
            perf.prompt_tokens = _i(nums, 1)
            perf.prompt_ms_per_token = _f(nums, 2)
            perf.prompt_tokens_per_sec = _f(nums, 3)
        elif label == "eval time":
            perf.eval_ms = _f(nums, 0)
            perf.eval_runs = _i(nums, 1)
            perf.eval_ms_per_token = _f(nums, 2)
            perf.eval_tokens_per_sec = _f(nums, 3)
        elif label == "total time":
            perf.total_ms = _f(nums, 0)
            perf.total_tokens = _i(nums, 1)
        elif label == "unaccounted time":
            perf.unaccounted_ms = _f(nums, 0)
            perf.unaccounted_pct = _f(nums, 1)
        elif label == "graphs reused":
            perf.graphs_reused = _i(nums, 0)
    return perf


def parse_device(stderr: str, requested_device: Optional[str] = None) -> DeviceInfo:
    info = DeviceInfo()
    in_device_info = False
    for raw in stderr.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw).rstrip()
        stripped = line.strip()
        if "warning:" in stripped.lower():
            info.warnings.append(stripped)
        if "device_info:" in stripped:
            in_device_info = True
            continue
        if in_device_info:
            dm = _DEVICE_INFO_LINE.match(line)
            if dm:
                info.device_info_lines.append(f"{dm.group(1)}: {dm.group(2).strip()}")
                continue
            if stripped and not stripped.startswith("-"):
                in_device_info = False
        if "xdna" in stripped.lower() and (
            "selected backend" in stripped.lower()
            or "initializing xdna" in stripped.lower()
            or "ggml_backend_xdna" in stripped.lower()
        ):
            if stripped not in info.xdna_messages:
                info.xdna_messages.append(stripped)
            info.xdna_selected = True
        if "ggml-xdna:" in stripped or "supports_op(" in stripped:
            if "supports_op" in stripped or "supports_op stats" in stripped or "graph scheduling debug" in stripped:
                if stripped not in info.supports_op_lines:
                    info.supports_op_lines.append(stripped)
                info.xdna_selected = True
            if "graph_compute" in stripped:
                if len(info.graph_compute_lines) < 24 and stripped not in info.graph_compute_lines:
                    info.graph_compute_lines.append(stripped)
                info.xdna_selected = True
        if stripped.startswith("## SPLIT") or "SPLIT #" in stripped:
            if stripped not in info.sched_split_lines:
                info.sched_split_lines.append(stripped)
        if stripped.startswith("node #") and ("[" in stripped):
            # Keep a small sample of scheduler node→backend assignments
            if len(info.sched_node_lines) < 40 and stripped not in info.sched_node_lines:
                info.sched_node_lines.append(stripped)
        m = _ARCH.search(line)
        if m:
            info.arch = m.group(1)
        m = _SYS.search(line)
        if m:
            info.system_info = m.group(1).strip()
            tm = _THREADS.search(info.system_info)
            if tm:
                info.n_threads = int(tm.group(1))
        m = _BUFFER.search(line)
        if m and ("buffer" in m.group(1).lower() or "CPU" in m.group(1) or "Metal" in m.group(1) or "XDNA" in m.group(1)):
            info.buffers.append(m.group(1).strip())

    if requested_device:
        info.selected_backend = requested_device
        if requested_device.upper() == "XDNA":
            info.xdna_selected = True
    elif info.xdna_selected:
        info.selected_backend = "XDNA"
    elif any("XDNA" in x for x in info.device_info_lines):
        info.selected_backend = "XDNA"
        info.xdna_selected = True
    return info


@dataclass
class LayerStep:
    idx: int
    op: str
    name: str
    size: str
    backend: str
    cause: str
    compute: bool
    split: int
    split_backend: str


def parse_layer_trace(text: str) -> list[LayerStep]:
    """Extract one graph's ordered op→backend assignments.

    The scheduler reprints the assignment on every (re)split, so several
    identical blocks can appear. We keep the *first* complete block (node #0
    up to the last node before node #0 shows up again) — that is the prompt
    graph and contains every layer in execution order.
    """
    steps: list[LayerStep] = []
    cur_split = -1
    cur_split_backend = "?"
    started = False
    for raw in text.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw).strip()

        sm = _SCHED_SPLIT.search(line)
        if sm:
            cur_split = int(sm.group(1))
            cur_split_backend = sm.group(2)
            continue

        nm = _SCHED_NODE.search(line)
        if not nm:
            continue

        idx = int(nm.group(1))
        # A second node #0 means the next graph eval started — stop after the
        # first full block so the trace is a single, complete layer sequence.
        if idx == 0 and started and steps:
            break
        started = True

        bracket = nm.group(5).split()
        backend = bracket[0] if bracket else "?"
        cause = " ".join(bracket[1:]) if len(bracket) > 1 else ""
        compute = nm.group(7) is None or nm.group(7) == "1"

        steps.append(LayerStep(
            idx=idx,
            op=nm.group(2).strip(),
            name=nm.group(3).strip(),
            size=nm.group(4).strip(),
            backend=backend,
            cause=cause,
            compute=compute,
            split=cur_split,
            split_backend=cur_split_backend,
        ))
    return steps


def write_layer_trace(path: Path, result: "RunResult") -> int:
    """Write the per-node execution trace (op → device) to ``path``.

    Returns the number of compute nodes written. One line per node in graph
    order: index, op, tensor name, size, and the backend that ran it.
    """
    steps = result.layer_trace
    counts: dict[str, int] = {}
    for s in steps:
        if s.compute:
            counts[s.backend] = counts.get(s.backend, 0) + 1

    lines: list[str] = []
    lines.append(f"# layer/op execution trace (op -> device), graph order")
    lines.append(f"# prompt: {result.prompt!r}")
    lines.append(f"# command: {' '.join(result.command)}")
    if counts:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.append(f"# compute nodes by backend: {summary}")
    lines.append(f"# total nodes: {len(steps)}")
    lines.append("#")
    lines.append(f"# {'idx':>4}  {'split':>5}  {'op':<12}  {'device':<6}  {'compute':<7}  {'size':>7}  name  [cause]")

    last_split = None
    for s in steps:
        if s.split != last_split:
            lines.append(f"# --- SPLIT #{s.split} -> {s.split_backend} ---")
            last_split = s.split
        flag = "run" if s.compute else "-"
        cause = f"  [{s.cause}]" if s.cause else ""
        lines.append(
            f"  {s.idx:>4}  {s.split:>5}  {s.op:<12}  {s.backend:<6}  {flag:<7}  {s.size:>7}  {s.name}{cause}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sum(counts.values())


def extract_response(stdout: str) -> tuple[str, Optional[str]]:
    text = stdout.strip()
    # Drop ggml/llama debug lines that sometimes leak onto stdout at high -lv.
    cleaned: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("ggml-xdna:") or s.startswith("ggml_") or s.startswith("llama_"):
            continue
        if s.startswith("common_perf_print:") or s.startswith("## SPLIT"):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned).strip()

    m = _ASSISTANT.search(text)
    body = m.group("body").strip() if m else text

    thinking: Optional[str] = None
    tm = _THINK.search(body)
    if tm:
        thinking = tm.group(1).strip() or None
        body = _THINK.sub("", body).strip()
    else:
        # Truncated generation often leaves an unclosed <think>… block.
        open_m = re.search(r"<think>(.*)$", body, re.DOTALL | re.IGNORECASE)
        if open_m:
            thinking = open_m.group(1).strip() or None
            body = body[: open_m.start()].strip()

    # Drop trailing interactive markers if any
    for marker in ("> EOF by user", "Exiting...", "[end of text]"):
        if marker in body:
            body = body.split(marker, 1)[0].strip()

    return body, thinking


def _xdna_mode_summary(d: DeviceInfo) -> str:
    """Human-readable XDNA mode from init / graph_compute lines."""
    init = next((m for m in d.xdna_messages if "XDNA backend init" in m), "")
    bits: list[str] = []
    if "ADD=NPU" in init:
        bits.append("ADD on NPU")
    elif "ADD=host" in init:
        bits.append("ADD on host")
    if "MUL_MAT=NPU" in init:
        bits.append("MUL_MAT on NPU")
    elif "GEMM_NPU=0" in init or "MUL_MAT=host" in init:
        bits.append("MUL_MAT on CPU backend (not claimed)")
    if any("on=NPU" in m and "MUL_MAT" in m for m in d.graph_compute_lines):
        bits.append("saw NPU MUL_MAT runs")
    if any("on=NPU" in m and "ADD" in m for m in d.graph_compute_lines):
        bits.append("saw NPU ADD runs")
    if bits:
        return "; ".join(bits)
    if d.xdna_selected:
        return "XDNA device selected (see init / supports_op)"
    return "n/a"


def make_run_id(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def allocate_run_dir(runs_dir: Path, run_id: str) -> Path:
    """Create ``runs_dir/run_id``, appending -2, -3, … on collision."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    candidate = runs_dir / run_id
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate
    n = 2
    while True:
        alt = runs_dir / f"{run_id}-{n}"
        if not alt.exists():
            alt.mkdir(parents=True)
            return alt
        n += 1


def prepare_run_paths(runs_dir: Path, run_id: Optional[str] = None) -> RunPaths:
    rid = make_run_id(run_id)
    run_dir = allocate_run_dir(runs_dir, rid)
    raw_dir = run_dir / "raw"
    processed_dir = run_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_id=run_dir.name,
        run_dir=run_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        stdout_path=raw_dir / "stdout.txt",
        stderr_path=raw_dir / "stderr.txt",
        llama_log_path=raw_dir / "llama.log",
        command_path=raw_dir / "command.txt",
        env_path=raw_dir / "env.txt",
        summary_path=processed_dir / "summary.txt",
        layer_trace_path=processed_dir / "layer-trace.txt",
        result_json_path=processed_dir / "result.json",
        meta_path=run_dir / "meta.json",
    )


def write_raw_artifacts(
    paths: RunPaths,
    cmd: list[str],
    env: dict[str, str],
    stdout: str,
    stderr: str,
) -> None:
    paths.command_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")
    interesting = sorted(
        k for k in env
        if k.startswith("GGML_") or k.startswith("XDNA") or k in ("XILINX_XRT", "LD_LIBRARY_PATH")
    )
    env_lines = [f"{k}={env.get(k, '')}" for k in interesting]
    paths.env_path.write_text("\n".join(env_lines) + ("\n" if env_lines else ""), encoding="utf-8")
    paths.stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
    paths.stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
    # llama.log is filled by llama-completion via --log-file; leave a stub if empty.
    if not paths.llama_log_path.exists():
        paths.llama_log_path.write_text("", encoding="utf-8")


def write_processed_artifacts(
    paths: RunPaths,
    result: RunResult,
    summary_text: str,
    layers_written: int,
) -> None:
    paths.summary_path.write_text(summary_text, encoding="utf-8")
    if result.layer_trace:
        write_layer_trace(paths.layer_trace_path, result)
    else:
        paths.layer_trace_path.write_text(
            "# no scheduler assignment lines captured — need GGML_SCHED_DEBUG=2 and -lv >= 5\n",
            encoding="utf-8",
        )
    payload = {
        "run_id": paths.run_id,
        "prompt": result.prompt,
        "response": result.response,
        "thinking": result.thinking,
        "wall_sec": result.wall_sec,
        "exit_code": result.exit_code,
        "command": result.command,
        "perf": asdict(result.perf),
        "device": asdict(result.device),
        "layer_trace": [asdict(s) for s in result.layer_trace],
        "layer_trace_compute_nodes": layers_written,
    }
    paths.result_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    meta = {
        "run_id": paths.run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "exit_code": result.exit_code,
        "prompt": result.prompt,
        "raw": {
            "stdout": str(paths.stdout_path.relative_to(paths.run_dir)),
            "stderr": str(paths.stderr_path.relative_to(paths.run_dir)),
            "llama_log": str(paths.llama_log_path.relative_to(paths.run_dir)),
            "command": str(paths.command_path.relative_to(paths.run_dir)),
            "env": str(paths.env_path.relative_to(paths.run_dir)),
        },
        "processed": {
            "summary": str(paths.summary_path.relative_to(paths.run_dir)),
            "layer_trace": str(paths.layer_trace_path.relative_to(paths.run_dir)),
            "result_json": str(paths.result_json_path.relative_to(paths.run_dir)),
        },
    }
    paths.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def default_threads() -> int:
    # Prefer portable os.cpu_count(); fall back to platform helpers.
    n = os.cpu_count()
    if n and n > 0:
        return n
    try:
        out = subprocess.check_output(["nproc"], text=True).strip()
        return max(1, int(out))
    except (OSError, ValueError, subprocess.CalledProcessError):
        pass
    try:
        # macOS
        out = subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True).strip()
        return max(1, int(out))
    except (OSError, ValueError, subprocess.CalledProcessError):
        return 4


def build_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.bin),
        "-m", str(args.model),
        "-ngl", str(args.ngl),
        "-c", str(args.ctx),
        "-n", str(args.n_predict),
        "-t", str(args.threads),
        "-lv", str(args.verbosity),
        "--perf",
        "--single-turn",
        "-fit", "off",
        "-p", args.prompt,
    ]
    if args.device:
        cmd.extend(["-dev", args.device])
    if args.no_warmup:
        cmd.append("--no-warmup")
    if args.log_file:
        cmd.extend(["--log-file", str(args.log_file)])
    if args.extra:
        cmd.extend(args.extra)
    return cmd


def run_query(args: argparse.Namespace) -> RunResult:
    if not Path(args.bin).is_file():
        raise FileNotFoundError(f"Binary not found: {args.bin}")
    if not Path(args.model).is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    run_paths: Optional[RunPaths] = None
    if args.runs_dir is not None:
        run_paths = prepare_run_paths(Path(args.runs_dir), args.run_id)
        print(f"run_llm_query: run dir {run_paths.run_dir}", file=sys.stderr, flush=True)

    if args.log_file is not None:
        log_path = Path(args.log_file)
    elif run_paths is not None:
        log_path = run_paths.llama_log_path
    else:
        log_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"llm-query-{os.getpid()}.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Point llama --log-file at the raw llama log before building the command.
    args.log_file = log_path

    cmd = build_cmd(args)
    if "--log-file" not in cmd:
        cmd.extend(["--log-file", str(log_path)])

    if args.device:
        print(
            f"run_llm_query: selecting backend device '{args.device}'",
            file=sys.stderr,
            flush=True,
        )

    env = os.environ.copy()
    if args.xdna_debug is not None:
        env["GGML_XDNA_DEBUG"] = str(args.xdna_debug)
    elif args.device and str(args.device).upper() == "XDNA":
        env.setdefault("GGML_XDNA_DEBUG", "1")
    if args.sched_debug is not None:
        env["GGML_SCHED_DEBUG"] = str(args.sched_debug)

    print(
        f"run_llm_query: GGML_XDNA_DEBUG={env.get('GGML_XDNA_DEBUG', '0')} "
        f"GGML_SCHED_DEBUG={env.get('GGML_SCHED_DEBUG', '0')}",
        file=sys.stderr,
        flush=True,
    )

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    wall = time.perf_counter() - t0

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    log_text = ""
    try:
        log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass

    if run_paths is not None:
        write_raw_artifacts(run_paths, cmd, env, stdout, stderr)
        # Ensure llama.log exists even if the binary wrote elsewhere / nothing.
        if not run_paths.llama_log_path.exists() or run_paths.llama_log_path.stat().st_size == 0:
            if log_text:
                run_paths.llama_log_path.write_text(log_text, encoding="utf-8", errors="replace")
            elif log_path.resolve() != run_paths.llama_log_path.resolve() and Path(log_path).is_file():
                run_paths.llama_log_path.write_text(
                    Path(log_path).read_text(encoding="utf-8", errors="replace"),
                    encoding="utf-8",
                    errors="replace",
                )

    # Prefer the log file when it has the verbose load dump; else stderr.
    if "common_perf_print" in log_text or "print_backend_buffers_info" in log_text:
        combined = log_text
        if stderr and "warning:" in stderr.lower() and "warning:" not in log_text.lower():
            combined = stderr + "\n" + log_text
    else:
        combined = stderr or log_text

    device = parse_device(combined, requested_device=args.device)
    seen: set[str] = set()
    unique_buffers: list[str] = []
    for b in device.buffers:
        if b not in seen:
            seen.add(b)
            unique_buffers.append(b)
    device.buffers = unique_buffers

    response, thinking = extract_response(stdout)
    return RunResult(
        prompt=args.prompt,
        response=response,
        thinking=thinking,
        wall_sec=wall,
        exit_code=proc.returncode,
        command=cmd,
        perf=parse_perf(combined),
        device=device,
        log_path=str(log_path),
        raw_stdout=stdout,
        raw_stderr=stderr,
        combined_log=combined,
        layer_trace=parse_layer_trace(combined),
        run_paths=run_paths,
    )


def format_human(result: RunResult, show_thinking: bool, show_raw: bool) -> str:
    p = result.perf
    d = result.device
    out: list[str] = []

    def w(line: str = "") -> None:
        out.append(line)

    w("=" * 72)
    w("BACKEND")
    w("=" * 72)
    if d.xdna_selected or (d.selected_backend and d.selected_backend.upper() == "XDNA"):
        w("selected:    XDNA")
        w(f"mode:        {_xdna_mode_summary(d)}")
    else:
        w(f"selected:    {d.selected_backend or 'default / CPU'}")
    if d.xdna_messages:
        w("xdna logs:")
        for msg in d.xdna_messages:
            w(f"  - {msg}")
    elif d.xdna_selected:
        w("xdna logs:   (requested via -dev XDNA; see device_info below)")
    if d.supports_op_lines:
        w("supports_op:")
        uniq = [m for m in d.supports_op_lines if "supports_op(" in m or "graph scheduling debug" in m]
        stats = [m for m in d.supports_op_lines if "supports_op stats" in m]
        for msg in uniq:
            w(f"  - {msg}")
        if stats:
            w(f"  - {stats[-1]}")
        w(f"  ({len(uniq)} lines; set --xdna-debug 2 for denser stats)")
    if d.graph_compute_lines:
        w("graph_compute sample (proof XDNA ran the op):")
        for msg in d.graph_compute_lines[:12]:
            w(f"  - {msg}")
        if len(d.graph_compute_lines) > 12:
            w(f"  - ... ({len(d.graph_compute_lines)} unique lines captured)")
    if d.sched_split_lines or d.sched_node_lines:
        w("scheduler graph splits:")
        for msg in d.sched_split_lines[:20]:
            w(f"  - {msg}")
        if d.sched_node_lines:
            w("scheduler node sample (op → backend):")
            for msg in d.sched_node_lines[:20]:
                w(f"  - {msg}")
            if len(d.sched_node_lines) >= 40:
                w("  - ... (truncated; full dump is in raw/llama.log)")
    if d.device_info_lines:
        w("device_info:")
        for line in d.device_info_lines:
            w(f"  - {line}")
    w()

    w("=" * 72)
    w("PROMPT")
    w("=" * 72)
    w(result.prompt)
    w()

    show_think = show_thinking or (bool(result.thinking) and not result.response)
    if show_think and result.thinking:
        w("=" * 72)
        w("THINKING" + ("" if result.response else " (truncated — raise -n for a final answer)"))
        w("=" * 72)
        w(result.thinking)
        w()

    w("=" * 72)
    w("RESPONSE")
    w("=" * 72)
    if result.response:
        w(result.response)
    elif result.thinking:
        w("(empty — all -n tokens went into <think>; try -n 128 or --thinking)")
    else:
        w("(empty)")
    w()

    w("=" * 72)
    w("DEVICE / BUFFERS")
    w("=" * 72)
    w(f"arch:        {d.arch or 'n/a'}")
    w(f"n_threads:   {d.n_threads or 'n/a'}")
    w(f"system_info: {d.system_info or 'n/a'}")
    if d.buffers:
        w("buffers:")
        for b in d.buffers:
            w(f"  - {b}")
    else:
        w("buffers:     (none parsed — try -lv 4)")
    if d.warnings:
        w("warnings:")
        for warn in d.warnings[:8]:
            w(f"  - {warn}")
    w()

    w("=" * 72)
    w("PERFORMANCE")
    w("=" * 72)
    w(f"wall clock:           {result.wall_sec * 1000:.2f} ms")
    w(f"load time:            {fmt_ms(p.load_ms)}")
    w(f"prompt eval:          {fmt_ms(p.prompt_eval_ms)} / {fmt_int(p.prompt_tokens)} tok"
      f"  ({fmt_ms(p.prompt_ms_per_token)}/tok, {fmt_tps(p.prompt_tokens_per_sec)})")
    w(f"generation (eval):    {fmt_ms(p.eval_ms)} / {fmt_int(p.eval_runs)} tok"
      f"  ({fmt_ms(p.eval_ms_per_token)}/tok, {fmt_tps(p.eval_tokens_per_sec)})")
    w(f"sampling:             {fmt_ms(p.sampling_ms)}")
    w(f"total (libllama):     {fmt_ms(p.total_ms)} / {fmt_int(p.total_tokens)} tok")
    w(f"unaccounted:          {fmt_ms(p.unaccounted_ms)} ({fmt_pct(p.unaccounted_pct)})")
    w(f"graphs reused:        {fmt_int(p.graphs_reused)}")
    w(f"exit code:            {result.exit_code}")
    if result.run_paths is not None:
        w(f"run dir:              {result.run_paths.run_dir}")
        w(f"raw llama log:        {result.run_paths.llama_log_path}")
    else:
        w(f"log file:             {result.log_path}")
    w()

    if show_raw:
        w("=" * 72)
        w("RAW COMBINED LOG (tail)")
        w("=" * 72)
        lines = (result.combined_log or result.raw_stderr).strip().splitlines()
        w("\n".join(lines[-40:]))

    return "\n".join(out) + "\n"


def print_human(result: RunResult, show_thinking: bool, show_raw: bool) -> str:
    text = format_human(result, show_thinking=show_thinking, show_raw=show_raw)
    sys.stdout.write(text)
    return text


def fmt_ms(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.2f} ms"


def fmt_tps(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    if v != v or v == float("inf"):  # NaN or inf
        return "n/a"
    return f"{v:.2f} t/s"


def fmt_pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.1f} %"


def fmt_int(v: Optional[int]) -> str:
    return "n/a" if v is None else str(v)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run local LLM via llama-completion and print answer + metrics.",
    )
    p.add_argument("prompt", nargs="?", help="User prompt / question")
    p.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL, help="GGUF model path")
    p.add_argument("--bin", type=Path, default=DEFAULT_BIN, help="llama-completion binary")
    p.add_argument("-n", "--n-predict", type=int, default=128, help="Max tokens to generate")
    p.add_argument("-c", "--ctx", type=int, default=2048, help="Context size")
    p.add_argument("-t", "--threads", type=int, default=0, help="CPU threads (0 = auto)")
    p.add_argument("-ngl", type=int, default=0, help="GPU layers (0 = keep model weights on CPU)")
    p.add_argument("-dev", "--device", default="XDNA",
                   help="Offload device to select (default: XDNA). Use 'none' to disable.")
    p.add_argument("--xdna-debug", type=int, default=None, metavar="N",
                   help="Set GGML_XDNA_DEBUG (default 1 when -dev XDNA). "
                        "1=unique supports_op fallbacks, 2=verbose per-call")
    p.add_argument("--sched-debug", type=int, default=2, metavar="N",
                   help="Set GGML_SCHED_DEBUG (default 2). "
                        "1=graph split lines only, 2=per-node op->backend assignment "
                        "(needed for the layer-trace file). Requires -lv >= 5 (auto-raised when >0).")
    p.add_argument("-lv", "--verbosity", type=int, default=4, help="Log verbosity (3=info, 4=trace, 5=debug)")
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
                   help=f"Root directory for per-run artifacts (default: {DEFAULT_RUNS_DIR})")
    p.add_argument("--no-runs-dir", dest="runs_dir", action="store_const", const=None,
                   help="Do not create a runs/{{id}} directory")
    p.add_argument("--run-id", default=None,
                   help="Run id under --runs-dir (default: YYYYMMDD-HHMMSS)")
    p.add_argument("--log-file", type=Path, default=None,
                   help="Override path for raw llama --log-file "
                        "(default: runs/{{id}}/raw/llama.log)")
    p.add_argument("--layers-file", type=Path, default=None, metavar="PATH",
                   help="Also write layer-trace to this path "
                        "(always written to runs/{{id}}/processed/layer-trace.txt when --runs-dir is set)")
    p.add_argument("--no-layers-file", dest="layers_file", action="store_const", const=False,
                   help="Do not write an extra layer-trace copy outside the run dir")
    p.add_argument("--warmup", dest="no_warmup", action="store_false", default=True,
                   help="Enable model warmup (default: disabled)")
    p.add_argument("--thinking", action="store_true", help="Print model thinking block")
    p.add_argument("--raw", action="store_true", help="Print tail of raw logs")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON to stdout")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra args after -- passed to llama-completion")
    args = p.parse_args(argv)
    if not args.prompt:
        p.error("prompt is required")
    if args.threads <= 0:
        args.threads = default_threads()
    if args.device and args.device.lower() == "none":
        args.device = "none"
    # GGML_LOG_DEBUG (scheduler splits/nodes) maps to common LOG_LEVEL_DEBUG=5
    if args.sched_debug and args.sched_debug > 0 and args.verbosity < 5:
        args.verbosity = 5
    # argparse.REMAINDER keeps leading '--' if present
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    # layers_file: None → default copy to cwd only if no runs dir;
    # False → disabled; Path → explicit extra copy.
    if args.layers_file is None and args.runs_dir is None:
        args.layers_file = REPO_ROOT / "layer-trace.txt"
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = run_query(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    layers_written = 0
    if result.layer_trace:
        layers_written = sum(1 for s in result.layer_trace if s.compute)

    # Extra layer-trace copy outside the run dir (legacy / convenience).
    if args.layers_file not in (None, False) and isinstance(args.layers_file, Path):
        if result.layer_trace:
            write_layer_trace(args.layers_file, result)
        else:
            args.layers_file.write_text(
                "# no scheduler assignment lines captured — need GGML_SCHED_DEBUG=2 and -lv >= 5\n",
                encoding="utf-8",
            )

    summary_text = format_human(result, show_thinking=args.thinking, show_raw=args.raw)

    if result.run_paths is not None:
        write_processed_artifacts(
            result.run_paths, result, summary_text, layers_written=layers_written,
        )

    if args.json:
        payload: dict[str, Any] = {
            "prompt": result.prompt,
            "response": result.response,
            "thinking": result.thinking,
            "wall_sec": result.wall_sec,
            "exit_code": result.exit_code,
            "command": result.command,
            "log_path": result.log_path,
            "run_dir": str(result.run_paths.run_dir) if result.run_paths else None,
            "run_id": result.run_paths.run_id if result.run_paths else None,
            "perf": asdict(result.perf),
            "device": asdict(result.device),
            "layer_trace": [asdict(s) for s in result.layer_trace],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        sys.stdout.write(summary_text)
        if result.run_paths is not None:
            print("=" * 72)
            print("RUN ARTIFACTS")
            print("=" * 72)
            print(f"run dir:     {result.run_paths.run_dir}")
            print(f"raw:         {result.run_paths.raw_dir}")
            print(f"processed:   {result.run_paths.processed_dir}")
            print(f"layer-trace: {result.run_paths.layer_trace_path} "
                  f"({len(result.layer_trace)} nodes, {layers_written} compute)")
            print()

    return 0 if result.exit_code == 0 else result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
