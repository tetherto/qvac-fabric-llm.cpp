#!/usr/bin/env python3
"""
KV cache quality benchmark orchestrator using RULER tasks.

Schedules ruler-bench.sh jobs across multiple GPUs in parallel,
collects CSV results, and produces aggregated comparison tables.

Usage:
    python3 tests/test-kv-cache-ruler.py -c smoke --gpus 0,1 --output-dir results/
    python3 tests/test-kv-cache-ruler.py -c small --gpus 0,1
    python3 tests/test-kv-cache-ruler.py -c large --gpus 0,1 --dry-run
    python3 tests/test-kv-cache-ruler.py -c large --gpus 0,1 --output-dir results/

Config presets:
    smoke  - 1 sample,  all ctx (4096 6144 16384), subset quants   (quick sanity)
    small  - 20 samples, 4096 only,                subset quants   (fast iteration)
    mid    - 25 samples, 4096 6144,                subset quants   (balanced)
    large  - 30 samples, 4096 6144 16384,          all quants      (full study)
"""

import argparse
import csv
import logging
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
RULER_BENCH = SCRIPT_DIR / "ruler-bench.sh"

# ── Bits-per-weight for KV cache types ──────────────────────────────────────

BPW = {
    "f16":    16.0,
    "q8_0":   8.5,
    "q4_0":   4.5,
    "tbq3_0": 4.25,
    "tbq4_0": 5.25,
    "pq3_0":  3.25,
    "pq4_0":  4.25,
}


def bpw_label(k_type: str, v_type: str) -> str:
    k_bpw = BPW.get(k_type, "?")
    v_bpw = BPW.get(v_type, "?")
    return f"K:{k_bpw} V:{v_bpw}"


# ── Quant configurations ───────────────────────────────────────────────────

QUANT_CONFIGS_ALL = [
    ("tbq4_0", "pq4_0"),
    ("tbq3_0", "pq3_0"),
    ("tbq3_0", "q4_0"),
    ("tbq4_0", "q4_0"),
    ("pq4_0",  "pq4_0"),
    ("pq3_0",  "pq3_0"),
    ("q4_0",   "q4_0"),
]

QUANT_CONFIGS_SUBSET = [
    ("tbq4_0", "pq4_0"),
    ("tbq3_0", "pq3_0"),
    ("tbq4_0", "q4_0"),
    ("q4_0",   "q4_0"),
]

# ── Tasks ──────────────────────────────────────────────────────────────────

TASKS_MAIN = "niah_single_1 niah_mk_k8q4_noise vt"
TASKS_V2   = "niah_mk_k8q4v2_noise"

# ── Model definitions ─────────────────────────────────────────────────────


@dataclass
class ModelDef:
    path: str
    tokenizer: str
    label: str
    family: str  # "7B-q4", "7B-f16", "14B-q4", etc.


MODELS_DEFAULT = [
    ModelDef(
        path="models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        tokenizer="mistralai/Mistral-7B-Instruct-v0.3",
        label="Mistral-7B-Q4",
        family="7B-q4",
    ),
    ModelDef(
        path="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        tokenizer="NousResearch/Meta-Llama-3.1-8B-Instruct",
        label="Llama-8B-Q4",
        family="7B-q4",
    ),
    ModelDef(
        path="models/Mistral-7B-Instruct-v0.3.fp16.gguf",
        tokenizer="mistralai/Mistral-7B-Instruct-v0.3",
        label="Mistral-7B-F16",
        family="7B-f16",
    ),
    ModelDef(
        path="models/Llama-3.1-8B-Instruct-f16.gguf",
        tokenizer="NousResearch/Meta-Llama-3.1-8B-Instruct",
        label="Llama-8B-F16",
        family="7B-f16",
    ),
    ModelDef(
        path="models/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        tokenizer="Qwen/Qwen2.5-14B-Instruct",
        label="Qwen-14B-Q4",
        family="14B-q4",
    ),
]

# ── Config presets ─────────────────────────────────────────────────────────


@dataclass
class ConfigPreset:
    name: str
    num_samples: int
    ctx_lengths: str
    quant_configs: list
    tasks_main: str = TASKS_MAIN
    tasks_v2: str = TASKS_V2


PRESETS = {
    "smoke": ConfigPreset("smoke", 1,  "4096 6144 16384",   QUANT_CONFIGS_SUBSET),
    "small": ConfigPreset("small", 20, "4096",            QUANT_CONFIGS_SUBSET),
    "mid":   ConfigPreset("mid",   25, "4096 6144",       QUANT_CONFIGS_SUBSET),
    "large": ConfigPreset("large", 30, "4096 6144 16384", QUANT_CONFIGS_ALL),
}

# ── Job definition ─────────────────────────────────────────────────────────


@dataclass
class Job:
    model: ModelDef
    k_type: str
    v_type: str
    tasks: str
    ctx_lengths: str
    num_samples: int
    gpu_id: int = 0
    csv_file: str = ""
    tag: str = ""  # "main" or "v2"


# ── Run a single ruler-bench job ───────────────────────────────────────────

_print_lock = threading.Lock()


class ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    def tick(self) -> str:
        with self.lock:
            self.completed += 1
            elapsed = time.time() - self.start_time
            if self.completed > 0:
                avg = elapsed / self.completed
                remaining = self.total - self.completed
                eta_secs = avg * remaining
                eta_str = _fmt_duration(eta_secs)
            else:
                eta_str = "?"
            elapsed_str = _fmt_duration(elapsed)
            return f"[{self.completed}/{self.total}] elapsed {elapsed_str}, ETA {eta_str}"


def _fmt_duration(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


_progress: Optional[ProgressTracker] = None


def run_job(job: Job, extra_args: list[str], output_dir: Path) -> Optional[str]:
    jobs_dir = output_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    job_name = f"{job.model.label}_{job.k_type}_{job.v_type}_{job.tag}"
    job_name = job_name.replace("/", "_").replace(" ", "_")
    csv_path = str(jobs_dir / f"{job_name}.csv")
    job.csv_file = csv_path

    env = os.environ.copy()
    env["GGML_VK_VISIBLE_DEVICES"] = str(job.gpu_id)
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu_id)

    cmd = [
        "bash", str(RULER_BENCH),
        "-m", job.model.path,
        "-ctk", job.k_type,
        "-ctv", job.v_type,
        "--tasks", job.tasks,
        "--ctx-lengths", job.ctx_lengths,
        "--num-samples", str(job.num_samples),
        "--tokenizer", job.model.tokenizer,
        "--csv", csv_path,
    ]
    cmd.extend(extra_args)

    label = f"[GPU{job.gpu_id}] {job.model.label} K={job.k_type} V={job.v_type} ({job.tag})"
    cmd_str = " ".join(cmd)
    log_path = csv_path.replace(".csv", ".txt")
    with _print_lock:
        logger.info("  START: %s", label)
        logger.info("  $ %s", cmd_str)
        logger.info("  log: %s", log_path)
        logger.info("")

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=7200
        )
        if result.returncode != 0:
            with _print_lock:
                logger.info("  FAIL:  %s", label)
                logger.info("         stderr: %s", result.stderr[-500:])
            return None
    except subprocess.TimeoutExpired:
        with _print_lock:
            logger.info("  TIMEOUT: %s", label)
        return None

    # Extract overall accuracy from output
    accuracy = "?"
    for line in result.stdout.splitlines():
        if "Overall accuracy:" in line:
            accuracy = line.strip().split("Overall accuracy:")[1].strip()
            break

    progress_str = _progress.tick() if _progress else ""
    with _print_lock:
        logger.info("  DONE:  %s  =>  %s  %s", label, accuracy, progress_str)
    return csv_path


# ── Collect all CSV results ────────────────────────────────────────────────


@dataclass
class CellResult:
    model: str
    cache_k: str
    cache_v: str
    task: str
    ctx_len: int
    samples: int
    mean_pct: float
    stdev_pct: float


def collect_results(csv_paths: list[str]) -> list[CellResult]:
    results = []
    for path in csv_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    results.append(CellResult(
                        model=row["model"],
                        cache_k=row["cache_k"],
                        cache_v=row["cache_v"],
                        task=row["task"],
                        ctx_len=int(row["ctx_len"]),
                        samples=int(row["samples"]),
                        mean_pct=float(row["mean_pct"]),
                        stdev_pct=float(row["stdev_pct"]),
                    ))
                except (KeyError, ValueError) as e:
                    logger.warning("  WARN: bad CSV row in %s: %s", path, e)
    return results


# ── Table formatting ───────────────────────────────────────────────────────

TASK_SHORT = {
    "niah_single_1":        "t1",
    "niah_mk_k8q4_noise":   "t2",
    "vt":                   "t3",
    "niah_mk_k8q4v2_noise": "t4",
}

TASK_LEGEND = {
    "t1": "niah_single_1        — Single NIAH, noise haystack",
    "t2": "niah_mk_k8q4_noise   — Multi-key 8 keys, 4 queries, noise haystack",
    "t3": "vt                   — Variable tracking, 1 chain, 4 hops",
    "t4": "niah_mk_k8q4v2_noise — Multi-key 8 keys, 4 queries, 2 values, noise haystack",
}


def format_table(results: list[CellResult], title: str, ctx_lengths: list[int]) -> str:
    if not results:
        return f"\n{title}\n  (no results)\n"

    groups = defaultdict(list)
    for r in results:
        key = (r.model, r.cache_k, r.cache_v, r.task)
        groups[key].append(r)

    models = sorted(set(r.model for r in results))
    quants = sorted(set((r.cache_k, r.cache_v) for r in results))
    tasks = sorted(set(r.task for r in results))

    multi_task = len(tasks) > 1

    # Build column headers: t1@4096, t1@6144, ..., [4096-avg, ...], Score
    header_labels = []
    for task in tasks:
        t_short = TASK_SHORT.get(task, task[:6])
        for ctx in ctx_lengths:
            header_labels.append(f"{t_short}@{ctx}")
    if multi_task:
        for ctx in ctx_lengths:
            header_labels.append(f"{ctx}-avg")
    header_labels.append("Score")

    col_w = 12
    lines = []
    lines.append("")
    lines.append("=" * 120)
    lines.append(f" {title}")
    lines.append("=" * 120)

    for model_name in models:
        lines.append("")
        lines.append(f"  Model: {model_name}")

        header = f"  {'KV Config':<18}"
        for lbl in header_labels:
            header += f"{lbl:>{col_w}}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for k_type, v_type in quants:
            kv_label = f"{k_type}/{v_type}"
            row = f"  {kv_label:<18}"
            all_means = []
            ctx_sums = defaultdict(list)

            for task in tasks:
                for ctx in ctx_lengths:
                    key = (model_name, k_type, v_type, task)
                    cells = [c for c in groups.get(key, []) if c.ctx_len == ctx]
                    if cells:
                        c = cells[0]
                        row += f"{c.mean_pct:5.1f}±{c.stdev_pct:4.1f}%".rjust(col_w)
                        all_means.append(c.mean_pct)
                        ctx_sums[ctx].append(c.mean_pct)
                    else:
                        row += f"{'--':>{col_w}}"

            if multi_task:
                for ctx in ctx_lengths:
                    vals = ctx_sums.get(ctx, [])
                    if vals:
                        row += f"{sum(vals) / len(vals):.1f}%".rjust(col_w)
                    else:
                        row += f"{'--':>{col_w}}"

            if all_means:
                row += f"{sum(all_means) / len(all_means):.1f}%".rjust(col_w)
            else:
                row += f"{'--':>{col_w}}"

            lines.append(row)

    lines.append("")

    # Legend
    used_shorts = set()
    for task in tasks:
        t_short = TASK_SHORT.get(task, task[:6])
        used_shorts.add(t_short)
    lines.append("  Legend:")
    for short, desc in TASK_LEGEND.items():
        if short in used_shorts:
            lines.append(f"    {short} = {desc}")

    lines.append("=" * 120)
    return "\n".join(lines)


def aggregate_results(results: list[CellResult], agg_label: str) -> list[CellResult]:
    """Average results across models, producing one row per (cache_k, cache_v, task, ctx_len)."""
    groups = defaultdict(list)
    for r in results:
        key = (r.cache_k, r.cache_v, r.task, r.ctx_len)
        groups[key].append(r)

    agg = []
    for (ck, cv, task, ctx), cells in groups.items():
        mean = sum(c.mean_pct for c in cells) / len(cells)
        sd = sum(c.stdev_pct for c in cells) / len(cells)
        samples = sum(c.samples for c in cells)
        agg.append(CellResult(
            model=agg_label, cache_k=ck, cache_v=cv,
            task=task, ctx_len=ctx, samples=samples,
            mean_pct=round(mean, 1), stdev_pct=round(sd, 1),
        ))
    return agg


def write_combined_csv(results: list[CellResult], path: str):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "cache_k", "cache_v", "k_bpw", "v_bpw",
            "task", "ctx_len", "samples", "mean_pct", "stdev_pct"
        ])
        for r in results:
            writer.writerow([
                r.model, r.cache_k, r.cache_v,
                BPW.get(r.cache_k, "?"), BPW.get(r.cache_v, "?"),
                r.task, r.ctx_len, r.samples, r.mean_pct, r.stdev_pct,
            ])


# ── Output directory helpers ───────────────────────────────────────────────


def _next_available_dir(base: Path) -> Path:
    """Return base, base1, base2, ... — first that doesn't exist or is empty."""
    if not base.exists() or not any(base.iterdir()):
        return base
    i = 1
    while True:
        candidate = base.parent / f"{base.name}{i}"
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
        i += 1


def _find_latest_dir(base: Path) -> Optional[Path]:
    """Find the latest existing results dir (base, base1, base2, ...)."""
    latest = None
    if base.exists() and any(base.iterdir()):
        latest = base
    i = 1
    while True:
        candidate = base.parent / f"{base.name}{i}"
        if candidate.exists() and any(candidate.iterdir()):
            latest = candidate
            i += 1
        else:
            break
    return latest


def _job_csv_path(output_dir: Path, job: "Job") -> Path:
    job_name = f"{job.model.label}_{job.k_type}_{job.v_type}_{job.tag}"
    job_name = job_name.replace("/", "_").replace(" ", "_")
    return output_dir / "jobs" / f"{job_name}.csv"


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="KV cache RULER benchmark orchestrator")
    parser.add_argument("-c", "--config", default="small", choices=PRESETS.keys(),
                        help="Config preset (default: small)")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs (default: 0)")
    parser.add_argument("--models", nargs="*", help="Model paths to use (overrides defaults)")
    parser.add_argument("--output-dir", default="results", help="Directory for output files")
    parser.add_argument("--skip-v2", action="store_true", help="Skip v2 (hard) tasks")
    parser.add_argument("--skip-main", action="store_true", help="Skip main tasks")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without running")
    parser.add_argument("--rerun-missing", action="store_true",
                        help="Reuse latest output dir, skip jobs with existing CSV, run the rest")
    parser.add_argument("--extra", nargs="*", default=[], help="Extra args for ruler-bench")
    args = parser.parse_args()

    preset = PRESETS[args.config]
    gpu_ids = [int(g) for g in args.gpus.split(",")]

    base_dir = Path(args.output_dir)
    if args.rerun_missing:
        # Find latest existing output dir
        output_dir = _find_latest_dir(base_dir)
        if output_dir is None:
            logger.info("No existing '%s' dir found, creating fresh one.", base_dir)
            output_dir = base_dir
        else:
            logger.info("Rerun-missing: reusing %s", output_dir)
    else:
        output_dir = _next_available_dir(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter models to those that exist
    models = MODELS_DEFAULT
    if args.models:
        models = []
        for p in args.models:
            for m in MODELS_DEFAULT:
                if m.path == p:
                    models.append(m)
                    break
            else:
                models.append(ModelDef(
                    path=p,
                    tokenizer="",  # will auto-detect
                    label=Path(p).stem,
                    family="custom",
                ))

    available_models = [m for m in models if os.path.exists(m.path)]
    if not available_models:
        logger.error("ERROR: No model files found. Expected:")
        for m in models:
            logger.error("  %s", m.path)
        sys.exit(1)

    missing = [m for m in models if not os.path.exists(m.path)]
    if missing:
        logger.warning("WARNING: Skipping missing models:")
        for m in missing:
            logger.warning("  %s", m.path)

    # Build job list
    jobs_main = []
    jobs_v2 = []

    extra_args = ["-ngl", "99", "-fa", "1", "--split-mode", "none"] + args.extra

    for model in available_models:
        for k_type, v_type in preset.quant_configs:
            if not args.skip_main:
                jobs_main.append(Job(
                    model=model, k_type=k_type, v_type=v_type,
                    tasks=preset.tasks_main,
                    ctx_lengths=preset.ctx_lengths,
                    num_samples=preset.num_samples,
                    tag="main",
                ))
            if not args.skip_v2:
                jobs_v2.append(Job(
                    model=model, k_type=k_type, v_type=v_type,
                    tasks=preset.tasks_v2,
                    ctx_lengths=preset.ctx_lengths,
                    num_samples=preset.num_samples,
                    tag="v2",
                ))

    all_jobs = jobs_main + jobs_v2

    # --rerun-missing: skip jobs with existing CSV, collect their paths
    skipped_csvs_main = []
    skipped_csvs_v2 = []
    if args.rerun_missing:
        remaining = []
        for job in all_jobs:
            csv_p = _job_csv_path(output_dir, job)
            if csv_p.exists() and csv_p.stat().st_size > 50:
                logger.info("  SKIP (exists): %s K=%s V=%s (%s)",
                            job.model.label, job.k_type, job.v_type, job.tag)
                if job.tag == "main":
                    skipped_csvs_main.append(str(csv_p))
                else:
                    skipped_csvs_v2.append(str(csv_p))
            else:
                remaining.append(job)
        all_jobs = remaining
        if not all_jobs:
            logger.info("\nAll jobs already completed. Aggregating existing results.\n")

    # Assign GPUs round-robin
    for i, job in enumerate(all_jobs):
        job.gpu_id = gpu_ids[i % len(gpu_ids)]

    # Banner
    n_skipped = len(skipped_csvs_main) + len(skipped_csvs_v2) if args.rerun_missing else 0
    skip_note = f" ({n_skipped} skipped with existing results)" if n_skipped else ""
    logger.info("")
    logger.info("  $ python3 %s", " ".join(sys.argv))
    logger.info("")
    logger.info("=" * 60)
    logger.info(" KV Cache RULER Benchmark")
    logger.info("=" * 60)
    logger.info("  Config:    %s", preset.name)
    logger.info("  Samples:   %s", preset.num_samples)
    logger.info("  Contexts:  %s", preset.ctx_lengths)
    logger.info("  GPUs:      %s", gpu_ids)
    logger.info("  Models:    %s", len(available_models))
    for m in available_models:
        logger.info("    - %s: %s", m.label, m.path)
    logger.info("  Quants:    %s", len(preset.quant_configs))
    for k, v in preset.quant_configs:
        logger.info("    - K=%s V=%s (%s)", k, v, bpw_label(k, v))
    logger.info("  Jobs:      %s to run%s", len(all_jobs), skip_note)
    logger.info("  Output:    %s", output_dir)
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("\nDRY RUN — jobs that would be scheduled:\n")
        for j in all_jobs:
            logger.info("  [GPU%s] %s K=%s V=%s tasks='%s' ctx='%s' n=%s (%s)",
                        j.gpu_id, j.model.label, j.k_type, j.v_type,
                        j.tasks, j.ctx_lengths, j.num_samples, j.tag)
        return

    # Include previously completed results
    csv_paths_main = list(skipped_csvs_main)
    csv_paths_v2 = list(skipped_csvs_v2)

    # Execute remaining jobs with GPU parallelism
    if all_jobs:
        global _progress
        _progress = ProgressTracker(len(all_jobs))
        logger.info("\nRunning %s jobs across %s GPU(s)...\n", len(all_jobs), len(gpu_ids))

        max_workers = len(gpu_ids)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_job = {}
            for job in all_jobs:
                future = pool.submit(run_job, job, extra_args, output_dir)
                future_to_job[future] = job

            for future in as_completed(future_to_job):
                job = future_to_job[future]
                csv_path = future.result()
                if csv_path:
                    if job.tag == "main":
                        csv_paths_main.append(csv_path)
                    else:
                        csv_paths_v2.append(csv_path)

    # Collect and aggregate
    results_main = collect_results(csv_paths_main)
    results_v2 = collect_results(csv_paths_v2)
    ctx_lengths = [int(c) for c in preset.ctx_lengths.split()]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model family lookups
    model_families = {}
    for m in available_models:
        model_families[os.path.basename(m.path)] = m.family

    def filter_by_family(results, family):
        return [r for r in results if model_families.get(r.model, "") == family]

    # ── All tables ─────────────────────────────────────────────────────────
    all_tables = []

    # 1) Main — all models, per model
    if results_main:
        t = format_table(results_main, "KV Cache Quality — Main Tasks (all models)", ctx_lengths)
        logger.info(t)
        all_tables.append(t)

        # 2) Main — aggregated F16 weights (7B-f16)
        r_f16 = filter_by_family(results_main, "7B-f16")
        if r_f16:
            agg_f16 = aggregate_results(r_f16, "7B-F16-avg (Mistral+Llama)")
            t = format_table(agg_f16, "Main Tasks — Aggregated 7B F16 weight models", ctx_lengths)
            logger.info(t)
            all_tables.append(t)

        # 3) Main — aggregated Q4 weights (7B-q4)
        r_q4 = filter_by_family(results_main, "7B-q4")
        if r_q4:
            agg_q4 = aggregate_results(r_q4, "7B-Q4-avg (Mistral+Llama)")
            t = format_table(agg_q4, "Main Tasks — Aggregated 7B Q4 weight models", ctx_lengths)
            logger.info(t)
            all_tables.append(t)

        csv_main_path = output_dir / f"kv-ruler_main_{preset.name}_{timestamp}.csv"
        write_combined_csv(results_main, str(csv_main_path))
        logger.info("\nMain CSV: %s", csv_main_path)

    # 4) V2 — all models, per model
    if results_v2:
        t = format_table(results_v2, "KV Cache Quality — Hard V2 (multi-value, stresses V-side)", ctx_lengths)
        logger.info(t)
        all_tables.append(t)

        # 5) V2 — aggregated F16 weights
        r_f16_v2 = filter_by_family(results_v2, "7B-f16")
        if r_f16_v2:
            agg_f16_v2 = aggregate_results(r_f16_v2, "7B-F16-avg (Mistral+Llama)")
            t = format_table(agg_f16_v2, "Hard V2 — Aggregated 7B F16 weight models", ctx_lengths)
            logger.info(t)
            all_tables.append(t)

        # 6) V2 — aggregated Q4 weights
        r_q4_v2 = filter_by_family(results_v2, "7B-q4")
        if r_q4_v2:
            agg_q4_v2 = aggregate_results(r_q4_v2, "7B-Q4-avg (Mistral+Llama)")
            t = format_table(agg_q4_v2, "Hard V2 — Aggregated 7B Q4 weight models", ctx_lengths)
            logger.info(t)
            all_tables.append(t)

        csv_v2_path = output_dir / f"kv-ruler_v2_{preset.name}_{timestamp}.csv"
        write_combined_csv(results_v2, str(csv_v2_path))
        logger.info("\nV2 CSV: %s", csv_v2_path)

    # Write full text report
    report_path = output_dir / f"kv-ruler_report_{preset.name}_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write("KV Cache RULER Benchmark Report\n")
        f.write("Config: %s\n" % preset.name)
        f.write("Date: %s\n" % timestamp)
        f.write("Command: python3 %s\n" % " ".join(sys.argv))
        f.write("Models: %s\n\n" % ", ".join(m.label for m in available_models))
        for t in all_tables:
            f.write(t)
            f.write("\n\n")
    logger.info("Report: %s", report_path)

    logger.info("\nJob logs/CSVs: %s/", output_dir / "jobs")
    logger.info("Done.")


if __name__ == "__main__":
    main()
