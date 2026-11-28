#!/usr/bin/env python3
"""
LongBench-E KV cache quality benchmark orchestrator.

Schedules longbench-bench.sh jobs across multiple GPUs in parallel, collects
per-task CSV results, and produces aggregated comparison tables in the layout
used by the TurboQuant paper Table 1 (six category columns + Average).

Usage:
    python3 tests/test-kv-cache-longbench.py -c smoke --gpus 0,1 --output-dir results/
    python3 tests/test-kv-cache-longbench.py -c small --gpus 0,1
    python3 tests/test-kv-cache-longbench.py -c paper --gpus 0,1 --dry-run

Config presets:
    smoke  -  2 samples/task, 13 tasks, subset quants                  (plumbing)
    small  - 10 samples/task, 13 tasks, subset quants                  (iteration)
    mid    - 50 samples/task, 13 tasks, all quants                     (signal)
    paper  - full LongBench-E (~300 samples/task), 13 tasks, all quants
             (paper Table 1 reproduction — ~24h/cell on a 5090)

Mirrors tests/test-kv-cache-ruler.py: same Job, ProgressTracker, --rerun-missing,
results<N>/jobs/ layout, CSV+report writers. Differences:
    * No ctx_lengths axis (LongBench-E samples have fixed mixed lengths).
    * 6-category aggregation in format_table (SingleQA/MultiQA/Summ/FewShot/
      Synth/Code) instead of RULER's per-(task, ctx) grid.
    * "Score" column is the paper's "Average": unweighted mean of the 6
      category means, NOT a sample-weighted overall mean. Matches §4.3 Table 1.
"""

import argparse
import csv
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
LONGBENCH_BENCH = SCRIPT_DIR / "longbench-bench.sh"

sys.path.insert(0, str(SCRIPT_DIR))
import kv_cache_eval_common as kv_common  # noqa: E402
from kv_cache_eval_common import (  # noqa: E402
    BPW, bpw_label, ModelDef, MODELS_DEFAULT,
    ProgressTracker,
    _next_available_dir, _find_latest_dir,
)

# ── Quant configurations ───────────────────────────────────────────────────
# Same shape as test-kv-cache-ruler.py so cross-bench comparisons line up.
# (pq3_0 ~3.25 bpw is the closest llama.cpp type to the paper's 3.5-bit
# TurboQuant; tbq3_0 ~4.25 bpw is the next-step-up integer variant.)

QUANT_CONFIGS_ALL = [
    ("f16",    "f16"),       # baseline (paper Table 1 first row)
    ("tbq4_0", "pq4_0"),
    ("tbq3_0", "pq3_0"),
    ("tbq3_0", "q4_0"),
    ("tbq4_0", "q4_0"),
    ("pq4_0",  "pq4_0"),
    ("pq3_0",  "pq3_0"),
    ("q4_0",   "q4_0"),
]

# Subset always includes f16 — without it we can't see whether TBQ matches the
# unquantized cache (the paper's headline claim).
QUANT_CONFIGS_SUBSET = [
    ("f16",    "f16"),
    ("tbq4_0", "pq4_0"),
    ("tbq3_0", "pq3_0"),
    ("tbq4_0", "q4_0"),
    ("q4_0",   "q4_0"),
]

# ── Config presets ─────────────────────────────────────────────────────────


@dataclass
class ConfigPreset:
    name: str
    num_samples: int  # -1 means "use full LongBench-E split" (paper preset).
    max_length: int   # Token cap before middle-truncation.
    quant_configs: list = field(default_factory=list)


PRESETS = {
    # smoke uses a tight max_length so the slowest samples don't dominate the
    # plumbing-check runtime — most LongBench-E samples have 4k-30k tokens,
    # and at 31500 a single prefill on an 8B Q4 model is ~4.5 min.
    "smoke": ConfigPreset("smoke", 2,   4096, QUANT_CONFIGS_SUBSET),
    "small": ConfigPreset("small", 10,  8192, QUANT_CONFIGS_SUBSET),
    "mid":   ConfigPreset("mid",   50, 16384, QUANT_CONFIGS_ALL),
    # Paper preset: full LongBench-E split, 31500-token cap matches what the
    # paper's setup would have allowed for Llama-3.1-8B-Instruct (32k usable
    # context after generation budget + chat-template overhead).
    "paper": ConfigPreset("paper", -1, 31500, QUANT_CONFIGS_ALL),
}

# ── Task / category layout (mirrors longbench-bench.sh) ────────────────────

CATEGORY_TASKS = {
    "SingleQA":      ["qasper", "multifieldqa_en"],
    "MultiQA":       ["hotpotqa", "2wikimqa"],
    "Summarization": ["gov_report", "multi_news"],
    "FewShot":       ["trec", "triviaqa", "samsum"],
    "Synthetic":     ["passage_count", "passage_retrieval_en"],
    "Code":          ["lcc", "repobench-p"],
}
CATEGORY_ORDER = list(CATEGORY_TASKS.keys())
ALL_TASKS = [t for ts in CATEGORY_TASKS.values() for t in ts]

# ── Job definition ─────────────────────────────────────────────────────────


@dataclass
class Job:
    model: ModelDef
    k_type: str
    v_type: str
    num_samples: int
    max_length: int
    gpu_id: int = 0
    csv_file: str = ""


# ── Run a single longbench-bench.sh job ────────────────────────────────────


def run_job(job: Job, extra_args: list[str], output_dir: Path) -> Optional[str]:
    jobs_dir = output_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    job_name = f"{job.model.label}_{job.k_type}_{job.v_type}"
    job_name = job_name.replace("/", "_").replace(" ", "_")
    csv_path = str(jobs_dir / f"{job_name}.csv")
    job.csv_file = csv_path

    env = os.environ.copy()
    # Pin one GPU per worker. ggml-vulkan honours GGML_VK_VISIBLE_DEVICES;
    # we set CUDA_VISIBLE_DEVICES too in case the same model is rebuilt with
    # CUDA for sm_120 (RTX 5090) once CUDA 12.8 lands here.
    env["GGML_VK_VISIBLE_DEVICES"] = str(job.gpu_id)
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu_id)

    ns_arg = "all" if job.num_samples < 0 else str(job.num_samples)
    cmd = [
        "bash", str(LONGBENCH_BENCH),
        "-m", job.model.path,
        "-ctk", job.k_type,
        "-ctv", job.v_type,
        "--num-samples", ns_arg,
        "--max-length", str(job.max_length),
        "--tokenizer", job.model.tokenizer,
        "--csv", csv_path,
    ]
    cmd.extend(extra_args)

    label = f"[GPU{job.gpu_id}] {job.model.label} K={job.k_type} V={job.v_type}"
    log_path = csv_path.replace(".csv", ".txt")
    # 12 h cap covers the paper preset (~24h * 0.5 if we get any GPU
    # speedup); reraise as a soft fail rather than blocking everything.
    ok = kv_common.run_bench_subprocess(
        cmd, env, label, log_path,
        timeout=12 * 3600,
        score_marker="Category-avg score:",
        errors="replace",
    )
    return csv_path if ok else None


# ── Collect all CSV results ────────────────────────────────────────────────


@dataclass
class CellResult:
    model: str
    cache_k: str
    cache_v: str
    task: str
    category: str
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
                        category=row["category"],
                        samples=int(row["samples"]),
                        mean_pct=float(row["mean_pct"]),
                        stdev_pct=float(row["stdev_pct"]),
                    ))
                except (KeyError, ValueError) as e:
                    logger.warning("  WARN: bad CSV row in %s: %s", path, e)
    return results


# ── Aggregation: paper Table 1 layout ──────────────────────────────────────


def category_means_for_cell(
    results: list[CellResult],
    model: str,
    k: str,
    v: str,
) -> dict[str, Optional[float]]:
    """Return {category: mean_of_task_means} for a single (model, K, V) cell.

    Paper convention: each category is the unweighted mean of its task means.
    Missing tasks => the category is None (skipped from the Average column).
    """
    out: dict[str, Optional[float]] = {}
    for cat, cat_tasks in CATEGORY_TASKS.items():
        per_task = []
        for t in cat_tasks:
            hits = [r for r in results
                    if r.model == model and r.cache_k == k and r.cache_v == v
                    and r.task == t]
            if hits:
                per_task.append(hits[0].mean_pct)
        out[cat] = round(sum(per_task) / len(per_task), 2) if per_task else None
    return out


def paper_table(results: list[CellResult], title: str) -> str:
    """Six-category table per (model, K, V) — the paper's Table 1 layout."""
    if not results:
        return f"\n{title}\n  (no results)\n"

    models = sorted({r.model for r in results})
    quants = sorted({(r.cache_k, r.cache_v) for r in results},
                    key=lambda kv: (BPW.get(kv[0], 99) + BPW.get(kv[1], 99)))

    col_w = 13
    lines = []
    lines.append("")
    lines.append("=" * 132)
    lines.append(f" {title}")
    lines.append("=" * 132)

    headers = ["KV Config", "BPW"] + CATEGORY_ORDER + ["Average"]
    for model_name in models:
        lines.append("")
        lines.append(f"  Model: {model_name}")
        header_str = f"  {headers[0]:<22}{headers[1]:>6}"
        for h in headers[2:]:
            header_str += f"{h:>{col_w}}"
        lines.append(header_str)
        lines.append("  " + "-" * (len(header_str) - 2))

        for k, v in quants:
            avg_bpw = (BPW.get(k, 0) + BPW.get(v, 0)) / 2
            cats = category_means_for_cell(results, model_name, k, v)
            cat_vals = [c for c in cats.values() if c is not None]
            paper_avg = round(sum(cat_vals) / len(cat_vals), 2) if cat_vals else None

            row = f"  {k}/{v:<14}{avg_bpw:>6.2f}"
            for cat in CATEGORY_ORDER:
                row += f"{cats[cat]:>{col_w}.2f}" if cats[cat] is not None else f"{'--':>{col_w}}"
            row += f"{paper_avg:>{col_w}.2f}" if paper_avg is not None else f"{'--':>{col_w}}"
            lines.append(row)

    lines.append("")
    lines.append("  Average = unweighted mean of the 6 category means (paper Table 1 convention).")
    lines.append("=" * 132)
    return "\n".join(lines)


def per_task_table(results: list[CellResult], title: str) -> str:
    """Per-task breakdown — useful for spotting which task drives a category drop."""
    if not results:
        return ""
    models = sorted({r.model for r in results})
    quants = sorted({(r.cache_k, r.cache_v) for r in results},
                    key=lambda kv: (BPW.get(kv[0], 99) + BPW.get(kv[1], 99)))
    col_w = 11
    lines = []
    lines.append("")
    lines.append("=" * 200)
    lines.append(f" {title}")
    lines.append("=" * 200)
    for model_name in models:
        lines.append("")
        lines.append(f"  Model: {model_name}")
        header = f"  {'KV Config':<22}"
        for t in ALL_TASKS:
            header += f"{t[:col_w - 1]:>{col_w}}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for k, v in quants:
            row = f"  {k}/{v:<14}"
            for t in ALL_TASKS:
                hits = [r for r in results
                        if r.model == model_name and r.cache_k == k
                        and r.cache_v == v and r.task == t]
                if hits:
                    row += f"{hits[0].mean_pct:>{col_w}.1f}"
                else:
                    row += f"{'--':>{col_w}}"
            lines.append(row)
    lines.append("=" * 200)
    return "\n".join(lines)


def write_combined_csv(results: list[CellResult], path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "cache_k", "cache_v", "k_bpw", "v_bpw",
            "task", "category", "samples", "mean_pct", "stdev_pct",
        ])
        for r in results:
            w.writerow([
                r.model, r.cache_k, r.cache_v,
                BPW.get(r.cache_k, "?"), BPW.get(r.cache_v, "?"),
                r.task, r.category, r.samples, r.mean_pct, r.stdev_pct,
            ])


# ── Output dir helpers (RULER pattern) ─────────────────────────────────────


def _job_csv_path(output_dir: Path, job: Job) -> Path:
    name = f"{job.model.label}_{job.k_type}_{job.v_type}"
    name = name.replace("/", "_").replace(" ", "_")
    return output_dir / "jobs" / f"{name}.csv"


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="LongBench-E KV cache quality orchestrator")
    parser.add_argument("-c", "--config", default="small", choices=PRESETS.keys(),
                        help="Config preset (default: small)")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs (default: 0)")
    parser.add_argument("--models", nargs="*",
                        help="Model paths to use (overrides defaults)")
    parser.add_argument("--output-dir", default="results-longbench",
                        help="Directory for output files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print scheduled jobs without launching")
    parser.add_argument("--rerun-missing", action="store_true",
                        help="Reuse latest output dir; skip jobs whose CSV already exists")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="Extra args forwarded to longbench-bench.sh / llama-server")
    args = parser.parse_args()

    preset = PRESETS[args.config]
    gpu_ids = [int(g) for g in args.gpus.split(",")]

    base_dir = Path(args.output_dir)
    if args.rerun_missing:
        output_dir = _find_latest_dir(base_dir) or base_dir
        if output_dir == base_dir and not base_dir.exists():
            logger.info("No existing '%s' dir found, creating fresh one.", base_dir)
        else:
            logger.info("Rerun-missing: reusing %s", output_dir)
    else:
        output_dir = _next_available_dir(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model set
    if args.models:
        models = []
        for p in args.models:
            for m in MODELS_DEFAULT:
                if m.path == p:
                    models.append(m)
                    break
            else:
                # Custom model: use Llama-3.1's tokenizer as a safe default;
                # user can also pre-detect by editing the dataclass.
                models.append(ModelDef(
                    path=p,
                    tokenizer="NousResearch/Meta-Llama-3.1-8B-Instruct",
                    label=Path(p).stem,
                    family="custom",
                ))
    else:
        models = MODELS_DEFAULT

    available = [m for m in models if os.path.exists(m.path)]
    if not available:
        logger.error("ERROR: No model files found. Expected:")
        for m in models:
            logger.error("  %s", m.path)
        sys.exit(1)
    missing = [m for m in models if not os.path.exists(m.path)]
    if missing:
        logger.warning("WARNING: Skipping missing models:")
        for m in missing:
            logger.warning("  %s", m.path)

    # Build job list: every (model × quant_config) cell.
    all_jobs: list[Job] = []
    extra_args = ["-ngl", "99", "-fa", "1", "--split-mode", "none"] + args.extra
    for model in available:
        for k_type, v_type in preset.quant_configs:
            all_jobs.append(Job(
                model=model, k_type=k_type, v_type=v_type,
                num_samples=preset.num_samples,
                max_length=preset.max_length,
            ))

    # --rerun-missing: skip jobs whose CSV is already populated.
    skipped_csvs: list[str] = []
    if args.rerun_missing:
        remaining = []
        for job in all_jobs:
            p = _job_csv_path(output_dir, job)
            if p.exists() and p.stat().st_size > 50:
                logger.info("  SKIP (exists): %s K=%s V=%s",
                            job.model.label, job.k_type, job.v_type)
                skipped_csvs.append(str(p))
            else:
                remaining.append(job)
        all_jobs = remaining

    for i, job in enumerate(all_jobs):
        job.gpu_id = gpu_ids[i % len(gpu_ids)]

    # Banner
    n_skipped = len(skipped_csvs)
    logger.info("")
    logger.info("  $ python3 %s", " ".join(sys.argv))
    logger.info("")
    logger.info("=" * 60)
    logger.info(" KV Cache LongBench-E Benchmark")
    logger.info("=" * 60)
    logger.info("  Config:      %s (num_samples=%s, max_length=%d)",
                preset.name, "all" if preset.num_samples < 0 else preset.num_samples,
                preset.max_length)
    logger.info("  GPUs:        %s", gpu_ids)
    logger.info("  Models:      %d", len(available))
    for m in available:
        logger.info("    - %s: %s", m.label, m.path)
    logger.info("  Quants:      %d", len(preset.quant_configs))
    for k, v in preset.quant_configs:
        logger.info("    - K=%s V=%s (%s)", k, v, bpw_label(k, v))
    logger.info("  Jobs:        %d to run%s",
                len(all_jobs),
                f" ({n_skipped} skipped)" if n_skipped else "")
    logger.info("  Output:      %s", output_dir)
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("\nDRY RUN — jobs that would be scheduled:\n")
        for j in all_jobs:
            logger.info("  [GPU%d] %s K=%s V=%s n=%s m=%d",
                        j.gpu_id, j.model.label, j.k_type, j.v_type,
                        "all" if j.num_samples < 0 else j.num_samples,
                        j.max_length)
        return

    csv_paths: list[str] = list(skipped_csvs)

    if all_jobs:
        kv_common.set_progress(ProgressTracker(len(all_jobs)))
        logger.info("\nRunning %d jobs across %d GPU(s)...\n",
                    len(all_jobs), len(gpu_ids))
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            future_to_job = {pool.submit(run_job, j, extra_args, output_dir): j
                             for j in all_jobs}
            for fut in as_completed(future_to_job):
                p = fut.result()
                if p:
                    csv_paths.append(p)

    results = collect_results(csv_paths)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not results:
        logger.error("No results collected. Inspect %s/jobs/*.txt for failures.", output_dir)
        sys.exit(1)

    all_tables: list[str] = []

    # 1) Paper Table 1 layout — one row per (model, K, V).
    t1 = paper_table(results, "LongBench-E — paper Table 1 layout (6-category aggregation)")
    logger.info(t1)
    all_tables.append(t1)

    # 2) Per-task breakdown — same rows, fine-grained columns.
    t2 = per_task_table(results, "LongBench-E — per-task breakdown")
    logger.info(t2)
    all_tables.append(t2)

    combined_csv = output_dir / f"kv-longbench_{preset.name}_{timestamp}.csv"
    write_combined_csv(results, str(combined_csv))
    logger.info("\nCombined CSV: %s", combined_csv)

    report_path = output_dir / f"kv-longbench_report_{preset.name}_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write("KV Cache LongBench-E Benchmark Report\n")
        f.write(f"Config: {preset.name}\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Command: python3 {' '.join(sys.argv)}\n")
        f.write(f"Models: {', '.join(m.label for m in available)}\n\n")
        for t in all_tables:
            f.write(t)
            f.write("\n\n")
    logger.info("Report: %s", report_path)
    logger.info("Job logs/CSVs: %s/", output_dir / "jobs")
    logger.info("Done.")


if __name__ == "__main__":
    main()
