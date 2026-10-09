#!/usr/bin/env python3
"""
ZeroSCROLLS KV cache quality benchmark orchestrator.

Schedules zeroscrolls-bench.sh jobs across multiple GPUs in parallel,
collects per-task CSV results, and produces a 3-category summary table
(Summarization / QA / MultiChoice) plus a per-task breakdown.

Usage:
    python3 tests/test-kv-cache-zeroscrolls.py -c smoke --gpus 0,1
    python3 tests/test-kv-cache-zeroscrolls.py -c full  --gpus 0,1 --output-dir results-zeroscrolls

Config presets:
    smoke -  5 samples/task, 8 tasks, subset quants    (plumbing)
    full  - all validation samples (~20/task), all quants

Mirrors tests/test-kv-cache-longbench.py — same Job, ProgressTracker,
--rerun-missing, results<N>/jobs/ layout, CSV+report writers. The only
material differences are the task / category lists and the smaller
preset matrix (ZeroSCROLLS validation has only ~20 samples per task, so
"full" is the natural unit of work).
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
ZS_BENCH = SCRIPT_DIR / "zeroscrolls-bench.sh"

sys.path.insert(0, str(SCRIPT_DIR))
import kv_cache_eval_common as kv_common  # noqa: E402
from kv_cache_eval_common import (  # noqa: E402
    BPW, bpw_label, ModelDef, MODELS_DEFAULT,
    ProgressTracker,
    _next_available_dir, _find_latest_dir,
)

# ── Quant configs (same shape as LongBench orchestrator) ────────────────────

QUANT_CONFIGS_ALL = [
    ("f16",    "f16"),
    ("tbq4_0", "pq4_0"),
    ("tbq3_0", "pq3_0"),
    ("tbq3_0", "q4_0"),
    ("tbq4_0", "q4_0"),
    ("pq4_0",  "pq4_0"),
    ("pq3_0",  "pq3_0"),
    ("q4_0",   "q4_0"),
]

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
    num_samples: int  # -1 = all validation
    max_length: int
    quant_configs: list = field(default_factory=list)


PRESETS = {
    # ZeroSCROLLS validation is tiny (~20/task × 8 tasks = ~160 samples) so
    # even "full" is cheap. The smoke preset just caps further for fast
    # smoke-testing of new shaders / commits.
    "smoke": ConfigPreset("smoke",  5,  8192, QUANT_CONFIGS_SUBSET),
    "small": ConfigPreset("small", 10, 16384, QUANT_CONFIGS_SUBSET),
    "full":  ConfigPreset("full",  -1, 16384, QUANT_CONFIGS_ALL),
}

# ── Task / category layout (mirrors zeroscrolls-bench.sh) ──────────────────

CATEGORY_TASKS = {
    "Summarization": ["gov_report", "summ_screen_fd", "qmsum", "squality"],
    "QA":            ["qasper", "narrative_qa", "musique"],
    "MultiChoice":   ["quality"],
}
CATEGORY_ORDER = list(CATEGORY_TASKS.keys())
ALL_TASKS = [t for ts in CATEGORY_TASKS.values() for t in ts]

# ── Job / runner plumbing (identical to LongBench orchestrator) ────────────


@dataclass
class Job:
    model: ModelDef
    k_type: str
    v_type: str
    num_samples: int
    max_length: int
    gpu_id: int = 0
    csv_file: str = ""


def run_job(job: Job, extra_args: list[str], output_dir: Path) -> Optional[str]:
    jobs_dir = output_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_name = f"{job.model.label}_{job.k_type}_{job.v_type}".replace("/", "_").replace(" ", "_")
    csv_path = str(jobs_dir / f"{job_name}.csv")
    job.csv_file = csv_path

    env = os.environ.copy()
    env["GGML_VK_VISIBLE_DEVICES"] = str(job.gpu_id)
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu_id)

    ns_arg = "all" if job.num_samples < 0 else str(job.num_samples)
    cmd = [
        "bash", str(ZS_BENCH),
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
    ok = kv_common.run_bench_subprocess(
        cmd, env, label, log_path,
        timeout=12 * 3600,
        score_marker="Category-avg score:",
        errors="replace",
    )
    return csv_path if ok else None


# ── Results collection + tables ────────────────────────────────────────────


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
    results: list[CellResult] = []
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


def category_means_for_cell(results, model, k, v):
    out = {}
    for cat, ts in CATEGORY_TASKS.items():
        per_task = []
        for t in ts:
            hits = [r for r in results
                    if r.model == model and r.cache_k == k and r.cache_v == v and r.task == t]
            if hits:
                per_task.append(hits[0].mean_pct)
        out[cat] = round(sum(per_task) / len(per_task), 2) if per_task else None
    return out


def paper_table(results, title):
    if not results:
        return f"\n{title}\n  (no results)\n"
    models = sorted({r.model for r in results})
    quants = sorted({(r.cache_k, r.cache_v) for r in results},
                    key=lambda kv: (BPW.get(kv[0], 99) + BPW.get(kv[1], 99)))
    col_w = 14
    lines = ["", "=" * 110, f" {title}", "=" * 110]
    headers = ["KV Config", "BPW"] + CATEGORY_ORDER + ["Average"]
    for model_name in models:
        lines.append("")
        lines.append(f"  Model: {model_name}")
        header = f"  {headers[0]:<22}{headers[1]:>6}"
        for h in headers[2:]:
            header += f"{h:>{col_w}}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
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
    lines.append("  Average = unweighted mean of the category means.")
    lines.append("=" * 110)
    return "\n".join(lines)


def per_task_table(results, title):
    if not results:
        return ""
    models = sorted({r.model for r in results})
    quants = sorted({(r.cache_k, r.cache_v) for r in results},
                    key=lambda kv: (BPW.get(kv[0], 99) + BPW.get(kv[1], 99)))
    col_w = 14
    lines = ["", "=" * 160, f" {title}", "=" * 160]
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
                row += f"{hits[0].mean_pct:>{col_w}.1f}" if hits else f"{'--':>{col_w}}"
            lines.append(row)
    lines.append("=" * 160)
    return "\n".join(lines)


def write_combined_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "cache_k", "cache_v", "k_bpw", "v_bpw",
                    "task", "category", "samples", "mean_pct", "stdev_pct"])
        for r in results:
            w.writerow([r.model, r.cache_k, r.cache_v,
                        BPW.get(r.cache_k, "?"), BPW.get(r.cache_v, "?"),
                        r.task, r.category, r.samples, r.mean_pct, r.stdev_pct])


def _job_csv_path(output_dir, job):
    name = f"{job.model.label}_{job.k_type}_{job.v_type}".replace("/", "_").replace(" ", "_")
    return output_dir / "jobs" / f"{name}.csv"


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="ZeroSCROLLS KV cache quality orchestrator")
    parser.add_argument("-c", "--config", default="full", choices=PRESETS.keys())
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--output-dir", default="results-zeroscrolls")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-missing", action="store_true")
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    preset = PRESETS[args.config]
    gpu_ids = [int(g) for g in args.gpus.split(",")]

    base_dir = Path(args.output_dir)
    if args.rerun_missing:
        output_dir = _find_latest_dir(base_dir) or base_dir
        if output_dir == base_dir and not base_dir.exists():
            logger.info("No existing '%s' dir; creating fresh.", base_dir)
        else:
            logger.info("Rerun-missing: reusing %s", output_dir)
    else:
        output_dir = _next_available_dir(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
                    tokenizer="NousResearch/Meta-Llama-3.1-8B-Instruct",
                    label=Path(p).stem,
                    family="custom",
                ))
    else:
        models = MODELS_DEFAULT

    available = [m for m in models if os.path.exists(m.path)]
    if not available:
        logger.error("ERROR: No model files found:")
        for m in models:
            logger.error("  %s", m.path)
        sys.exit(1)
    missing = [m for m in models if not os.path.exists(m.path)]
    if missing:
        logger.warning("WARNING: Skipping missing models:")
        for m in missing:
            logger.warning("  %s", m.path)

    all_jobs = []
    extra_args = ["-ngl", "99", "-fa", "1", "--split-mode", "none"] + args.extra
    for model in available:
        for k, v in preset.quant_configs:
            all_jobs.append(Job(
                model=model, k_type=k, v_type=v,
                num_samples=preset.num_samples,
                max_length=preset.max_length,
            ))

    skipped = []
    if args.rerun_missing:
        remaining = []
        for j in all_jobs:
            p = _job_csv_path(output_dir, j)
            if p.exists() and p.stat().st_size > 50:
                logger.info("  SKIP (exists): %s K=%s V=%s",
                            j.model.label, j.k_type, j.v_type)
                skipped.append(str(p))
            else:
                remaining.append(j)
        all_jobs = remaining
    for i, j in enumerate(all_jobs):
        j.gpu_id = gpu_ids[i % len(gpu_ids)]

    logger.info("")
    logger.info("  $ python3 %s", " ".join(sys.argv))
    logger.info("")
    logger.info("=" * 60)
    logger.info(" KV Cache ZeroSCROLLS Benchmark")
    logger.info("=" * 60)
    logger.info("  Config:      %s (num_samples=%s, max_length=%d)",
                preset.name,
                "all" if preset.num_samples < 0 else preset.num_samples,
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
                f" ({len(skipped)} skipped)" if skipped else "")
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

    csv_paths = list(skipped)
    if all_jobs:
        kv_common.set_progress(ProgressTracker(len(all_jobs)))
        logger.info("\nRunning %d jobs across %d GPU(s)...\n", len(all_jobs), len(gpu_ids))
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            futures = {pool.submit(run_job, j, extra_args, output_dir): j for j in all_jobs}
            for fut in as_completed(futures):
                p = fut.result()
                if p:
                    csv_paths.append(p)

    results = collect_results(csv_paths)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not results:
        logger.error("No results collected. Inspect %s/jobs/*.txt for failures.", output_dir)
        sys.exit(1)

    tables = []
    t1 = paper_table(results, "ZeroSCROLLS — category aggregation")
    logger.info(t1)
    tables.append(t1)
    t2 = per_task_table(results, "ZeroSCROLLS — per-task breakdown")
    logger.info(t2)
    tables.append(t2)

    combined = output_dir / f"kv-zeroscrolls_{preset.name}_{timestamp}.csv"
    write_combined_csv(results, str(combined))
    logger.info("\nCombined CSV: %s", combined)

    report = output_dir / f"kv-zeroscrolls_report_{preset.name}_{timestamp}.txt"
    with open(report, "w") as f:
        f.write("KV Cache ZeroSCROLLS Benchmark Report\n")
        f.write(f"Config: {preset.name}\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Command: python3 {' '.join(sys.argv)}\n")
        f.write(f"Models: {', '.join(m.label for m in available)}\n\n")
        for t in tables:
            f.write(t)
            f.write("\n\n")
    logger.info("Report: %s", report)
    logger.info("Job logs/CSVs: %s/", output_dir / "jobs")
    logger.info("Done.")


if __name__ == "__main__":
    main()
