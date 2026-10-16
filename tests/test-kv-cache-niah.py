#!/usr/bin/env python3
"""
NIAH (Needle In A Haystack) KV cache quality benchmark orchestrator.

Schedules niah-bench.sh jobs across multiple GPUs in parallel, collects
per-cell (ctx × depth) CSV results, and produces a per-cache-config
heatmap plus a cross-config summary.

NIAH's output is fundamentally different from the other 4 evals — there's
no per-task aggregation, just a 2-D grid of binary recall scores. We
render three views:

    1. Per (model, K, V) heatmap     — rows = ctx, cols = depth, cells = 0/100.
    2. Cross-config row-avg table    — rows = (K, V), cols = ctx, cells = mean across depths.
    3. Corners-only table            — only the (ctx, depth) cells where at
                                       least one cache config fell below 1.0.
                                       This is where the signal lives; on strong
                                       models the bulk of the grid is 100/100.

Usage:
    python3 tests/test-kv-cache-niah.py -c smoke --gpus 0
    python3 tests/test-kv-cache-niah.py -c full  --gpus 0,1 --output-dir results-niah

Config presets:
    smoke -  3 ctx × 3 depths × subset quants    ( 9 cells/quant)
    small -  4 ctx × 5 depths × subset quants    (20 cells/quant)
    full  -  5 ctx × 5 depths × all quants       (25 cells/quant, canonical 5×5)
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

logger = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
NH_BENCH = SCRIPT_DIR / "niah-bench.sh"

sys.path.insert(0, str(SCRIPT_DIR))
import kv_cache_eval_common as kv_common  # noqa: E402
from kv_cache_eval_common import (  # noqa: E402
    BPW, bpw_label, ModelDef, MODELS_DEFAULT,
    ProgressTracker,
    _next_available_dir, _find_latest_dir,
)


QUANT_CONFIGS_ALL = [
    ("f16", "f16"),
    ("tbq4_0", "pq4_0"),
    ("tbq3_0", "pq3_0"),
    ("tbq3_0", "q4_0"),
    ("tbq4_0", "q4_0"),
    ("pq4_0", "pq4_0"),
    ("pq3_0", "pq3_0"),
    ("q4_0", "q4_0"),
]

QUANT_CONFIGS_SUBSET = [
    ("f16", "f16"),
    ("tbq4_0", "pq4_0"),
    ("tbq3_0", "pq3_0"),
    ("tbq4_0", "q4_0"),
    ("q4_0", "q4_0"),
]


@dataclass
class ConfigPreset:
    name: str
    ctx_lengths: list
    depth_percents: list
    quant_configs: list = field(default_factory=list)


PRESETS = {
    "smoke": ConfigPreset("smoke",
                          ctx_lengths=[2048, 4096, 8192],
                          depth_percents=[0, 50, 100],
                          quant_configs=QUANT_CONFIGS_SUBSET),
    "small": ConfigPreset("small",
                          ctx_lengths=[2048, 4096, 8192, 16384],
                          depth_percents=[0, 25, 50, 75, 100],
                          quant_configs=QUANT_CONFIGS_SUBSET),
    "full":  ConfigPreset("full",
                          ctx_lengths=[2048, 4096, 8192, 16384, 32768],
                          depth_percents=[0, 25, 50, 75, 100],
                          quant_configs=QUANT_CONFIGS_ALL),
}


@dataclass
class Job:
    model: ModelDef
    k_type: str
    v_type: str
    ctx_lengths: list
    depth_percents: list
    gpu_id: int = 0
    csv_file: str = ""


def run_job(job: Job, extra_args, output_dir: Path):
    jobs_dir = output_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_name = f"{job.model.label}_{job.k_type}_{job.v_type}".replace("/", "_").replace(" ", "_")
    csv_path = str(jobs_dir / f"{job_name}.csv")
    job.csv_file = csv_path

    env = os.environ.copy()
    env["GGML_VK_VISIBLE_DEVICES"] = str(job.gpu_id)
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu_id)

    cmd = [
        "bash", str(NH_BENCH),
        "-m", job.model.path,
        "-ctk", job.k_type,
        "-ctv", job.v_type,
        "--ctx-lengths", " ".join(str(c) for c in job.ctx_lengths),
        "--depth-percents", " ".join(str(d) for d in job.depth_percents),
        "--tokenizer", job.model.tokenizer,
        "--csv", csv_path,
    ]
    cmd.extend(extra_args)

    label = f"[GPU{job.gpu_id}] {job.model.label} K={job.k_type} V={job.v_type}"
    log_path = csv_path.replace(".csv", ".txt")
    ok = kv_common.run_bench_subprocess(
        cmd, env, label, log_path,
        timeout=6 * 3600,
        score_marker="Grid average:",
        errors="replace",
    )
    return csv_path if ok else None


@dataclass
class CellResult:
    model: str
    cache_k: str
    cache_v: str
    ctx_len: int
    depth_percent: int
    score: float  # 0..100 (percent); a single cell is 0 or 100 by construction


def collect_results(csv_paths):
    out = []
    for path in csv_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    out.append(CellResult(
                        model=row["model"], cache_k=row["cache_k"], cache_v=row["cache_v"],
                        ctx_len=int(row["ctx_len"]),
                        depth_percent=int(round(float(row["depth_percent"]))),
                        score=float(row["score"]),
                    ))
                except (KeyError, ValueError) as e:
                    logger.warning("  WARN: bad CSV row in %s: %s", path, e)
    return out


def _sorted_quants(results):
    return sorted({(r.cache_k, r.cache_v) for r in results},
                  key=lambda kv: (BPW.get(kv[0], 99) + BPW.get(kv[1], 99)))


def heatmap_table(results, title):
    """Per-(model, K, V) ctx × depth heatmap. One block per cache config."""
    if not results:
        return f"\n{title}\n  (no results)\n"
    models = sorted({r.model for r in results})
    quants = _sorted_quants(results)
    depths = sorted({r.depth_percent for r in results})
    ctxs = sorted({r.ctx_len for r in results})
    lines = ["", "=" * 120, f" {title}", "=" * 120]
    for model_name in models:
        for k, v in quants:
            avg_bpw = (BPW.get(k, 0) + BPW.get(v, 0)) / 2
            lines.append("")
            lines.append(f"  {model_name}  K={k} V={v}  (avg {avg_bpw:.2f} bpw)")
            header = f"    {'ctx ↓':<8}"
            for d in depths:
                header += f"  {d:>3}%"
            header += "   row-avg"
            lines.append(header)
            lines.append("    " + "-" * (len(header) - 4))
            grid_sum, grid_n = 0.0, 0
            for ctx in ctxs:
                row = f"    {ctx:<8}"
                row_sum, row_n = 0.0, 0
                for d in depths:
                    hits = [r for r in results
                            if r.model == model_name and r.cache_k == k and r.cache_v == v
                            and r.ctx_len == ctx and r.depth_percent == d]
                    if hits:
                        s = hits[0].score
                        row += f"  {int(round(s)):>3}"
                        row_sum += s
                        row_n += 1
                        grid_sum += s
                        grid_n += 1
                    else:
                        row += f"  {'-':>3}"
                if row_n > 0:
                    row += f"   {row_sum / row_n:>6.1f}"
                else:
                    row += f"   {'-':>6}"
                lines.append(row)
            if grid_n > 0:
                lines.append(f"    {'grid avg':<8}" + "  " * (len(depths) + 1)
                             + f"  {grid_sum / grid_n:>6.1f}")
    lines.append("")
    lines.append("=" * 120)
    return "\n".join(lines)


def cross_config_table(results, title):
    """Rows = (K, V) configs, columns = ctx lengths, cells = row-avg across depths.

    One block per model. This is the "where does each quant start to lose
    retrieval as ctx grows" view — the headline aggregation for NIAH.
    """
    if not results:
        return ""
    models = sorted({r.model for r in results})
    quants = _sorted_quants(results)
    ctxs = sorted({r.ctx_len for r in results})
    col_w = 10
    lines = ["", "=" * 120, f" {title}", "=" * 120]
    for model_name in models:
        lines.append("")
        lines.append(f"  Model: {model_name}")
        header = f"  {'KV Config':<22}{'BPW':>6}"
        for ctx in ctxs:
            header += f"{ctx:>{col_w}}"
        header += f"{'all-avg':>{col_w}}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for k, v in quants:
            avg_bpw = (BPW.get(k, 0) + BPW.get(v, 0)) / 2
            row = f"  {k}/{v:<14}{avg_bpw:>6.2f}"
            all_sum, all_n = 0.0, 0
            for ctx in ctxs:
                hits = [r for r in results
                        if r.model == model_name and r.cache_k == k and r.cache_v == v
                        and r.ctx_len == ctx]
                if hits:
                    avg = sum(h.score for h in hits) / len(hits)
                    row += f"{avg:>{col_w}.1f}"
                    all_sum += sum(h.score for h in hits)
                    all_n += len(hits)
                else:
                    row += f"{'--':>{col_w}}"
            if all_n > 0:
                row += f"{all_sum / all_n:>{col_w}.1f}"
            else:
                row += f"{'--':>{col_w}}"
            lines.append(row)
    lines.append("")
    lines.append("  Cells are mean recall (0..100) averaged across the depth column at that ctx.")
    lines.append("=" * 120)
    return "\n".join(lines)


def corners_only_table(results, title):
    """Only the (model, ctx, depth) cells where at least one cache config
    dropped below 1.0. This is where quantization-quality signal actually
    lives — on strong instruct models the bulk of the grid is 100.

    If every cell across every config is 100, returns a single-line note.
    """
    if not results:
        return ""
    models = sorted({r.model for r in results})
    quants = _sorted_quants(results)
    depths = sorted({r.depth_percent for r in results})
    ctxs = sorted({r.ctx_len for r in results})

    interesting = set()
    for model_name in models:
        for ctx in ctxs:
            for d in depths:
                hits = [r for r in results
                        if r.model == model_name and r.ctx_len == ctx and r.depth_percent == d]
                if any(h.score < 100.0 for h in hits):
                    interesting.add((model_name, ctx, d))

    if not interesting:
        return ("\n" + "=" * 120 + f"\n {title}\n" + "=" * 120
                + "\n  All cells = 100 across every (model, ctx, depth, cache config). "
                  "Cache quantization preserves perfect needle retrieval on this matrix.\n"
                + "=" * 120)

    lines = ["", "=" * 120, f" {title}", "=" * 120]
    for model_name in models:
        mod_cells = sorted(c for c in interesting if c[0] == model_name)
        if not mod_cells:
            continue
        lines.append("")
        lines.append(f"  Model: {model_name}  ({len(mod_cells)} interesting cells)")
        header = f"  {'ctx':>6}  {'depth':>6}"
        for k, v in quants:
            header += f"  {k + '/' + v:>14}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for (_, ctx, d) in mod_cells:
            row = f"  {ctx:>6}  {d:>5}%"
            for k, v in quants:
                hits = [r for r in results
                        if r.model == model_name and r.cache_k == k and r.cache_v == v
                        and r.ctx_len == ctx and r.depth_percent == d]
                row += f"  {int(round(hits[0].score)) if hits else '--':>14}"
            lines.append(row)
    lines.append("=" * 120)
    return "\n".join(lines)


def write_combined_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "cache_k", "cache_v", "k_bpw", "v_bpw",
                    "ctx_len", "depth_percent", "score"])
        for r in results:
            w.writerow([r.model, r.cache_k, r.cache_v,
                        BPW.get(r.cache_k, "?"), BPW.get(r.cache_v, "?"),
                        r.ctx_len, r.depth_percent, r.score])


def _job_csv_path(output_dir, job):
    name = f"{job.model.label}_{job.k_type}_{job.v_type}".replace("/", "_").replace(" ", "_")
    return output_dir / "jobs" / f"{name}.csv"


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="NIAH KV cache quality orchestrator")
    parser.add_argument("-c", "--config", default="full", choices=PRESETS.keys())
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--output-dir", default="results-niah")
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
                ctx_lengths=preset.ctx_lengths,
                depth_percents=preset.depth_percents,
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

    cells_per_job = len(preset.ctx_lengths) * len(preset.depth_percents)
    logger.info("")
    logger.info("  $ python3 %s", " ".join(sys.argv))
    logger.info("")
    logger.info("=" * 60)
    logger.info(" KV Cache NIAH (Needle In A Haystack) Benchmark")
    logger.info("=" * 60)
    logger.info("  Config:      %s (%d ctx × %d depths = %d cells/job)",
                preset.name, len(preset.ctx_lengths),
                len(preset.depth_percents), cells_per_job)
    logger.info("  GPUs:        %s", gpu_ids)
    logger.info("  Models:      %d", len(available))
    for m in available:
        logger.info("    - %s: %s", m.label, m.path)
    logger.info("  Quants:      %d", len(preset.quant_configs))
    for k, v in preset.quant_configs:
        logger.info("    - K=%s V=%s (%s)", k, v, bpw_label(k, v))
    logger.info("  ctx_lengths: %s", " ".join(str(c) for c in preset.ctx_lengths))
    logger.info("  depth (%%):   %s", " ".join(str(d) for d in preset.depth_percents))
    logger.info("  Jobs:        %d to run%s",
                len(all_jobs),
                f" ({len(skipped)} skipped)" if skipped else "")
    logger.info("  Output:      %s", output_dir)
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("\nDRY RUN — jobs that would be scheduled:\n")
        for j in all_jobs:
            logger.info("  [GPU%d] %s K=%s V=%s ctx=%s depths=%s",
                        j.gpu_id, j.model.label, j.k_type, j.v_type,
                        " ".join(str(c) for c in j.ctx_lengths),
                        " ".join(str(d) for d in j.depth_percents))
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
    t1 = cross_config_table(results, "NIAH — cross-config row averages (mean across depth column)")
    logger.info(t1)
    tables.append(t1)
    t2 = corners_only_table(results, "NIAH — corner cells (where at least one config dropped below 100)")
    logger.info(t2)
    tables.append(t2)
    t3 = heatmap_table(results, "NIAH — per-config heatmaps (ctx × depth)")
    logger.info(t3)
    tables.append(t3)

    combined = output_dir / f"kv-niah_{preset.name}_{timestamp}.csv"
    write_combined_csv(results, str(combined))
    logger.info("\nCombined CSV: %s", combined)

    report = output_dir / f"kv-niah_report_{preset.name}_{timestamp}.txt"
    with open(report, "w") as f:
        f.write("KV Cache NIAH (Needle In A Haystack) Benchmark Report\n")
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
