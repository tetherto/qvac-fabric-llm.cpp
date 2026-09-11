"""
Shared infrastructure for the KV-cache eval orchestrators.

The four orchestrators (test-kv-cache-{ruler,longbench,zeroscrolls,leval}.py)
share this module's constants and helpers. Eval-specific dataclasses (Job,
ConfigPreset, CellResult), CSV schemas, and table formatters stay per-eval
because their shapes legitimately differ.

The single most important piece here is MODELS_DEFAULT: keeping it in one
place prevents the recurring wrong-tokenizer bug we hit when one orchestrator
got harmonized and the others didn't.
"""

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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
    return f"K:{BPW.get(k_type, '?')} V:{BPW.get(v_type, '?')}"


# ── Model definitions ──────────────────────────────────────────────────────


@dataclass
class ModelDef:
    path: str
    tokenizer: str
    label: str
    # family: used by RULER to bucket cross-model aggregations.
    # Non-RULER orchestrators set family=label, which makes the
    # aggregation degenerate to per-model rows.
    family: str


# Three q8 models mirroring tests/test-kv-cache-quantization-perp.sh's presets.
# Keeping the same matrix across RULER / LongBench / ZeroSCROLLS / L-Eval /
# perp / perf means every cell can be cross-referenced cell-by-cell.
MODELS_DEFAULT = [
    ModelDef(
        path="models/Mistral-7B-Instruct-v0.3-Q8_0.gguf",
        tokenizer="mistralai/Mistral-7B-Instruct-v0.3",
        label="Mistral-7B",
        family="Mistral-7B",
    ),
    ModelDef(
        path="models/Llama-3.1-8B-Instruct-Q8_0.gguf",
        # Un-gated mirror — same tokenizer.json SHA as the official meta-llama
        # repo, no HF login required.
        tokenizer="NousResearch/Meta-Llama-3.1-8B-Instruct",
        label="Llama-3.1-8B",
        family="Llama-3.1-8B",
    ),
    ModelDef(
        path="models/Qwen3.5-4B-Q8_0.gguf",
        tokenizer="unsloth/Qwen3.5-4B",
        label="Qwen3.5-4B",
        family="Qwen3.5-4B",
    ),
]


# ── Duration formatting ────────────────────────────────────────────────────


def _fmt_duration(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


# ── Progress tracking ──────────────────────────────────────────────────────
# A single ProgressTracker is registered per orchestrator run via
# set_progress(); run_job callers read it back via get_progress(). The
# free-function indirection is so that orchestrators that did `from
# kv_cache_eval_common import _progress` would see updates — direct `import`
# of a module-level name binds at import time and would miss assignments.

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
                remaining = max(0, self.total - self.completed)
                eta_str = _fmt_duration(avg * remaining)
            else:
                eta_str = "?"
            elapsed_str = _fmt_duration(elapsed)
            return f"[{self.completed}/{self.total}] elapsed {elapsed_str}, ETA {eta_str}"


_progress: Optional[ProgressTracker] = None


def set_progress(p: Optional[ProgressTracker]) -> None:
    global _progress
    _progress = p


def get_progress() -> Optional[ProgressTracker]:
    return _progress


# ── Output-directory helpers ───────────────────────────────────────────────


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


# ── Subprocess scaffolding for run_job ─────────────────────────────────────


def run_bench_subprocess(
    cmd: list,
    env: dict,
    label: str,
    log_path: str,
    timeout: int,
    score_marker: str,
    errors: Optional[str],
) -> bool:
    """Run a bench script as a subprocess, handle progress + logging.

    Returns True on success (exit 0), False on non-zero exit or timeout.
    The progress tracker (if any) ticks once on success only.

    `score_marker` is the substring used to find a score line in stdout
    for the DONE log message — e.g. "Overall accuracy:" (RULER) or
    "Category-avg score:" (LongBench / ZeroSCROLLS / L-Eval). If no line
    contains the marker, "?" is logged.

    `errors` is forwarded to subprocess.run's decoding mode. Callers pass
    "replace" for the bench scripts that emit user-visible model output
    (LongBench / ZeroSCROLLS / L-Eval) since `printf "%.Ns"` in bash can
    split a multi-byte codepoint mid-sequence — see the longbench-bench.sh
    rationale for the bash-side iconv guard. Pass None for strict decoding
    (RULER, whose output is fully ASCII metrics).

    Caller is responsible for assembling cmd[] (per-eval bench script
    path + per-eval flag set) and computing csv_path / log_path / env;
    those are the bits that legitimately differ across orchestrators.
    """
    cmd_str = " ".join(cmd)
    with _print_lock:
        logger.info("  START: %s", label)
        logger.info("  $ %s", cmd_str)
        logger.info("  log: %s", log_path)
        logger.info("")

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            errors=errors, timeout=timeout,
        )
        if result.returncode != 0:
            with _print_lock:
                logger.info("  FAIL:  %s", label)
                logger.info("         stderr: %s", result.stderr[-500:])
            return False
    except subprocess.TimeoutExpired:
        with _print_lock:
            logger.info("  TIMEOUT: %s", label)
        return False

    score = "?"
    for line in result.stdout.splitlines():
        if score_marker in line:
            score = line.strip().split(score_marker)[1].strip()
            break

    progress = get_progress()
    progress_str = progress.tick() if progress else ""
    with _print_lock:
        logger.info("  DONE:  %s  =>  %s  %s", label, score, progress_str)
    return True
