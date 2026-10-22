#!/usr/bin/env python3
"""Push qvac-bench results to the dashboards TimescaleDB.

Used live from qvac-bench.py --db-url, on a results directory with this
script, or on GitLab CI artifacts through qvac_bench_gitlab.py.

Row keys match the dashboards repo importer (import_llama_bench_output.py),
so both paths upsert the same rows. psycopg is only needed when pushing.
"""

import argparse
import json
import logging
import os
import re
import statistics
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def log(label: str, msg: str) -> None:
    logger.info("[%s] %s", label, msg)


BENCH_COLUMNS = (
    "time", "commit_sha", "branch", "build_number", "device", "backend",
    "model", "metric", "value", "stddev", "n_samples", "samples", "run_url",
)
BENCH_KEY_COLUMNS = ("time", "commit_sha", "device", "backend", "model", "metric")
# value and run_url are not overwritten on re-push.
BENCH_UPDATE_COLUMNS = tuple(
    c for c in BENCH_COLUMNS if c not in (*BENCH_KEY_COLUMNS, "value", "run_url")
)


def _upsert_sql(table: str, columns: tuple, key: tuple, update: tuple) -> str:
    head = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ("
    values = ", ".join(["%s"] * len(columns))
    tail = (
        f") ON CONFLICT ({', '.join(key)}) DO UPDATE SET "
        + ", ".join(f"{c} = EXCLUDED.{c}" for c in update)
    )
    return head + values + tail


# Keep a stored run_url, only fill a NULL one.
BENCH_UPSERT_SQL = _upsert_sql(
    "benchmark_results", BENCH_COLUMNS, BENCH_KEY_COLUMNS, BENCH_UPDATE_COLUMNS,
) + ", run_url = COALESCE(benchmark_results.run_url, EXCLUDED.run_url)"

BUILD_TAG_RE = re.compile(r"^b(\d+)$")
BUILD_NUMBER_RE = re.compile(r"b(\d+)$")

# Stems are <tool>-<sha>-<backend>-<model>-<hash>; other tools are skipped.
KNOWN_TOOLS = ("llama-finetune-lora", "llama-mtmd-cli", "llama-bench")

# llama-bench says "Vulkan", .status files say "vulkan".
BACKEND_LABELS = {"vulkan": "Vulkan", "cuda": "CUDA", "metal": "Metal", "cpu": "CPU"}

# finetune progress is \r-separated on one line.
VAL_SEGMENT_RE = re.compile(r"val:[^\r\n]*")
LOSS_RE = re.compile(r"loss=(nan|[0-9.eE+-]+)(?:±(nan|[0-9.eE+-]+))?")
# Appended to the finetune stdout by qvac-bench.py.
WALL_CLOCK_RE = re.compile(r"qvac_bench_wall_clock_sec:\s*([0-9.]+)")


def tool_from_stem(stem: str) -> "str | None":
    for tool in KNOWN_TOOLS:
        if stem.startswith(f"{tool}-"):
            return tool
    return None


def normalize_backend(backend: "str | None") -> str:
    if not backend:
        return ""
    return BACKEND_LABELS.get(backend.lower(), backend)


def _num(value) -> "float | None":
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def normalize_branch_label(branch: "str | None") -> "str | None":
    if not branch:
        return None
    if BUILD_TAG_RE.match(branch):
        return f"upstream/{branch}"
    return branch


def build_number_from_label(label: "str | None") -> "int | None":
    if not label:
        return None
    m = BUILD_NUMBER_RE.search(label)
    return int(m.group(1)) if m else None


def iter_records(file: Path):
    """Yield result dicts from a .stdout file, skipping non-JSON lines."""
    text = file.read_text(encoding="utf-8", errors="replace")
    if not text.lstrip():
        return
    dec = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = dec.raw_decode(text, idx)
        except json.JSONDecodeError:
            nl = text.find("\n", idx)
            if nl < 0:
                break
            idx = nl + 1
            continue
        idx = end
        if isinstance(obj, dict):
            yield obj
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    yield item


def sibling(stem: Path, suffix: str) -> Path:
    # stems contain dots, so with_suffix() does not work
    return stem.with_name(stem.name + suffix)


def load_json(path: Path) -> "dict | None":
    try:
        obj = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def record_to_row(r: dict, ctx: dict) -> tuple:
    metric = (
        f"pp{r['n_prompt']}_tokens_per_sec"
        if r.get("n_prompt")
        else f"tg{r['n_gen']}_tokens_per_sec"
    )
    # psycopg rejects mixed int/float lists
    raw_samples = r.get("samples_ts") or None
    samples = [float(x) for x in raw_samples] if raw_samples else None
    return (
        r.get("test_time"),
        r["build_commit"],
        ctx["branch"],
        r["build_number"],
        r.get("gpu_info") or r.get("cpu_info") or "",
        r.get("backends") or "",
        r.get("model_type") or r.get("model_filename") or "",
        metric,
        float(r["avg_ts"]),
        float(r.get("stddev_ts") or 0),
        len(samples) if samples else 1,
        samples,
        ctx.get("run_url"),
    )


def ctx_to_row(ctx: dict, metric: str, value: float, stddev: float,
               n_samples: int, samples: "list[float] | None") -> tuple:
    return (
        ctx["time"],
        ctx["commit_sha"],
        ctx["branch"],
        ctx["build_number"],
        ctx["device"],
        ctx["backend"],
        ctx["model"],
        metric,
        value,
        stddev,
        n_samples,
        samples,
        ctx.get("run_url"),
    )


def parse_finetune_loss(text: str) -> "tuple[float, float] | None":
    """(loss, spread) from the last val: segment."""
    for seg in reversed(VAL_SEGMENT_RE.findall(text)):
        m = LOSS_RE.search(seg)
        if m:
            return float(m.group(1)), float(m.group(2) or 0.0)
    return None


def finetune_rows(ctx: dict, stem: Path) -> list:
    """Failed runs get NaN loss and epoch time."""
    loss, spread, epoch_s = float("nan"), 0.0, float("nan")
    if ctx.get("status") != "success":
        log("db", f"{stem.name}: status={ctx.get('status')!r}; recording NaN loss/epoch_s")
    else:
        stdout_path = sibling(stem, ".stdout")
        # stdout can contain GDB dumps with junk bytes
        text = (stdout_path.read_text(encoding="utf-8", errors="replace")
                if stdout_path.is_file() else "")
        parsed = parse_finetune_loss(text)
        if parsed:
            loss, spread = parsed
        else:
            log("db", f"{stem.name}: no val: loss in stdout; recording NaN loss")
        m = None
        for m in WALL_CLOCK_RE.finditer(text):
            pass
        if m:
            epoch_s = float(m.group(1))
        else:
            log("db", f"{stem.name}: no wall clock in stdout; recording NaN epoch_s")
    return [
        ctx_to_row(ctx, "finetune_val_loss", loss, spread, 1, None),
        ctx_to_row(ctx, "finetune_epoch_s", epoch_s, 0.0, 1, None),
    ]


def mtmd_rows(ctx: dict, stem: Path) -> list:
    if ctx.get("status") != "success":
        log("db", f"{stem.name}: status={ctx.get('status')!r}; skipping")
        return []
    stdout_path = sibling(stem, ".stdout")
    if not stdout_path.is_file():
        log("db", f"{stem.name}: missing stdout; skipping")
        return []
    pp_samples: list = []
    tg_samples: list = []
    for rec in iter_records(stdout_path):
        t0, tn = _num(rec.get("t0_wall_t")), _num(rec.get("tN_wall_t"))
        n_predict = _num(rec.get("n_predict"))
        if t0 is None or tn is None:
            continue
        pp_samples.append(t0)
        if n_predict and tn - t0 > 0:
            tg_samples.append(n_predict / (tn - t0))
        else:
            log("db", f"{stem.name}: rep {rec.get('rep')} has no decode span; dropped from tg")
    rows = []
    for metric, samples in (("mtmd_pp_wall_s", pp_samples),
                            ("mtmd_tg_tokens_per_sec", tg_samples)):
        if not samples:
            log("db", f"{stem.name}: no samples for {metric}")
            continue
        sd = statistics.stdev(samples) if len(samples) >= 2 else 0.0
        rows.append(ctx_to_row(ctx, metric, statistics.mean(samples), sd,
                               len(samples), samples))
    return rows


def context_from_status(status: dict) -> "dict | None":
    ref = status.get("ref") or ""
    branch = normalize_branch_label(status.get("branch"))
    if not ref or not branch:
        return None
    device = ""
    gpus = status.get("gpus") or []
    if gpus and isinstance(gpus[0], str):
        device = gpus[0].split(" - ")[0].strip()
    return {
        "commit_sha": ref[:9],
        "branch": branch,
        "backend": normalize_backend(status.get("backend")),
        "model": status.get("model") or "",
        "device": device,
        "build_number": build_number_from_label(branch) or 0,
        "time": status.get("date"),
        "run_url": status.get("run_url") or status.get("repo"),
        "status": status.get("status") or "",
    }


def collect_run_rows(stem: Path, run_url: "str | None" = None) -> list:
    """Rows for one run; stem is the result path without suffix.

    run_url, when given, replaces the one from the .status file.
    """
    status = load_json(sibling(stem, ".status"))
    if status is None:
        return []
    ctx = context_from_status(status)
    if ctx is None:
        return []
    if run_url:
        ctx["run_url"] = run_url

    tool = tool_from_stem(stem.name)
    if tool == "llama-finetune-lora":
        return finetune_rows(ctx, stem)
    if tool == "llama-mtmd-cli":
        return mtmd_rows(ctx, stem)
    if tool != "llama-bench" or ctx["status"] != "success":
        return []
    stdout_path = sibling(stem, ".stdout")
    if not stdout_path.is_file():
        return []
    rows: list = []
    for rec in iter_records(stdout_path):
        try:
            rows.append(record_to_row(rec, ctx))
        except (KeyError, TypeError, ValueError):
            pass
    return rows


class DbPreflightError(Exception):
    """A requested database push cannot work."""


class DbPusher:
    """Best-effort upserts: push errors are logged, never raised."""

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    def ping(self) -> None:
        """Raise DbPreflightError if the database is unreachable."""
        import psycopg  # ty: ignore[unresolved-import]

        try:
            with psycopg.connect(self.db_url, connect_timeout=10) as conn:
                conn.execute("SELECT 1 FROM benchmark_results LIMIT 1")
        except Exception as e:  # noqa: BLE001
            raise DbPreflightError(f"cannot reach database: {e}") from e
        log("db", "database reachable")

    def push_run(self, stem: Path, run_url: "str | None" = None) -> bool:
        try:
            rows = collect_run_rows(stem, run_url)
            if not rows:
                return True
            import psycopg  # ty: ignore[unresolved-import]

            with psycopg.connect(self.db_url, connect_timeout=10) as conn:
                with conn.cursor() as cursor:
                    cursor.executemany(BENCH_UPSERT_SQL, rows)
        except Exception as e:  # noqa: BLE001
            log("db", f"push failed for {stem.name}: {e}")
            return False
        log("db", f"pushed {stem.name}: bench={len(rows)}")
        return True

    def push_results_dir(self, results_dir: Path, pattern: str = "*",
                         run_url: "str | None" = None) -> int:
        """Push every run with a .status file under results_dir; returns failures."""
        failures = 0
        for status_path in sorted(results_dir.glob(f"{pattern}.status")):
            stem = status_path.with_name(status_path.name[:-len(".status")])
            if not self.push_run(stem, run_url):
                failures += 1
        return failures


def make_pusher(db_url: "str | None") -> "DbPusher | None":
    if not db_url:
        return None
    try:
        import psycopg  # noqa: F401  # ty: ignore[unresolved-import]
    except ImportError as e:
        raise DbPreflightError(
            "--db-url given but psycopg is not installed "
            "(pip install 'psycopg[binary]')"
        ) from e
    return DbPusher(db_url)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Push finished qvac-bench results from a results directory "
        "to the dashboards database (idempotent; safe to re-run)."
    )
    p.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "bench-workdir" / "results",
        help="qvac-bench results directory (default: bench-workdir/results)",
    )
    p.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL"),
        help="Postgres/TimescaleDB URL (default: $DATABASE_URL)",
    )
    p.add_argument(
        "--glob",
        default="*",
        help="Only push result stems matching this glob, e.g. 'llama-bench-*'",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.db_url:
        sys.exit("db: no database URL (pass --db-url or set $DATABASE_URL)")
    if not args.results_dir.is_dir():
        sys.exit(f"db: not a directory: {args.results_dir}")
    try:
        pusher = make_pusher(args.db_url)
        assert pusher is not None  # args.db_url is non-empty here
        pusher.ping()
    except DbPreflightError as e:
        sys.exit(f"db: {e}")
    failures = pusher.push_results_dir(args.results_dir, args.glob)
    if failures:
        sys.exit(f"db: {failures} run(s) failed to push")


if __name__ == "__main__":
    main()
