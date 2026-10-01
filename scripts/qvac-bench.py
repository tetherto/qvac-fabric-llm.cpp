from abc import ABC, abstractmethod
from itertools import product
from functools import cache
from pathlib import Path
from socket import gethostname
from typing import Any, Sequence
import argparse
import csv
import datetime
import hashlib
import json
import logging
import math
import os
import pandas as pd
import re
import shutil
import subprocess
import sys
import time

from qvac_bench_db import DbPreflightError, make_pusher

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
WORKDIR: Path = REPO_ROOT / "bench-workdir"
RESULTS_DIR: Path = WORKDIR / "results"
MODELS_DIR: Path = WORKDIR / "models"
IMAGES_DIR: Path = WORKDIR / "images"

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")

OptionsType = dict[str, Any]

class Worktree:
    backend_defines: dict[str, list[str]] = {
        "vulkan": ["-DGGML_VULKAN=ON"],
        "opencl": ["-DGGML_OPENCL=ON"],
        "cuda": ["-DGGML_CUDA=ON"],
        "metal": ["-DGGML_METAL=ON"],
        "default": []
    }

    def __init__(self, build: 'Build', sha: str) -> None:
        self.build = build
        self.name = build.name
        self.backend = build.backend
        self.sha = sha
        self.path: Path = WORKDIR / "wt" / sha
        self.build_path: Path = WORKDIR / "wt" / sha / ("build-" + self.backend)

    def exists(self) -> bool:
        return self.path.is_dir()

    def create(self) -> None:
        git(["worktree", "add", "--detach", str(self.path), self.sha])
        wt_sha = git(["-C", str(self.path), "rev-parse", "HEAD"], capture_output=True).stdout.strip().decode()
        if wt_sha != self.sha:
            raise RuntimeError(f"worktree {self.path} is at {wt_sha}, expected {self.sha}")

    def binary_path(self, name: str) -> Path:
        return self.build_path / "bin" / name

    def build_binaries(self, binaries: Sequence[str], rebuild: bool, num_jobs: int) -> None:
        missing_binaries = [binary for binary in binaries if not self.binary_path(binary).is_file()]
        if not rebuild and not missing_binaries:
            log(self.name, f"reusing existing build: {self.build_path} (pass --rebuild to force)")
            return

        if rebuild or not self.build_path.is_dir():
            shutil.rmtree(self.build_path, ignore_errors=True)
            log(self.name, f"cmake configure")
            with (RESULTS_DIR / f"{self.sha}-{self.backend}-cmake.log").open("w") as logfile:
                subprocess.run(
                    [
                        "cmake", "-S", self.path, "-B", self.build_path,
                        "-DCMAKE_BUILD_TYPE=Release",
                        "-DLLAMA_CURL=OFF",
                        "-DLLAMA_BUILD_TESTS=ON",
                        "-DLLAMA_BUILD_EXAMPLES=ON",
                        "-DLLAMA_BUILD_SERVER=ON",
                        "-DLLAMA_BUILD_TOOLS=ON"
                    ] + self.backend_defines[self.backend],
                    stdout=logfile, stderr=subprocess.STDOUT, check=True)
            missing_binaries = list(binaries)
        else:
            log(self.name, f"partially reusing existing build: {self.build_path}, building only missing binaries: {' '.join(missing_binaries)} (pass --rebuild to force)")

        log(self.name, f"cmake build")
        with (RESULTS_DIR / f"{self.sha}-{self.backend}-build.log").open("w") as logfile:
            subprocess.run(
                ["cmake", "--build", self.build_path, "-j", str(num_jobs), "--target"] + missing_binaries,
                stdout=logfile,
                stderr=subprocess.STDOUT,
                check=True
            )

        still_missing_binaries = [binary for binary in binaries if not self.binary_path(binary).is_file()]
        if still_missing_binaries:
            raise RuntimeError(f"build failed, {' '.join(still_missing_binaries)} not found")


class Build:
    def __init__(self, repo: str, branch: str, backend: str, name: str | None = None) -> None:
        self.repo = repo
        self.branch = branch
        self.name = name or f"{branch}-{backend}"
        self.backend = backend
        if self.backend not in Worktree.backend_defines:
            raise ValueError(f"unsupported backend: {self.backend}")

    def create_worktree(self) -> Worktree:
        sha = git_fetch(self.repo, self.branch)
        worktree = Worktree(self, sha)
        if worktree.exists():
            log(self.name, f"reusing existing worktree: {worktree.path}")
        else:
            log(self.name, f"creating worktree: {worktree.path}")
            worktree.create()
        return worktree


class Model:
    def __init__(self, hf_repo: str, file: str, mmproj: str | None = None, tokenizer: str | None = None, options: OptionsType | None = None) -> None:
        self.hf_repo = hf_repo
        self.file = file
        self.mmproj = mmproj
        self.tokenizer = tokenizer
        self.options = options
        if self.options:
            self.name = f"{file} {options}"
        else:
            self.name = f"{file}"

    def _download_file(self, file: str) -> None:
        target = MODELS_DIR / file
        name = Path(file).stem
        if target.is_file():
            log(name, f"reusing cached: {target}")
            return
        log(name, f"downloading from {self.hf_repo}")
        subprocess.run(
            [
                f"{REPO_ROOT}/scripts/hf.sh", "--repo", self.hf_repo, "--file",
                file, "--outdir",
                str(MODELS_DIR)
            ],
            check=True
        )
        if not target.is_file():
            raise RuntimeError(f"downloaded file not at expected path: {target}")

    def download(self) -> None:
        self._download_file(self.file)
        if self.mmproj:
            self._download_file(self.mmproj)

class RunContext:
    def __init__(self, worktree: Worktree, model: Model, options: OptionsType) -> None:
        self.worktree = worktree
        self.model = model
        self.options = options.copy()
        if model.options:
            self.options = self.options | model.options
        # Ignore options that don't affect the resuls of a run
        for ignore in ("restart", "retry_failed", "rebuild", "num_jobs", "clean_unused"):
            self.options.pop(ignore, None)

class Benchmark(ABC):
    def __init__(self, name: str, binaries: Sequence[str], default_options: OptionsType) -> None:
        self.name = name
        self.binaries = binaries
        self.default_options = default_options

    def result_path_stem(self, run_ctx: RunContext) -> Path:
        options_str = json.dumps(run_ctx.options, sort_keys=True)
        options_sha = hashlib.sha256(options_str.encode()).hexdigest()[:10]
        return RESULTS_DIR / f"{self.name}-{run_ctx.worktree.sha}-{run_ctx.worktree.backend}-{run_ctx.model.file}-{options_sha}"

    def result_path(self, run_ctx: RunContext, suffix: str) -> Path:
        stem = self.result_path_stem(run_ctx)
        return stem.with_suffix(stem.suffix + suffix)

    def status_path(self, run_ctx: RunContext) -> Path:
        return self.result_path(run_ctx, ".status")

    def download_assets(self, options: OptionsType) -> None:
        pass

    def clean_results(self, run_ctx: RunContext) -> None:
        stem = self.result_path_stem(run_ctx)
        for f in stem.parent.glob(stem.name + ".*"):
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()

    @abstractmethod
    def run(self, run_ctx: RunContext) -> None:
        pass

    @abstractmethod
    def verify_output(self, run_ctx: RunContext) -> bool:
        pass

    @abstractmethod
    def get_results(self, run_ctx: RunContext) -> pd.DataFrame:
        pass

    @abstractmethod
    def create_report(self, worktrees: Sequence[Worktree], models: Sequence[Model], options: OptionsType, report_options: OptionsType, df: pd.DataFrame) -> None:
        pass



class NoModel(Model):
    def __init__(self) -> None:
        self.file = "none"

    def download(self) -> None:
        pass


def make_env(options: OptionsType) -> dict[str, str]:
    """Environment for a benchmark subprocess"""
    env = os.environ.copy()
    env["GGML_VK_VISIBLE_DEVICES"] = str(options["ggml_vk_visible_devices"])
    if "ggml_cuda_enable_unified_memory" in options:
        # ggml-cuda only checks whether the variable is set, so "0" would still enable it
        if str(options["ggml_cuda_enable_unified_memory"]).strip().lower() not in ("", "0", "false", "no", "off"):
            env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
        else:
            env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
    if "cuda_visible_devices" in options:
        env["CUDA_VISIBLE_DEVICES"] = str(options["cuda_visible_devices"])
    return env


def log(label: str, msg: str) -> None:
    logging.info(f"[{label}] {msg}")


def git(args: Sequence[str | Path], **kwargs):
    return subprocess.run(["git", "-C", REPO_ROOT] + list(args), check=True, **kwargs)


def git_fetch(repo: str, branch: str):
    git(["fetch", "--quiet", repo, branch])
    return git(["rev-parse", "FETCH_HEAD"], capture_output=True).stdout.strip().decode()


def deep_merge(base: dict[Any, Any], override: dict[Any, Any]) -> dict[Any, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def parse_options(items: list[str]) -> dict[str, Any]:
    out = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid option: {item} (expected key:value)")
        v: Any
        k, v = item.split(":", 1)

        v = v.strip()
        if v.isdigit():
            v = int(v)
        else:
            try:
                v = float(v)
            except ValueError:
                pass

        out[k.strip()] = v

    return out


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--config", "-c", help="Path to config JSON file")
    p.add_argument("--benchmark")

    p.add_argument(
        "--model",
        action="append",
        help="Format: JSON model object (repeatable)",
    )

    p.add_argument(
        "--build",
        action="append",
        help="Format: JSON build object (repeatable)",
    )

    p.add_argument(
        "--option",
        action="append",
        help="Format: key:value (repeatable)",
    )

    p.add_argument(
        "--report-option",
        action="append",
        help="Format: key:value (repeatable)",
    )

    p.add_argument("--restart", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument(
        "--num-jobs",
        default=os.cpu_count(),
        type=int,
        help="Number of parallel jobs for building"
    )
    p.add_argument(
        "--cooldown",
        default=0.0,
        type=float,
        help="Number of seconds to wait between runs"
    )
    p.add_argument(
        "--warmup",
        default=0,
        type=int,
        help="Number of warmup runs before measuring"
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        help="Directory to look for models in (default: bench-workdir/models). "
             "Files already present are reused; only missing ones are downloaded."
    )
    p.add_argument("--clean-unused", action="store_true")
    p.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL"),
        help="Postgres/TimescaleDB URL to push results to as they complete "
             "(default: $DATABASE_URL; no push when unset)"
    )

    return p


def apply_overrides(config, args):
    override = {}

    if args.benchmark:
        override["benchmark"] = args.benchmark

    if args.model:
        for model_str in args.model:
            override.setdefault("models", {}).append(json.loads(model_str))

    if args.build:
        for build_str in args.build:
            override.setdefault("builds", {}).append(json.loads(build_str))

    if args.option:
        override["options"] = parse_options(args.option)

    if args.report_option:
        override["report_options"] = parse_options(args.report_option)

    override.setdefault("options", {})
    override["options"]["restart"] = args.restart
    override["options"]["retry_failed"] = args.retry_failed
    override["options"]["rebuild"] = args.rebuild
    override["options"]["num_jobs"] = args.num_jobs
    override["options"]["cooldown"] = args.cooldown
    override["options"]["warmup"] = args.warmup
    override["options"]["clean_unused"] = args.clean_unused

    return deep_merge(config, override)

def visible_devices(run_ctx: RunContext) -> str:
    if run_ctx.worktree.backend == "vulkan":
        return str(run_ctx.options.get("ggml_vk_visible_devices", "0"))
    elif run_ctx.worktree.backend == "cuda":
        return str(run_ctx.options.get("cuda_visible_devices", "0"))
    return "0"

@cache
def allvulkaninfo() -> list[str]:
    try:
        vi = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ["vulkaninfo unavailable or failed"]

    try:
        devices = []
        current = None
        for line in vi.splitlines():
            line = line.strip()

            if not line:
                continue

            m = re.match(r"^(GPU\d+):$", line)
            if m:
                current = {"id": m.group(1)}
                devices.append(current)
                continue

            if current and "=" in line:
                key, value = map(str.strip, line.split("=", 1))
                current[key] = value

        return [f"{device.get('deviceName', 'Unknown')} - {device.get('driverName', 'Unknown')} {device.get('driverInfo', 'Unknown')}" for device in devices]

    except Exception:
        return ["vulkaninfo summary parsing failed"]


@cache
def allopenclinfo() -> list[str]:
    try:
        ci = subprocess.run(["clinfo"], capture_output=True, text=True, check=True).stdout
    except Exception:
        return ["clinfo unavailable or failed"]

    try:
        devices = []
        current = None
        for line in ci.splitlines():
            match = re.match(r"^\s*Device Name\s{2,}(.+?)\s*$", line)
            if match:
                current = {"name": match.group(1)}
                devices.append(current)
                continue

            if current:
                match = re.match(r"^\s*(Driver Version|Device Version)\s{2,}(.+?)\s*$", line)
                if match:
                    current[match.group(1)] = match.group(2)

        if not devices:
            return ["clinfo found no devices"]
        return [f"{device['name']} - {device.get('Device Version', 'Unknown')} {device.get('Driver Version', 'Unknown')}" for device in devices]
    except Exception:
        return ["clinfo parsing failed"]


def allgpuinfo() -> list[str]:
    return allvulkaninfo() + allopenclinfo()


def gpuinfo(worktrees, models, options) -> list[tuple[int, str, str]]:
    def split_devices(device_str: object) -> list[int]:
        # parse_options() coerces numeric --option values to int, so this may not be a str;
        # every other consumer of this option already wraps it in str().
        return [int(x) for x in str(device_str).split(",") if x.strip().isdigit()]

    visible_devices = set()

    for model in models:
        run_ctx = RunContext(None, model, options)
        for d in split_devices(run_ctx.options.get("ggml_vk_visible_devices", "")):
            visible_devices.add(d)

    backends = {worktree.backend for worktree in worktrees}
    api_devices = []
    if "opencl" in backends:
        api_devices.append(("OpenCL", allopenclinfo()))
        visible_devices.add(0)
    if "vulkan" in backends:
        api_devices.append(("Vulkan", allvulkaninfo()))

    return [(d, api, devices[d] if d < len(devices) else "Unknown")
            for d in sorted(visible_devices) for api, devices in api_devices]


def get_status(statuspath) -> str | None:
    try:
        with statuspath.open("r") as f:
            j = json.load(f)
            return j.get("status", None)
    except FileNotFoundError:
        return None

class LlamaBench(Benchmark):
    def __init__(self) -> None:
        super().__init__(
            "llama-bench",
            ["llama-bench"],
            {
                "reps": 5,
                "fa": "0",
                "ngl": 99,
                "ggml_vk_visible_devices": "0"
            }
        )

    def run(self, run_ctx: RunContext) -> None:
        model_path = MODELS_DIR / run_ctx.model.file

        outpath = self.result_path(run_ctx, ".stdout")
        errpath = self.result_path(run_ctx, ".stderr")

        log("llama-bench", f"storing results in: {outpath.stem}.*")

        env = make_env(run_ctx.options)

        with outpath.open("w") as out, errpath.open("w") as err:
            subprocess.run(
                [
                    run_ctx.worktree.binary_path(self.binaries[0]), "-m", str(model_path), "-ngl", str(run_ctx.options["ngl"]),
                    "-r", str(run_ctx.options["reps"]), "-fa", str(run_ctx.options["fa"]), "-o", "jsonl"
                ],
                stdout=out,
                stderr=err,
                env=env,
                check=True
            )

    def verify_output(self, run_ctx: RunContext) -> bool:
        outpath = self.result_path(run_ctx, ".stdout")
        with outpath.open("r") as f:
            for l in f.readlines():
                data = json.loads(l)
                if "avg_ts" not in data:
                    return False
        return True

    def get_results(self, run_ctx: RunContext) -> pd.DataFrame:
        def tps(row: dict[str, str]) -> float:
            return float(row.get("avg_ts", 0.0))

        def sd(row: dict[str, str]) -> float:
            return float(row.get("stddev_ts", 0.0))

        def metric_label(row: dict[str, str]) -> str:
            n_prompt = int(row.get("n_prompt", 0))
            n_gen = int(row.get("n_gen", 0))
            if n_prompt and not n_gen:
                return f"pp{n_prompt}"
            if n_gen and not n_prompt:
                return f"tg{n_gen}"
            return f"pp{n_prompt}+tg{n_gen}"

        statuspath = self.status_path(run_ctx)
        if get_status(statuspath) != "success":
            return pd.DataFrame()

        outpath = self.result_path(run_ctx, ".stdout")
        with outpath.open("r") as f:
            rows = []
            for line in f:
                row = json.loads(line)
                label = metric_label(row)
                rows.append({"model": run_ctx.model.name, "metric": label, "worktree": run_ctx.worktree.name, "tps": tps(row), "sd": sd(row)})

            return pd.DataFrame(rows)

    def create_report(self, worktrees: Sequence[Worktree], models: Sequence[Model], options: OptionsType, report_options: OptionsType, df: pd.DataFrame) -> None:
        reference = report_options.get("reference", worktrees[0].name)
        lines = []
        lines.append(f"# Vulkan benchmark: {', '.join(f'`{worktree.name}`' for worktree in worktrees)}")
        lines.append("")
        lines.append(f"- **Date**: {datetime.datetime.now().isoformat()}")
        lines.append(f"- **Host**: {gethostname()}")
        for i, api, gpu in gpuinfo(worktrees, models, options):
            lines.append(f"- **GPU{i} ({api})**: {gpu}")
        for worktree in worktrees:
            lines.append(f"- **Ref `{worktree.name}`**: `{worktree.build.repo}` `{worktree.build.branch}` @ `{worktree.sha[:12]}`")
        lines.append(f"- **Deltas vs {reference}**")
        lines.append(f"- **llama-bench**: `-ngl {options['ngl']} -r {options['reps']} -fa {options['fa']} -o jsonl` (default pp512 + tg128)")
        lines.append("")

        df = df.reindex(columns=["model", "metric", "worktree", "tps", "sd"])
        rows = []
        for (model, metric), group in df.groupby(["model", "metric"], sort=False):
            # The reference may have no row here, e.g. if its run failed for this model.
            ref_values = group.loc[group.worktree == reference, "tps"]
            ref = ref_values.iloc[0] if not ref_values.empty else None
            row = { "Model": model, "Metric": metric, }
            for _, r in group.iterrows():
                value = f"{r.tps:.2f} ±{r.sd:.2f}"
                if r.worktree != reference and ref is not None and ref != 0.0:
                    pct = (r.tps - ref) / ref * 100
                    value += f" ({pct:+.2f}%)"
                row[f"{r.worktree} t/s"] = value
            rows.append(row)

        report_path = RESULTS_DIR / f"{self.name}-report.md"
        with report_path.open("w") as f:
            f.write("\n".join(lines))
            f.write("\n")
            f.write(pd.DataFrame(rows).to_markdown(index=False, tablefmt="github"))
            f.write("\n")

        print(f"wrote {report_path}", file=sys.stderr)

class LlamaFinetuneLora(Benchmark):
    loss_re = re.compile(r"loss\s*=?\s*(([0-9.eE+-]+)|inf|nan|-inf)")
    wall_clock_sec_re = re.compile(r"qvac_bench_wall_clock_sec:\s*([0-9.]+)")

    @staticmethod
    def parse_regex(regex, f) -> str | None:
        for line in reversed(f.readlines()):
            match = regex.search(line)
            if match:
                return match.group(1)
        return None

    def __init__(self) -> None:
        super().__init__(
            "llama-finetune-lora",
            ["llama-finetune-lora"],
            {
                "dataset": str(REPO_ROOT / "scripts" / "biomed.jsonl"),
                "ctx": 128,
                "batch": 128,
                "ubatch": 128,
                "ngl": 99,
                "lr": 1e-5,
                "lr_min": 1e-8,
                "lr_sched": "cosine",
                "warmup_ratio": 0.1,
                "lora_modules": "attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down",
                "fa": "off",
                "ggml_vk_visible_devices": "0"
            }
        )

    def run(self, run_ctx: RunContext) -> None:
        model_path = MODELS_DIR / run_ctx.model.file

        outpath = self.result_path(run_ctx, ".stdout")

        log("llama-finetune-lora", f"storing results in: {outpath.stem}.*")

        env = make_env(run_ctx.options)

        with outpath.open("w") as out:
            start = time.perf_counter()
            subprocess.run(
                [
                    run_ctx.worktree.binary_path(self.binaries[0]),
                    "-m", model_path,
                    "-f", run_ctx.options["dataset"],
                    "--assistant-loss-only",
                    "-c", str(run_ctx.options["ctx"]),
                    "-b", str(run_ctx.options["batch"]),
                    "-ub", str(run_ctx.options["ubatch"]),
                    "-ngl", str(run_ctx.options["ngl"]),
                    "-fa", str(run_ctx.options["fa"]),
                    "--checkpoint-save-steps", "0",
                    "--learning-rate", str(run_ctx.options["lr"]),
                    "--lr-min", str(run_ctx.options["lr_min"]),
                    "--lr-scheduler", str(run_ctx.options["lr_sched"]),
                    "--warmup-ratio", str(run_ctx.options["warmup_ratio"]),
                    "--num-epochs", "1",
                    "--lora-modules", run_ctx.options["lora_modules"]
                ],
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                check=True
            )
            elapsed = time.perf_counter() - start
            out.writelines(["qvac_bench_wall_clock_sec: " + str(elapsed)])

    def verify_output(self, run_ctx: RunContext) -> bool:
        outpath = self.result_path(run_ctx, ".stdout")
        with outpath.open("r") as f:
            if LlamaFinetuneLora.parse_regex(LlamaFinetuneLora.loss_re, f) is None:
                return False
            f.seek(0)
            if LlamaFinetuneLora.parse_regex(LlamaFinetuneLora.wall_clock_sec_re, f) is None:
                return False
        return True

    def get_results(self, run_ctx: RunContext) -> pd.DataFrame:
        if get_status(self.status_path(run_ctx)) != "success":
            return pd.DataFrame()

        outpath = self.result_path(run_ctx, ".stdout")
        with outpath.open("r") as f:
            loss = LlamaFinetuneLora.parse_regex(LlamaFinetuneLora.loss_re, f)
            f.seek(0)
            epoch_s = LlamaFinetuneLora.parse_regex(LlamaFinetuneLora.wall_clock_sec_re, f)

        return pd.DataFrame([{
            "model": run_ctx.model.name,
            "worktree": run_ctx.worktree.name,
            "loss": float(loss) if loss is not None else float("nan"),
            "epoch_s": float(epoch_s) if epoch_s is not None else float("nan"),
        }])

    def create_report(self, worktrees: Sequence[Worktree], models: Sequence[Model], options: OptionsType, report_options: OptionsType, df: pd.DataFrame) -> None:
        reference = report_options.get("reference", worktrees[0].name)
        loss_tol = report_options.get("loss_tolerance", 0.1)
        df = df.reindex(columns=["model", "worktree", "loss", "epoch_s"])

        lines = []
        lines.append(f"# LoRA finetune regression: {', '.join(f'`{worktree.name}`' for worktree in worktrees)}")
        lines.append("")
        lines.append(f"- **Date**: {datetime.datetime.now().isoformat()}")
        lines.append(f"- **Host**: {gethostname()}")
        for i, api, gpu in gpuinfo(worktrees, models, options):
            lines.append(f"- **GPU{i} ({api})**: {gpu}")
        for worktree in worktrees:
            lines.append(f"- **Ref `{worktree.name}`**: `{worktree.build.repo}` `{worktree.build.branch}` @ `{worktree.sha[:12]}`")
        lines.append(f"- **Loss tolerance (vs {reference})**: ±{loss_tol*100:.1f}%")
        lines.append("")

        header = ["Model"] + [f"{wt.name} loss" for wt in worktrees] + [f"{wt.name} epoch_s" for wt in worktrees] + \
                 [f"{wt.name} verdict (loss Δ%)" for wt in worktrees if wt.name != reference]
        align  = ["---"] + ["---:"] * len(worktrees) + ["---:"] * len(worktrees) + ["---"] * len([wt for wt in worktrees if wt.name != reference])
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(align) + "|")

        for model in models:
            row = f"| {model.name} |"
            metrics_loss = {}
            metrics_time = {}
            for worktree in worktrees:
                result = df[(df["model"] == model.name) & (df["worktree"] == worktree.name)]
                metrics_loss[worktree.name] = result["loss"].iloc[0] if not result.empty else float("nan")
                metrics_time[worktree.name] = result["epoch_s"].iloc[0] if not result.empty else float("nan")

            for worktree in worktrees:
                row += f" {metrics_loss[worktree.name]:.4f} |"

            for worktree in worktrees:
                row += f" {metrics_time[worktree.name]:.4f} |"

            for worktree in worktrees:
                if worktree.name == reference:
                    continue

                if math.isnan(metrics_loss[reference]) or math.isnan(metrics_loss[worktree.name]):
                    row += " FAIL |"
                else:
                    delta_pct = (metrics_loss[worktree.name] - metrics_loss[reference]) / metrics_loss[reference] * 100.0
                    if abs(delta_pct) <= loss_tol * 100.0:
                        row += f" PASS ({delta_pct:.2f}%) |"
                    else:
                        row += f" DIVERGE ({delta_pct:.2f}%) |"

            lines.append(row)


        report_path = RESULTS_DIR / f"{self.name}-report.md"
        with report_path.open("w") as f:
            f.write("\n".join(lines))

        print(f"wrote {report_path}", file=sys.stderr)


class LlamaMtmdCli(Benchmark):
    def __init__(self) -> None:
        super().__init__(
            "llama-mtmd-cli",
            ["llama-mtmd-cli"],
            {
                "reps": 5,
                "ngl": 99,
                "n_predict": 512,
                "jinja": True,
                "prompt": "Describe this image in one sentence",
                "ggml_vk_visible_devices": "0",
                "image_url": "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/media/llama0-banner.png"
            }
        )

    @staticmethod
    def _image_path(url: str) -> Path:
        url_sha = hashlib.sha256(url.encode()).hexdigest()[:10]
        return IMAGES_DIR / f"sample-{url_sha}.png"

    def download_assets(self, options: OptionsType) -> None:
        url = options["image_url"]
        log("llama-mtmd-cli", f"downloading sample image: {url}")
        path = LlamaMtmdCli._image_path(url)
        if path.is_file():
            log("llama-mtmd-cli", f"reusing cached sample image: {path}")
            return
        subprocess.run(["curl", "-fsSL", url, "-o", path], check=True)
        log("llama-mtmd-cli", f"downloaded sample image: {path}")

    def run(self, run_ctx: RunContext) -> None:
        model_path = MODELS_DIR / run_ctx.model.file
        if run_ctx.model.mmproj is None:
            raise ValueError(f"model {run_ctx.model.file} does not have an associated .mmproj file")
        mmproj_path = MODELS_DIR / run_ctx.model.mmproj

        outpath = self.result_path(run_ctx, ".stdout")
        errpath = self.result_path(run_ctx, ".stderr")

        log("llama-mtmd-cli", f"storing results in: {errpath.stem}.*")

        def run_n_predict(n_predict, err):
            env = make_env(run_ctx.options)
            subprocess.run(
                [
                    run_ctx.worktree.binary_path(self.binaries[0]),
                    "-m", model_path,
                    "--mmproj", mmproj_path,
                    "--image", LlamaMtmdCli._image_path(run_ctx.options["image_url"]),
                    "-p", run_ctx.options["prompt"],
                    "-ngl", str(run_ctx.options["ngl"]),
                    "-n", str(n_predict),
                    "--ignore-eos"
                ] + (["--jinja"] if run_ctx.options.get("jinja", True) else []),
                stdout=subprocess.DEVNULL,
                stderr=err,
                env=env,
                check=True
            )

        results = []
        with errpath.open("w") as err:
            for rep in range(run_ctx.options["reps"]):
                start_t0 = time.perf_counter()
                run_n_predict(0, err)
                elapsed_t0 = time.perf_counter() - start_t0

                start_tN = time.perf_counter()
                run_n_predict(run_ctx.options["n_predict"], err)
                elapsed_tN = time.perf_counter() - start_tN

                results.append({
                    "ref": run_ctx.worktree.name,
                    "model": run_ctx.model.file,
                    "mmproj": run_ctx.model.mmproj,
                    "rep": rep,
                    "n_predict": run_ctx.options["n_predict"],
                    "t0_wall_t": elapsed_t0,
                    "tN_wall_t": elapsed_tN
                    })

        with outpath.open("w") as out:
            json.dump(results, out, indent=2)

    def verify_output(self, run_ctx: RunContext) -> bool:
        outpath = self.result_path(run_ctx, ".stdout")
        try:
            with outpath.open("r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return False
            for row in data:
                if not all(k in row for k in ("n_predict", "t0_wall_t", "tN_wall_t")):
                    return False
            return True
        except Exception:
            return False

    def get_results(self, run_ctx: RunContext) -> pd.DataFrame:
        if get_status(self.status_path(run_ctx)) != "success":
            return pd.DataFrame()

        outpath = self.result_path(run_ctx, ".stdout")
        with outpath.open("r") as f:
            data = json.load(f)

        rows = []
        for result in data:
            rows.append({
                "model": run_ctx.model.name,
                "worktree": run_ctx.worktree.name,
                "metric": "pp wall (s, lower=better)",
                "value": result["t0_wall_t"],
            })
            rows.append({
                "model": run_ctx.model.name,
                "worktree": run_ctx.worktree.name,
                "metric": "tg (decode, t/s, higher=better)",
                "value": result["n_predict"] / (result["tN_wall_t"] - result["t0_wall_t"]),
            })
        return pd.DataFrame(rows)

    def create_report(self, worktrees: Sequence[Worktree], models: Sequence[Model], options: OptionsType, report_options: OptionsType, df: pd.DataFrame) -> None:
        reference = report_options.get("reference", worktrees[0].name)
        df = df.reindex(columns=["model", "worktree", "metric", "value"])

        lines = []
        lines.append(f"# Vulkan multimodal benchmark: {', '.join(f'`{wt.name}`' for wt in worktrees)}")
        lines.append("")
        lines.append(f"- **Date**: {datetime.datetime.now().isoformat()}")
        lines.append(f"- **Host**: {gethostname()}")
        for i, api, gpu in gpuinfo(worktrees, models, options):
            lines.append(f"- **GPU{i} ({api})**: {gpu}")
        for worktree in worktrees:
            lines.append(f"- **Ref `{worktree.name}`**: `{worktree.build.repo}` `{worktree.build.branch}` @ `{worktree.sha[:12]}`")
        lines.append(f"- **Deltas vs {reference}**")
        lines.append(f"- **llama-mtmd-cli**: `-ngl {options['ngl']} -n {options['n_predict']}`, reps={options['reps']}")
        lines.append("")

        header_cells = ["Model", "Metric"] + [f"{wt.name}" for wt in worktrees]
        align_cells  = ["---", "---"] + ["---:"] * len(worktrees)
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "|".join(align_cells) + "|")

        for model in models:
            # pp_wall_s = wallclock for -n 0 (load + image-encode + prompt-eval), lower is better.
            # tg_tps    = N / (T1 - T0), higher is better.
            # Report mean ±sd across reps.

            metrics = ("pp wall (s, lower=better)", "tg (decode, t/s, higher=better)")
            for metric_name in metrics:
                row = [model.name, metric_name]
                ref_values = df[(df["model"] == model.name) & (df["metric"] == metric_name) & (df["worktree"] == reference)]["value"]
                ref_mean = ref_values.mean() if not ref_values.empty else None
                for worktree in worktrees:
                    values = df[(df["model"] == model.name) & (df["metric"] == metric_name) & (df["worktree"] == worktree.name)]["value"]
                    if values.empty:
                        row.append("-")
                    else:
                        mean = values.mean()
                        sd = values.std() if len(values) > 1 else 0.0
                        cell = f"{mean:.2f} ±{sd:.2f}"
                        if worktree.name != reference and ref_mean is not None and ref_mean != 0.0:
                            delta_pct = (mean - ref_mean) / ref_mean * 100.0
                            cell += f" ({delta_pct:+.2f}%)"
                        row.append(cell)
                lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        lines.append("Raw json outputs are in this directory next to this report.")

        report_path = RESULTS_DIR / f"{self.name}-report.md"
        with report_path.open("w") as f:
            f.write("\n".join(lines))

        print(f"wrote {report_path}", file=sys.stderr)

class Turboquant(Benchmark):
    def __init__(self) -> None:
        super().__init__(
            "turboquant",
            ["llama-bench", "llama-perplexity", "llama-server", "llama-completion"],
            {
                "benchmarks": ["perf", "perp", "ruler", "longbench", "zeroscrolls", "leval", "niah"],
                "perf_force_coopmat": "",
                "perf_args": ["-c", "large", "--pp-sizes", "2k,8k", "--skip-scalar"],
                "perp_args": ["-c", "mid"],
                "ruler_args": ["-c", "mid"],
                "longbench_args": ["-c", "mid"],
                "zeroscrolls_args": ["-c", "small"],
                "leval_args": ["-c", "small"],
                "niah_args": ["-c", "full"],
                "ggml_vk_visible_devices": "0",
            }
        )

    @staticmethod
    def _find_latest_csv(bench_dir: Path, glob_pattern: str) -> "Path | None":
        """Return the lexicographically latest file matching *glob_pattern* inside *bench_dir*.

        Because all combined CSVs embed a ``YYYYMMDD_HHMMSS`` timestamp in their
        name, lexicographic order equals chronological order.
        """
        if not bench_dir.is_dir():
            return None
        candidates = sorted(bench_dir.glob(glob_pattern))
        return candidates[-1] if candidates else None

    @staticmethod
    def _load_eval_csv(bench_dir: Path, bench: str) -> list[dict]:
        if bench == "ruler":
            glob = "kv-ruler_main_*.csv"
        else:
            glob = f"kv-{bench}_*.csv"

        csv_path = Turboquant._find_latest_csv(bench_dir, glob)
        if csv_path is not None:
            try:
                with csv_path.open(newline="", encoding="utf-8") as f:
                    return list(csv.DictReader(f))
            except Exception:
                pass

        return []

    def run(self, run_ctx: RunContext) -> None:
        benchmarks = run_ctx.options["benchmarks"]
        model_path = MODELS_DIR / run_ctx.model.file

        outpath = self.result_path(run_ctx, ".stdout")

        log("turboquant", f"storing results in: {outpath.stem}.*")

        env = make_env(run_ctx.options)

        for bench in [b for b in ["perf", "perp"] if b in benchmarks]:
            bench_env = env.copy()
            if bench == "perf" and run_ctx.options["perf_force_coopmat"] != "":
                bench_env["FORCE_COOPMAT"] = str(run_ctx.options["perf_force_coopmat"])
            bench_env["MODEL_DIR"] = str(MODELS_DIR)
            bench_env["MODEL_NAME"] = str(run_ctx.model.file)

            bench_path = self.result_path(run_ctx, f".{bench}")
            with outpath.open("a") as out:
                subprocess.run(
                    ["bash", REPO_ROOT / "tests" / f"test-kv-cache-quantization-{bench}.sh", "--csv", bench_path] + run_ctx.options[f"{bench}_args"] + [run_ctx.worktree.build_path],
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    env=bench_env,
                    check=True
                )

        # NIAH needs data from RULER
        if "niah" in benchmarks and not "ruler" in benchmarks:
            benchmarks = ["ruler"] + benchmarks

        for bench in [b for b in ["ruler", "longbench", "zeroscrolls", "leval", "niah"] if b in benchmarks]:
            print("running:", bench)
            bench_path = self.result_path(run_ctx, f".{bench}")
            if bench == "ruler":
                extra_args = ["--extra", "--cli-bin", f"{run_ctx.worktree.binary_path('llama-completion')}"]
                if run_ctx.model.tokenizer:
                    extra_args = ["--tokenizers", run_ctx.model.tokenizer] + extra_args
            else:
                extra_args = ["--extra", "--server-bin", f"{run_ctx.worktree.binary_path('llama-server')}"]

            with outpath.open("a") as out:
                subprocess.run(
                    ["python3", REPO_ROOT / "tests" / f"test-kv-cache-{bench}.py", "--output-dir", bench_path,
                     "--models", model_path, "--gpus", visible_devices(run_ctx)] +
                     run_ctx.options[f"{bench}_args"] + extra_args,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=True
                )

    def verify_output(self, run_ctx: RunContext) -> bool:
        if "perf" in run_ctx.options["benchmarks"]:
            bench_path = self.result_path(run_ctx, ".perf")
            try:
                with bench_path.open(encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                    for row in rows:
                        if "model" not in row or "pp_vs_f16_x" not in row or "tg_vs_f16_x" not in row:
                            return False
                return True
            except:
                return False

        if "perp" in run_ctx.options["benchmarks"]:
            bench_path = self.result_path(run_ctx, ".perp")
            try:
                with bench_path.open(encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                    for row in rows:
                        if "model" not in row or "bpw_avg" not in row or "ppl_mean" not in row:
                            return False
                return True
            except:
                return False

        return True

    def get_results(self, run_ctx: RunContext) -> pd.DataFrame:
        if get_status(self.status_path(run_ctx)) != "success":
            return pd.DataFrame()

        results = []
        for bench in run_ctx.options["benchmarks"]:
            if bench in ("perf", "perp"):
                bench_path = self.result_path(run_ctx, f".{bench}")
                bench_df = pd.read_csv(bench_path)
            else:
                bench_path = self.result_path(run_ctx, f".{bench}")
                bench_df = pd.DataFrame(Turboquant._load_eval_csv(bench_path, bench))
            if bench_df.empty:
                continue
            bench_df["model"] = run_ctx.model.name
            bench_df["worktree"] = run_ctx.worktree.name
            bench_df["benchmark"] = bench
            results.append(bench_df)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def create_report_perf(self, worktrees: Sequence[Worktree], models: Sequence[Model], df: pd.DataFrame) -> list[str]:
        lines = []
        for worktree in worktrees:
            for model in models:
                rows = df[(df["benchmark"] == "perf") & (df["model"] == model.name) & (df["worktree"] == worktree.name)]
                if rows.empty:
                    continue
                configs = rows["config"].drop_duplicates().tolist()
                coopmats = rows["coopmat_mode"].drop_duplicates().tolist()
                for config, coopmat in product(configs, coopmats):
                    lines.append(f"### {model.file} - {config} - {coopmat} - {worktree.name}")
                    lines.append("")
                    lines.append("| K | V | BPW | pp | tg |")
                    lines.append("| :---- | :---- | ----: | ----: | ----: |")
                    config_rows = rows[(rows["config"] == config) & (rows["coopmat_mode"] == coopmat)]
                    bpw_scale = 1.0
                    for _, row in config_rows.iterrows():
                        if (row["cache_k"] == "f16" and row["cache_v"] == "f16"):
                            bpw_scale = 16.0 / float(row['kv_size_mib'])
                            lines.append(f"| {row['cache_k']} | {row['cache_v']} | {float(row['kv_size_mib']) * bpw_scale} | {row['pp_avg']} t/s (baseline) | {row['tg_avg']} t/s (baseline) |")
                    for _, row in config_rows.iterrows():
                        if (row["cache_k"] != "f16" or row["cache_v"] != "f16"):
                            lines.append(f"| {row['cache_k']} | {row['cache_v']} | {float(row['kv_size_mib']) * bpw_scale} | {row['pp_vs_f16_x']}x | {row['tg_vs_f16_x']}x |")
                    lines.append("")

        return lines

    def create_report_perp(self, worktrees: Sequence[Worktree], models: Sequence[Model], df: pd.DataFrame) -> list[str]:
        lines = []
        combinations = [("f16", "f16"), ("tbq4_0", "pq4_0"), ("tbq3_0", "pq3_0"), ("pq4_0", "pq4_0"), ("pq3_0", "pq3_0"), ("q4_0", "q4_0")]
        reference = ("f16", "f16", "off")
        for worktree in worktrees:
            lines.append(f"### {worktree.name}")

            header = ["Model", f"{reference[0]}/{reference[1]} (ppl)"]  + [f"{k}/{v}" for k, v in combinations if (k, v) != reference[:2]]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| :---- |" + " ----: |" * len(combinations))

            for model in models:
                rows = df[(df["benchmark"] == "perp") & (df["model"] == model.name) & (df["worktree"] == worktree.name)]

                values = {}
                for _, row in rows.iterrows():
                    values[(row["cache_k"], row["cache_v"], row["norm_correction"])] = float(row["ppl_mean"])

                cells = [model.file]

                for (k,v) in combinations:
                    val = values.get((k, v, "off"), float("nan"))
                    val_nc = values.get((k, v, "on"), float("nan"))
                    if (k,v) == reference[:2]:
                        cells.append(f"{values.get(reference, float('nan')):.2f}")
                    else:
                        if reference in values and not math.isnan(values[reference]):
                            def delta_pct(v):
                                return (v - values[reference]) / values[reference] * 100.0
                            cells.append(f"{delta_pct(val):+.2f}%" + (f" (α: {delta_pct(val_nc):+.2f}%)" if not math.isnan(val_nc) else ""))
                        else:
                            cells.append(f"-")

                lines.append("| " + " | ".join(cells) + " |")

            lines.append("")

        return lines


    def create_report_turboquant_evals(self, worktrees: Sequence[Worktree], models: Sequence[Model], options: OptionsType, df: pd.DataFrame) -> list[str]:
        active_benches = [b for b in ["ruler", "longbench", "zeroscrolls", "leval", "niah"]
                          if b in options["benchmarks"]]
        if not active_benches:
            return []

        eval_configs: list[tuple[str, str]] = [
            ("f16",    "f16"),
            ("q4_0",   "q4_0"),
            ("tbq3_0", "pq3_0"),
            ("tbq4_0", "pq4_0"),
            ("tbq4_0", "q4_0"),
        ]

        eval_bpw: dict[str, float] = {
            "f16":    16.0,
            "q8_0":   8.5,
            "q4_0":   4.5,
            "tbq3_0": 4.25,
            "tbq4_0": 5.25,
            "pq3_0":  3.25,
            "pq4_0":  4.25,
        }

        bench_headers: dict[str, str] = {
            "ruler":       "RULER main %",
            "longbench":   "LongBench Avg",
            "zeroscrolls": "ZS Avg",
            "leval":       "L-Eval Avg",
            "niah":        "NIAH grid %",
        }

        score_col: dict[str, str] = {
            "ruler":       "mean_pct",
            "longbench":   "mean_pct",
            "zeroscrolls": "mean_pct",
            "leval":       "mean_pct",
            "niah":        "score",
        }

        lines: list[str] = []

        for worktree in worktrees:
            for model in models:
                model_label = model.file.removesuffix(".gguf")
                lines.append(f"### {model_label} - {worktree.name}")
                lines.append("")

                col_headers = ["Cache config", "BPW"] + [bench_headers[b] for b in active_benches]
                lines.append("| " + " | ".join(col_headers) + " |")
                lines.append("| :---- | ----: |" + " ----: |" * len(active_benches))

                values: dict[str, dict[tuple[str, str], "float | None"]] = {}
                for bench in active_benches:
                    col = score_col[bench]
                    bench_rows = df[(df["benchmark"] == bench) & (df["model"] == model.name) & (df["worktree"] == worktree.name)]
                    values[bench] = {}
                    for k, v in eval_configs:
                        if bench_rows.empty:
                            matching = pd.Series(dtype=float)
                        else:
                            matching = bench_rows[(bench_rows["cache_k"] == k) & (bench_rows["cache_v"] == v)][col]
                            matching = pd.to_numeric(matching, errors="coerce").dropna()
                        values[bench][(k, v)] = matching.mean() if not matching.empty else None

                for k, v in eval_configs:
                    bpw = (eval_bpw.get(k, 0.0) + eval_bpw.get(v, 0.0)) / 2.0
                    cells = [f"{k}/{v}", f"{bpw:.2f}"]
                    for bench in active_benches:
                        val = values[bench].get((k, v))
                        if val is None:
                            cells.append("-")
                        else:
                            cells.append(f"{val:.1f}")
                    lines.append("| " + " | ".join(cells) + " |")

                lines.append("")

        return lines

    def create_report(self, worktrees: Sequence[Worktree], models: Sequence[Model], options: OptionsType, report_options: OptionsType, df: pd.DataFrame) -> None:
        df = df.reindex(columns=list(df.columns) + [column for column in ("benchmark", "model", "worktree") if column not in df.columns])
        lines = []
        lines.append("# TurboQuant Benchmarks")
        lines.append("")
        lines.append("This report summarizes TurboQuant KV-cache benchmark and quality measurements across various models. Performance values are reported relative to each table's `f16/f16` baseline unless the cell is marked as the baseline.")
        lines.append("")
        lines.append(f"- **Date**: {datetime.datetime.now().isoformat()}")
        lines.append(f"- **Host**: {gethostname()}")
        for i, api, gpu in gpuinfo(worktrees, models, options):
            lines.append(f"- **GPU{i} ({api})**: {gpu}")
        for worktree in worktrees:
            lines.append(f"- **Ref `{worktree.name}`**: `{worktree.build.repo}` `{worktree.build.branch}` @ `{worktree.sha[:12]}`")
        lines.append("")

        if "perf" in options["benchmarks"]:
            lines.append("## Performance Benchmarks")
            lines.append("")
            lines.extend(self.create_report_perf(worktrees, models, df))

        if "perp" in options["benchmarks"]:
            lines.append("## Perplexity Benchmarks")
            lines.append("")
            lines.extend(self.create_report_perp(worktrees, models, df))

        eval_benches = [b for b in ["ruler", "longbench", "zeroscrolls", "leval", "niah"]
                        if b in options["benchmarks"]]
        if eval_benches:
            lines.append("## Cross-Eval Quality Summary")
            lines.append("")
            lines.extend(self.create_report_turboquant_evals(worktrees, models, options, df))

        report_path = RESULTS_DIR / f"{self.name}-report.md"
        with report_path.open("w") as f:
            f.write("\n".join(lines))

        print(f"wrote {report_path}", file=sys.stderr)


benchmarks = [
    LlamaBench(),
    LlamaFinetuneLora(),
    LlamaMtmdCli(),
    Turboquant()
]


def bench_driver(
    benchmark: Benchmark,
    models: Sequence[Model],
    builds: Sequence[Build],
    options: OptionsType,
    report_options: OptionsType,
    db_url: str | None = None,
) -> None:
    WORKDIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)

    options = benchmark.default_options | options

    try:
        pusher = make_pusher(db_url)
        if pusher:
            pusher.ping()
    except DbPreflightError as e:
        sys.exit(f"db: {e}")

    log("driver", f"downloading models for benchmark")
    for model in models:
        model.download()

    log("driver", f"preparing builds for benchmark")
    builds_with_worktrees = []
    for build in builds:
        worktree = build.create_worktree()
        worktree.build_binaries(benchmark.binaries, options["rebuild"], options["num_jobs"])
        builds_with_worktrees.append((build, worktree))

    if options["clean_unused"]:
        for wt in WORKDIR.glob("wt/*"):
            for build_dir in wt.glob("build-*"):
                if build_dir.is_dir() and not any(build_dir == w.build_path for _, w in builds_with_worktrees):
                    log("driver", f"removing unused build directory: {build_dir}")
                    shutil.rmtree(build_dir)

                if not any([b.is_dir() for b in wt.glob("build-*")]):
                    log("driver", f"removing unused worktree directory: {wt}")
                    subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "remove", "--force", str(wt)], check=True)


    for (build, worktree) in builds_with_worktrees:
        for model in models:
            run_ctx = RunContext(worktree, model, options)
            benchmark.download_assets(run_ctx.options)

            statuspath = benchmark.status_path(run_ctx)

            if not options["restart"]:
                status = get_status(statuspath)
                if status == "success" and benchmark.verify_output(run_ctx):
                    log(
                        "driver",
                        f"skipping {benchmark.name} on build {build.name} with model {model.file} (already successful, {statuspath.name})"
                    )
                    continue
                elif status == "failure" and not options["retry_failed"]:
                    log(
                        "driver",
                        f"skipping {benchmark.name} on build {build.name} with model {model.file} (previously failed, use --restart or --retry-failed, {statuspath.name})"
                    )
                    continue

            log("driver", f"cleaning results for {benchmark.name} on build {build.name} with model {model.file}")

            benchmark.clean_results(run_ctx)

            status = {
                "ref": worktree.sha,
                "repo": build.repo,
                "branch": build.branch,
                "backend": build.backend,
                "model": model.file,
                "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "options": run_ctx.options,
                "gpus": allgpuinfo(),
            }
            if os.environ.get("CI_JOB_URL"):
                status["run_url"] = os.environ["CI_JOB_URL"]

            try:
                if run_ctx.options["warmup"] > 0:
                    log("driver", f"performing {run_ctx.options['warmup']} warmup runs for {benchmark.name} on build {build.name} with model {model.file}")
                    for _ in range(run_ctx.options["warmup"]):
                        benchmark.run(run_ctx)
                        benchmark.clean_results(run_ctx)

                log("driver", f"running {benchmark.name} on build {build.name} with model {model.file}")

                benchmark.run(run_ctx)
                if not benchmark.verify_output(run_ctx):
                    raise ValueError(f"invalid benchmark output under {statuspath.stem}.*")
                status["status"] = "success"
            except Exception as e:
                log("driver", f"{benchmark.name} on build {build.name} with model {model.file} failed: {e}")
                status["status"] = "failure"

            with statuspath.open("w") as f:
                json.dump(status, f, indent=2)

            if pusher:
                pusher.push_run(benchmark.result_path_stem(run_ctx))

            time.sleep(run_ctx.options["cooldown"])

    if pusher:
        # Catch-up pass: runs skipped as already successful, plus anything the
        # per-run push missed. Upserts are idempotent.
        pusher.push_results_dir(RESULTS_DIR, f"{benchmark.name}-*")

    dfs = []
    for (build, worktree) in builds_with_worktrees:
        for model in models:
            dfs.append(benchmark.get_results(RunContext(worktree, model, options)))

    benchmark.create_report([w for (_, w) in builds_with_worktrees], models, options, report_options,
                            pd.concat(dfs, ignore_index=True) if len(dfs) > 0 else pd.DataFrame())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.config and not Path(args.config).is_file():
        raise FileNotFoundError(args.config)

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}

    config = apply_overrides(config, args)

    if args.models_dir:
        global MODELS_DIR
        MODELS_DIR = args.models_dir.expanduser().resolve()
        log("driver", f"using models directory: {MODELS_DIR}")

    if "benchmark" not in config:
        raise ValueError("missing required config value: benchmark")
    if "builds" not in config:
        raise ValueError("missing required config value: builds")

    print(json.dumps(config, indent=2))

    benchmark = next((x for x in benchmarks if x.name == config["benchmark"]), None)
    if benchmark is None:
        raise ValueError(f"unknown benchmark: {config['benchmark']}")
    builds = [Build(**b) for b in config["builds"]]
    if "models" in config:
        models = [Model(**m) for m in config["models"]]
    else:
        models = [NoModel()]

    bench_driver(
        benchmark,
        models,
        builds,
        config.get("options", {}),
        config.get("report_options", {}),
        db_url=args.db_url,
    )


if __name__ == "__main__":
    main()
