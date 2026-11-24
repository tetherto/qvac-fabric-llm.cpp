#!/usr/bin/env python3
"""Load-time benchmark harness for the Core ML runtime.

Measures what the Chatterbox Core ML runtime and its generation templates are
tuned from, on the workloads pinned in bench-manifest.json:

  sizes       multifunction spec sizes per stage and width set
  memory      the runtime's available-memory reading under normal load
  load        per function: cold load on a fresh identity, warm load + first
              prediction (expected_warm_load_ms), prediction time, and whether
              loaded functions share weight memory
  ttr         time until every function is ready, with and without the
              admission gate, the worst-case prepare-queue wait, and cached
              startup
  metal_cost  per-variant linear Metal cost constants for
              s3gen_estimate_metal_ms
  synthesis   warm tts-cli synthesis on the benchmark prompt: Metal, Core ML
              with 7 and 3 widths, Core ML with the encoder on Metal

Usage:
  bench-load-time.py --dry-run [--check-models]       validate the manifest, print the plan
  bench-load-time.py build-driver [--out PATH]        compile bench-load-driver
  bench-load-time.py run [MEASUREMENT ...] --tts-cli PATH [--out DIR]   (idle machine, AC power)
  bench-load-time.py report --results DIR [--out doc.md]
  bench-load-time.py smoke                            fresh-host.sh identity + e5rt cache smoke (device)
  bench-load-time.py cleanup                          delete every qvac-bench-* identity and cache

Cold runs use scripts/coreml/fresh-host.sh identities with sidecars cloned to a
fresh path; each identity and the e5rt cache it created are deleted after use.
Before each cold run the harness waits for ANECompilerService to stay below the
manifest's CPU limit. Results go to <out>/results.jsonl, one JSON object per
line; `report` renders them as the measurements doc.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_MANIFEST = HERE / "bench-manifest.json"
FRESH_HOST = HERE / "fresh-host.sh"
DRIVER_SRC = HERE / "bench-load-driver.mm"
DEFAULT_BUILD = REPO / "build-bench"
RUNTIME = REPO / "coreml"
RUNTIME_SOURCES = ("src/hints.cpp", "src/manager.cpp", "src/options.cpp", "src/pin.cpp", "src/scheduler.cpp",
                   "src/sha256.cpp", "src/status.cpp", "src/apple/device.mm")

STAGES = ("encoder", "cfm", "hift")
MEASUREMENTS = ("sizes", "memory", "load", "ttr", "metal_cost", "synthesis")
RUN_KEYS = ("metal", "cold_prebuilt", "cold_generate", "time_to_ready", "cached_startup", "in_memory_ready")
REQUIRED_RUNS = {"metal": 10, "cold_prebuilt": 10, "cold_generate": 10, "time_to_ready": 5, "cached_startup": 5,
                    "in_memory_ready": 5}
UNITS = ("all", "cpu_and_ane", "cpu_and_gpu", "cpu_only")
TTR_ROUTES = ("prebuilt", "generate")
TTR_MODES = ("manager", "manual-gated", "ungated")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# Loaded functions are counted as sharing weights when the extra footprint of
# functions 2..n stays below this fraction of one weight.bin per function.
SHARED_FRACTION = 0.25
# Driver load fields compared per function: +1 grows, -1 shrinks with a load.
SHARING_METRICS = {"footprint": 1, "resident": 1, "wired": 1, "available": -1}


class ManifestError(Exception):
    pass


# ---- small helpers ------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(path: Path) -> str:
    """SHA-256 over the sorted (relative path, file sha256) pairs of a directory."""
    h = hashlib.sha256()
    for p in sorted(q for q in path.rglob("*") if q.is_file()):
        h.update(str(p.relative_to(path)).encode() + b"\0" + sha256_file(p).encode() + b"\n")
    return h.hexdigest()


def tree_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def mad(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = statistics.median(xs)
    return statistics.median(abs(x - m) for x in xs)


def linfit(xs, ys):
    """Least squares y = a + b x; returns (a, b, r2)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        raise ValueError("linfit needs two or more (x, y) pairs")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("linfit needs two distinct x values")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, (1.0 - ss_res / ss_tot) if ss_tot else 1.0


def function_names(stage: str, frames, token_mel_ratio: int):
    """The exporter's function names (export-chatterbox-coreml.py function_name)."""
    if stage == "hift":
        return ["main"]
    if stage == "encoder":
        return [f"tokens_{f // token_mel_ratio}" for f in frames]
    return [f"frames_{f}" for f in frames]


def expand(path: str) -> Path:
    return Path(os.path.expanduser(path))


def models_root(arg: str | None) -> Path:
    return expand(arg or os.environ.get("QVAC_BENCH_MODELS", "~/tether/models"))


def resolve(entry: dict, models: Path, repo: Path = REPO) -> Path:
    return (repo if entry["root"] == "repo" else models) / entry["path"]


# ---- manifest -----------------------------------------------------------------


def load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ManifestError(f"{path}: {e}") from e


def _check_file(errors, where, entry, repo: Path, models: Path | None, check_models: bool):
    if not isinstance(entry, dict):
        errors.append(f"{where}: expected a file entry object")
        return
    for k in ("root", "path", "sha256", "bytes"):
        if k not in entry:
            errors.append(f"{where}: missing '{k}'")
    if errors and any(e.startswith(where + ": missing") for e in errors):
        return
    if entry["root"] not in ("repo", "models"):
        errors.append(f"{where}: root must be 'repo' or 'models'")
        return
    p = entry["path"]
    if not isinstance(p, str) or not p or p.startswith("/") or ".." in Path(p).parts:
        errors.append(f"{where}: path must be relative without '..': {p!r}")
        return
    if not isinstance(entry["sha256"], str) or not SHA256_RE.match(entry["sha256"]):
        errors.append(f"{where}: sha256 must be 64 lowercase hex characters")
        return
    if not isinstance(entry["bytes"], int) or entry["bytes"] <= 0:
        errors.append(f"{where}: bytes must be a positive integer")
        return
    if entry["root"] == "models" and not check_models:
        return
    f = resolve(entry, models or Path("/nonexistent"), repo)
    if not f.is_file():
        errors.append(f"{where}: {f} not found")
        return
    if f.stat().st_size != entry["bytes"]:
        errors.append(f"{where}: {f} is {f.stat().st_size} bytes, pinned {entry['bytes']}")
        return
    got = sha256_file(f)
    if got != entry["sha256"]:
        errors.append(f"{where}: {f} sha256 {got} != pinned {entry['sha256']}")


def _int_list(v):
    return isinstance(v, list) and v and all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in v)


def validate(m: dict, repo: Path = REPO, models: Path | None = None, check_models: bool = False) -> list:
    """Every schema error in the manifest; an empty list means valid."""
    errors: list = []
    if m.get("schema") != 1:
        errors.append("schema must be 1")
    proto = m.get("protocol", {})
    runs = proto.get("runs", {})
    for k in RUN_KEYS:
        if runs.get(k) != REQUIRED_RUNS[k]:
            errors.append(f"protocol.runs.{k} must be {REQUIRED_RUNS[k]} (the published run count), "
                          f"got {runs.get(k)!r}")
    if proto.get("units") not in UNITS:
        errors.append(f"protocol.units must be one of {UNITS}")
    if not isinstance(proto.get("ane_compiler_max_cpu_pct"), (int, float)) or proto["ane_compiler_max_cpu_pct"] <= 0:
        errors.append("protocol.ane_compiler_max_cpu_pct must be a positive number")
    if not isinstance(proto.get("ane_compile_reserve_factor"), (int, float)) or proto["ane_compile_reserve_factor"] <= 0:
        errors.append("protocol.ane_compile_reserve_factor must be a positive number")
    if not isinstance(proto.get("seed"), int):
        errors.append("protocol.seed must be an integer")

    exporter = m.get("exporter", {})
    script = exporter.get("script")
    if not script or not (repo / script).is_file():
        errors.append(f"exporter.script not found: {script!r}")

    sets = m.get("sidecar_sets", {})
    if "w7" not in sets or "w3" not in sets:
        errors.append("sidecar_sets must define w7 and w3 (the 3-vs-7 width comparison)")
    for name, s in sets.items():
        w = f"sidecar_sets.{name}"
        if not _int_list(s.get("frames")):
            errors.append(f"{w}.frames must be a non-empty list of positive integers")
        elif s["frames"] != sorted(set(s["frames"])):
            errors.append(f"{w}.frames must be strictly increasing")
        elif any(f % 2 for f in s["frames"]):
            errors.append(f"{w}.frames must be multiples of the token-mel ratio 2")
        if not isinstance(s.get("dir"), str) or s["dir"].startswith("/") or ".." in Path(s.get("dir", "")).parts:
            errors.append(f"{w}.dir must be a relative path under the models root")
        st = s.get("stages", [])
        if not st or any(x not in STAGES for x in st):
            errors.append(f"{w}.stages must be a non-empty subset of {STAGES}")
        if "hift" not in st:
            src = s.get("hift_from")
            if src not in sets or "hift" not in sets.get(src, {}).get("stages", []):
                errors.append(f"{w}: without a hift stage, hift_from must name a set that has one")
        if not isinstance(s.get("hift_window"), int):
            errors.append(f"{w}.hift_window must be an integer")
    if "w7" in sets and "w3" in sets and _int_list(sets["w7"].get("frames")) and _int_list(sets["w3"].get("frames")):
        if len(sets["w7"]["frames"]) != 7 or len(sets["w3"]["frames"]) != 3:
            errors.append("w7 must have 7 widths and w3 3 widths")
        elif not set(sets["w3"]["frames"]) <= set(sets["w7"]["frames"]):
            errors.append("w3 frames must be a subset of w7 frames")
        if script and (repo / script).is_file():
            default = _exporter_default_frames(repo / script)
            if default is not None and default != sets["w7"]["frames"]:
                errors.append(f"sidecar_sets.w7.frames {sets['w7']['frames']} != exporter DEFAULT_FRAMES {default}")

    variants = m.get("variants", {})
    if set(variants) != {"turbo", "mtl"}:
        errors.append("variants must be exactly turbo and mtl (the two Chatterbox models the runtime serves)")
    for name, v in variants.items():
        w = f"variants.{name}"
        for k in ("t3", "s3gen", "export_source"):
            _check_file(errors, f"{w}.{k}", v.get(k), repo, models, check_models)
        if not isinstance(v.get("sidecar_stem"), str) or not v["sidecar_stem"]:
            errors.append(f"{w}.sidecar_stem must be a non-empty string")
        elif isinstance(v.get("s3gen"), dict) and isinstance(v["s3gen"].get("path"), str):
            stem = _model_stem(Path(v["s3gen"]["path"]).name)
            if stem != v["sidecar_stem"]:
                errors.append(f"{w}.sidecar_stem {v['sidecar_stem']!r} != the engine's stem of the s3gen GGUF {stem!r}")
        if v.get("token_mel_ratio") != 2:
            errors.append(f"{w}.token_mel_ratio must be 2")
        if "language" not in v:
            errors.append(f"{w}.language must be present (null for Turbo)")
        hf = v.get("hf", {})
        if not hf.get("repo") or not re.match(r"^[0-9a-f]{40}$", str(hf.get("revision", ""))):
            errors.append(f"{w}.hf must pin a repo and a 40-hex revision")

    wl = m.get("workloads", {})
    bp = wl.get("benchmark-prompt")
    if not bp:
        errors.append("workloads.benchmark-prompt is missing")
    else:
        _check_file(errors, "workloads.benchmark-prompt.text", bp.get("text"), repo, models, check_models)
        if bp.get("max_sentence_chars") != 180:
            errors.append("workloads.benchmark-prompt.max_sentence_chars must be 180, "
                          "the benchmark's sentence-split length")
        if bp.get("seed") != 42:
            errors.append("workloads.benchmark-prompt.seed must be 42, the synthesis comparison's fixed seed")
    sw = wl.get("length-sweep")
    if not sw:
        errors.append("workloads.length-sweep is missing")
    else:
        warm = sw.get("warmup", "")
        sents = sw.get("sentences", [])
        if not isinstance(warm, str) or not warm.strip():
            errors.append("workloads.length-sweep.warmup must be a sentence")
        if len(sents) < 3:
            errors.append("workloads.length-sweep needs at least 3 sentences for a linear fit")
        for i, s in enumerate(sents):
            if not isinstance(s, str) or _sentence_count(s) != 1:
                errors.append(f"workloads.length-sweep.sentences[{i}] must be exactly one sentence")
            elif len(s) < len(warm):
                errors.append(f"workloads.length-sweep.sentences[{i}] must not be shorter than the warmup")
        if sents and len({len(s) for s in sents}) < 2:
            errors.append("workloads.length-sweep sentences need distinct lengths")

    q = m.get("quality_sets", {})
    if not q.get("wer"):
        errors.append("quality_sets.wer must list at least one set")
    for i, e in enumerate(q.get("wer", [])):
        if not e.get("name") or not e.get("language"):
            errors.append(f"quality_sets.wer[{i}] needs name and language")
        _check_file(errors, f"quality_sets.wer[{i}].audio", e.get("audio"), repo, models, check_models)
        _check_file(errors, f"quality_sets.wer[{i}].reference", e.get("reference"), repo, models, check_models)
    if not q.get("der"):
        errors.append("quality_sets.der must list at least one set")
    for i, e in enumerate(q.get("der", [])):
        if not e.get("name") or not isinstance(e.get("collar_ms"), int):
            errors.append(f"quality_sets.der[{i}] needs name and an integer collar_ms")
        _check_file(errors, f"quality_sets.der[{i}].audio", e.get("audio"), repo, models, check_models)
        _check_file(errors, f"quality_sets.der[{i}].rttm", e.get("rttm"), repo, models, check_models)

    meas = m.get("measurements", {})
    if set(meas) != set(MEASUREMENTS):
        errors.append(f"measurements must be exactly {sorted(MEASUREMENTS)}, got {sorted(meas)}")
    for name, cfg in meas.items():
        w = f"measurements.{name}"
        for v in cfg.get("variants", []):
            if v not in variants:
                errors.append(f"{w}: unknown variant {v!r}")
        for s in cfg.get("sets", []):
            if s not in sets:
                errors.append(f"{w}: unknown sidecar set {s!r}")
        if cfg.get("workload") is not None and cfg["workload"] not in wl:
            errors.append(f"{w}: unknown workload {cfg['workload']!r}")
        if "sharing_units" in cfg and (not cfg["sharing_units"] or any(u not in UNITS for u in cfg["sharing_units"])):
            errors.append(f"{w}.sharing_units must be a non-empty subset of {UNITS}")
        for k in ("runs", "warm_runs", "predictions", "sharing_runs", "samples", "interval_ms"):
            if k in cfg and (not isinstance(cfg[k], int) or cfg[k] < 1):
                errors.append(f"{w}.{k} must be a positive integer")
    ttr = meas.get("ttr", {})
    routes = set()
    gated = ungated = False
    for i, c in enumerate(ttr.get("configs", [])):
        w = f"measurements.ttr.configs[{i}]"
        if c.get("route") not in TTR_ROUTES:
            errors.append(f"{w}.route must be one of {TTR_ROUTES}")
        if c.get("mode") not in TTR_MODES:
            errors.append(f"{w}.mode must be one of {TTR_MODES}")
        if c.get("set") not in sets:
            errors.append(f"{w}.set must name a sidecar set")
        if not isinstance(c.get("runs"), int) or c["runs"] < 1:
            errors.append(f"{w}.runs must be a positive integer")
        elif c.get("route") == "prebuilt" and c.get("set") == "w7" and c["runs"] < runs.get("time_to_ready", 5):
            errors.append(f"{w}: the w7 time-to-ready runs must be at least protocol.runs.time_to_ready")
        routes.add(c.get("route"))
        if c.get("route") == "generate":
            gated |= c.get("mode") == "manager"
            ungated |= c.get("mode") == "ungated"
    if meas.get("ttr") is not None and not (gated and ungated):
        errors.append("measurements.ttr must time the generate route with the gate (manager) and without (ungated)")
    syn = meas.get("synthesis", {})
    names = {c.get("name") for c in syn.get("configs", [])}
    if meas.get("synthesis") is not None and not {"metal", "coreml-w7", "coreml-w3", "coreml-no-encoder"} <= names:
        errors.append("measurements.synthesis needs metal, coreml-w7, coreml-w3 and coreml-no-encoder configs")
    for i, c in enumerate(syn.get("configs", [])):
        if c.get("coreml"):
            if c.get("set") not in sets:
                errors.append(f"measurements.synthesis.configs[{i}].set must name a sidecar set")
            if not c.get("stages") or any(s not in STAGES for s in c["stages"]):
                errors.append(f"measurements.synthesis.configs[{i}].stages must be a subset of {STAGES}")
    return errors


def _model_stem(name: str) -> str:
    # Same rule as export-chatterbox-coreml.py model_stem and coreml_sidecar_path.
    stem = name[:-5] if name.endswith(".gguf") else name
    return re.sub(r"[.-](?:f32|f16|bf16|q8_0|q5_0|q4_0)$", "", stem, flags=re.IGNORECASE)


def _sentence_count(s: str) -> int:
    return len(re.findall(r"[.?!](?:\s|$)", s.strip()))


def _exporter_default_frames(script: Path):
    mt = re.search(r"^DEFAULT_FRAMES\s*=\s*\(([^)]*)\)", script.read_text(), re.M)
    return [int(x) for x in re.findall(r"\d+", mt.group(1))] if mt else None


# ---- the plan -----------------------------------------------------------------


def sidecar(m: dict, models: Path, variant: str, set_name: str, stage: str, kind: str = "mlmodelc") -> Path:
    s = m["sidecar_sets"][set_name]
    if stage not in s["stages"]:
        set_name = s["hift_from"]
        s = m["sidecar_sets"][set_name]
    return models / s["dir"] / f"{m['variants'][variant]['sidecar_stem']}-{stage}.{kind}"


def set_functions(m: dict, variant: str, set_name: str):
    """[(stage, [function, ...])] for a set, HiFT included (from hift_from when needed)."""
    s = m["sidecar_sets"][set_name]
    ratio = m["variants"][variant]["token_mel_ratio"]
    return [(st, function_names(st, s["frames"], ratio)) for st in STAGES]


def plan(m: dict, selected, variants_filter=None) -> list:
    """Human-readable steps, one per process group, for --dry-run."""
    steps = []
    meas = m["measurements"]
    vs = lambda cfg: [v for v in cfg.get("variants", m["variants"]) if not variants_filter or v in variants_filter]
    for name in selected:
        cfg = meas[name]
        if name == "sizes":
            for v in vs(cfg):
                for s in cfg["sets"]:
                    for st in m["sidecar_sets"][s]["stages"]:
                        steps.append((name, f"{v}/{s}/{st}: stat + zlib/LZFSE of model.mil and model.mlmodel"))
        elif name == "memory":
            steps.append((name, f"driver memory --samples {cfg['samples']} --interval-ms {cfg['interval_ms']}"))
        elif name == "load":
            for v in vs(cfg):
                for s in cfg["sets"]:
                    fns = set_functions(m, v, s)
                    n = sum(len(f) for _, f in fns)
                    steps.append((name, f"{v}/{s}: fresh identity; cold load of {n} functions "
                                        f"({', '.join(st for st, _ in fns)}); {cfg['warm_runs']} x {n} warm "
                                        f"processes (load + {cfg['predictions']} predictions); "
                                        f"{cfg['sharing_runs']} weight-sharing runs per multi-function stage and units "
                                        f"{cfg.get('sharing_units', [m['protocol']['units']])}"))
        elif name == "ttr":
            for v in vs(cfg):
                for c in cfg["configs"]:
                    n = sum(len(f) for _, f in set_functions(m, v, c["set"]))
                    extra = f" + {c.get('cached_startup_runs', 0)} cached startup" if c.get("cached_startup_runs") else ""
                    steps.append((name, f"{v}/{c['name']}: {c['runs']} x fresh identity, driver ttr --mode {c['mode']} "
                                        f"--route {c['route']} ({n} functions, set {c['set']}){extra}"))
        elif name == "metal_cost":
            sw = m["workloads"][cfg["workload"]]
            for v in vs(cfg):
                steps.append((name, f"{v}: {cfg['runs']} x {len(sw['sentences'])} tts-cli processes on Metal "
                                    "(CHATTERBOX_COREML_DISABLE=1, CHATTERBOX_CHUNK_PROFILE=1)"))
        elif name == "synthesis":
            for v in vs(cfg):
                for c in cfg["configs"]:
                    how = (f"fresh identity, sidecars {c['set']} {'+'.join(c['stages'])}, 1 priming run"
                           if c.get("coreml") else "Metal, 1 priming run")
                    steps.append((name, f"{v}/{c['name']}: {how}, then {cfg['runs']} warm tts-cli runs "
                                        "on the benchmark prompt"))
    return steps


# ---- process helpers ----------------------------------------------------------


def ane_compiler_cpu() -> float:
    out = subprocess.run(["ps", "-Ao", "pcpu,comm"], capture_output=True, text=True).stdout
    total = 0.0
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and "ANECompilerService" in parts[1]:
            try:
                total += float(parts[0])
            except ValueError:
                pass
    return total


def wait_quiet(proto: dict, enabled: bool) -> float:
    """Waits until ANECompilerService stays below the limit; returns seconds waited."""
    if not enabled:
        return 0.0
    limit = proto["ane_compiler_max_cpu_pct"]
    need = proto.get("quiet_calm_samples", 3)
    deadline = time.monotonic() + proto.get("quiet_timeout_s", 1800)
    t0 = time.monotonic()
    calm = 0
    while calm < need:
        if time.monotonic() > deadline:
            raise RuntimeError(f"ANECompilerService stayed above {limit}% CPU")
        calm = calm + 1 if ane_compiler_cpu() < limit else 0
        if calm < need:
            time.sleep(2)
    return time.monotonic() - t0


def run_json(cmd, env=None, timeout=3600, ok_codes=(0,)) -> dict:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env, timeout=timeout)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode not in ok_codes or not lines:
        raise RuntimeError(f"{' '.join(map(str, cmd[:3]))} ... exited {p.returncode}: {p.stderr.strip()[-2000:]}")
    out = json.loads(lines[-1])
    out["_exit"] = p.returncode
    return out


def clone(src: Path, dst: Path):
    """APFS clone (cp -c) so a fresh path costs no copy; plain copy elsewhere."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["cp", "-c", "-R", str(src), str(dst)], capture_output=True).returncode != 0:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


@contextlib.contextmanager
def fresh_identity(binary: Path, hosts_dir: Path):
    env = dict(os.environ, QVAC_BENCH_HOSTS=str(hosts_dir))
    p = subprocess.run(["bash", str(FRESH_HOST), "new", str(binary)], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"fresh-host.sh new failed: {p.stderr.strip()}")
    info = json.loads(p.stdout.strip().splitlines()[-1])
    try:
        yield info
    finally:
        subprocess.run(["bash", str(FRESH_HOST), "cleanup", info["binary"]], capture_output=True, env=env)


def lzfse_size(path: Path) -> int | None:
    if not shutil.which("compression_tool"):
        return None
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "x.lzfse"
        r = subprocess.run(["compression_tool", "-encode", "-a", "lzfse", "-i", str(path), "-o", str(out)],
                           capture_output=True)
        return out.stat().st_size if r.returncode == 0 and out.exists() else None


# ---- tts-cli output -----------------------------------------------------------

CHUNK_RE = re.compile(r"CHUNK_PROFILE n_total=(\d+) T_mu=(\d+) T_mel=(\d+) enc_ms=([\d.]+) cfm_ms=([\d.]+) "
                      r"hift_ms=([\d.]+) pipeline_ms=([\d.]+) audio_ms=([\d.]+)")
STAGE_LABEL_RE = re.compile(r"\[(encoder|cfm_step\d+|hift_decode)\]\s+[\d.]+ ms \(([^)]+)\)")


def parse_tts_stderr(text: str) -> dict:
    chunks = [dict(n_total=int(g[0]), T_mu=int(g[1]), T_mel=int(g[2]), enc_ms=float(g[3]), cfm_ms=float(g[4]),
                   hift_ms=float(g[5]), pipeline_ms=float(g[6]), audio_ms=float(g[7]))
              for g in CHUNK_RE.findall(text)]
    labels = {"encoder": set(), "cfm": set(), "hift": set()}
    for stage, label in STAGE_LABEL_RE.findall(text):
        key = "cfm" if stage.startswith("cfm") else "hift" if stage.startswith("hift") else "encoder"
        labels[key].add(label)
    ints = lambda pat: [int(x) for x in re.findall(pat, text)]
    peak = re.search(r"(\d+)\s+peak memory footprint", text)
    rss = re.search(r"(\d+)\s+maximum resident set size", text)
    return {
        "chunks": chunks,
        "labels": {k: sorted(v) for k, v in labels.items()},
        "t3_load_ms": ints(r"BENCH: T3_LOAD_MS=(\d+)"),
        "t3_infer_ms": ints(r"BENCH: T3_INFER_MS=(\d+)"),
        "s3gen_load_ms": ints(r"BENCH: S3GEN_LOAD_MS=(\d+)"),
        "s3gen_infer_ms": ints(r"BENCH: S3GEN_INFER_MS=(\d+)"),
        "peak_footprint": int(peak.group(1)) if peak else None,
        "max_rss": int(rss.group(1)) if rss else None,
    }


def check_labels(labels: dict, coreml_stages) -> list:
    """Mismatches between the stage labels a run reported and what it should have run on."""
    bad = []
    for stage in ("encoder", "cfm", "hift"):
        got = labels.get(stage, [])
        if not got:
            bad.append(f"{stage}: no label reported")
        elif stage in coreml_stages and any(not g.startswith("coreml-") for g in got):
            bad.append(f"{stage}: expected Core ML, got {got}")
        elif stage not in coreml_stages and any(g != "ggml" for g in got):
            bad.append(f"{stage}: expected ggml, got {got}")
    return bad


# ---- analysis -----------------------------------------------------------------


def prepare_waits(ttr: dict) -> list:
    """Per prepare job: ms from issue to start (every prepare is requested at issue)."""
    t0 = ttr.get("detail", {}).get("t_issue_ms", 0.0)
    return [s["start_ms"] - t0 for s in ttr.get("timeline", []) if s["kind"] == "prepare"]


def load_gate_waits(ttr: dict) -> list:
    """Per load: ms between its artifact becoming available (its stage's prepare
    ending, or the issue for prebuilt) and the load starting."""
    t0 = ttr.get("detail", {}).get("t_issue_ms", 0.0)
    ready = {s["label"]: s["end_ms"] for s in ttr.get("timeline", []) if s["kind"] == "prepare"}
    out = []
    for s in ttr.get("timeline", []):
        if s["kind"] == "load":
            stage = s["label"].split("/", 1)[0]
            out.append(s["start_ms"] - ready.get(stage, t0))
    return out


def sharing_verdict(increments, weight_bytes: int) -> str:
    """'shared' when functions 2..n add < SHARED_FRACTION of a weight.bin each."""
    if len(increments) < 2 or not weight_bytes:
        return "n/a"
    extra = sum(max(0, x) for x in increments[1:]) / (len(increments) - 1)
    return "shared" if extra < SHARED_FRACTION * weight_bytes else "not shared"


# ---- the runner ---------------------------------------------------------------


class Runner:
    def __init__(self, args, m: dict):
        self.args = args
        self.m = m
        self.proto = m["protocol"]
        self.models = models_root(args.models_root)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.out = Path(args.out or DEFAULT_BUILD / "results" / stamp)
        self.out.mkdir(parents=True, exist_ok=True)
        self.work = self.out / "work"
        self.hosts = self.out / "hosts"
        self.results = self.out / "results.jsonl"
        self.driver = Path(args.driver or DEFAULT_BUILD / "bench-load-driver").resolve()
        self.tts_cli = Path(args.tts_cli).resolve() if args.tts_cli else None
        self.units = args.units or self.proto["units"]
        self.quiet = not args.no_quiet_check
        self.git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip()

    def runs(self, n: int) -> int:
        return self.args.runs or n

    def variants(self, cfg: dict):
        return [v for v in cfg.get("variants", self.m["variants"]) if not self.args.variant or v in self.args.variant]

    def record(self, kind: str, **fields):
        rec = {"kind": kind, "t": datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields}
        with open(self.results, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{kind}] " + json.dumps({k: v for k, v in fields.items() if not isinstance(v, (list, dict))})[:300],
              flush=True)

    def need(self, what: Path, label: str):
        if what is None or not Path(what).exists():
            raise SystemExit(f"bench-load-time: {label} not found: {what}")

    # ---- environment ----
    def environment(self):
        sysctl = lambda k: subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        ident = run_json([self.driver, "identity"]) if self.driver.exists() else {}
        digests = {}
        for v in self.m["variants"]:
            for s in self.m["sidecar_sets"]:
                for st in self.m["sidecar_sets"][s]["stages"]:
                    p = sidecar(self.m, self.models, v, s, st)
                    if p.exists():
                        digests[f"{v}/{s}/{st}"] = tree_digest(p)
        self.record("env", git=self.git, manifest_sha256=sha256_file(self.args.manifest),
                    chip=sysctl("machdep.cpu.brand_string"), memsize=int(sysctl("hw.memsize") or 0),
                    os_build=ident.get("os_build"), os_version=ident.get("os_version"),
                    ac_power="AC Power" in power, units=self.units, sidecars=digests,
                    models_root=str(self.models))
        if "AC Power" not in power and not self.args.allow_battery:
            raise SystemExit("bench-load-time: benchmark runs need AC power (pmset reports battery); "
                             "pass --allow-battery only for a smoke run")

    # ---- sizes ----
    def sizes(self, cfg):
        for v in self.variants(cfg):
            for s in cfg["sets"]:
                for st in self.m["sidecar_sets"][s]["stages"]:
                    mc = sidecar(self.m, self.models, v, s, st)
                    pkg = sidecar(self.m, self.models, v, s, st, "mlpackage")
                    self.need(mc, "compiled sidecar")
                    mil = mc / "model.mil"
                    spec = pkg / "Data" / "com.apple.CoreML" / "model.mlmodel"
                    rec = {"variant": v, "set": s, "stage": st,
                           "functions": len(function_names(st, self.m["sidecar_sets"][s]["frames"],
                                                           self.m["variants"][v]["token_mel_ratio"])),
                           "mil_bytes": mil.stat().st_size,
                           "mil_zlib": len(zlib.compress(mil.read_bytes(), 9)),
                           "mil_lzfse": lzfse_size(mil),
                           "weight_bytes": (mc / "weights" / "weight.bin").stat().st_size,
                           "mlmodelc_bytes": tree_bytes(mc)}
                    if spec.exists():
                        rec.update(mlmodel_bytes=spec.stat().st_size,
                                   mlmodel_zlib=len(zlib.compress(spec.read_bytes(), 9)),
                                   mlmodel_lzfse=lzfse_size(spec))
                    self.record("sizes", **rec)

    # ---- memory ----
    def memory(self, cfg):
        self.need(self.driver, "driver (build-driver)")
        r = run_json([self.driver, "memory", "--samples", cfg["samples"], "--interval-ms", cfg["interval_ms"]],
                     timeout=cfg["samples"] * cfg["interval_ms"] / 1000 + 60)
        avail = [s["available_bytes"] for s in r["samples"]]
        reserves = {}
        f = self.proto["ane_compile_reserve_factor"]
        for v in self.m["variants"]:
            for st in STAGES:
                w = sidecar(self.m, self.models, v, "w7", st) / "weights" / "weight.bin"
                if w.exists():
                    reserves[f"{v}/{st}"] = int(f * w.stat().st_size)
        self.record("memory", memsize=r["memsize_bytes"], low_power=r["low_power"], samples=avail,
                    min=min(avail), median=median(avail), max=max(avail), reserves=reserves,
                    ane_compiler_cpu=ane_compiler_cpu())

    # ---- per-function load ----
    def stage_dir(self, v: str, s: str, stages, kind: str) -> Path:
        """Clones a set's artifacts to a fresh path."""
        d = self.work / f"{v}-{s}-{uuid.uuid4().hex[:8]}"
        for st in stages:
            src = sidecar(self.m, self.models, v, s, st, kind)
            self.need(src, f"{kind} sidecar")
            clone(src, d / src.name)
        return d

    def load(self, cfg):
        self.need(self.driver, "driver (build-driver)")
        for v in self.variants(cfg):
            for s in cfg["sets"]:
                fns = set_functions(self.m, v, s)
                with fresh_identity(self.driver, self.hosts) as ident:
                    d = self.stage_dir(v, s, [st for st, _ in fns], "mlmodelc")
                    waited = wait_quiet(self.proto, self.quiet)
                    host = ident["binary"]
                    common = {"variant": v, "set": s, "identity": ident["id"], "units": self.units}
                    for st, names in fns:
                        model = d / sidecar(self.m, self.models, v, s, st).name
                        r = run_json([host, "load", "--model", model, "--units", self.units, "--predict", "1",
                                      *sum((["--function", n] for n in names), [])])
                        for fr in r["functions"]:
                            self.record("load_cold", **common, stage=st, function=fr["function"],
                                        load_ms=fr["load_ms"], predict_ms=fr.get("predict_ms"),
                                        error=fr.get("error"), quiet_wait_s=waited)
                    for i in range(self.runs(cfg["warm_runs"])):
                        for st, names in fns:
                            model = d / sidecar(self.m, self.models, v, s, st).name
                            for n in names:
                                r = run_json([host, "load", "--model", model, "--units", self.units, "--function", n,
                                              "--predict", cfg["predictions"]])
                                fr = r["functions"][0]
                                self.record("load_warm", **common, run=i, stage=st, function=n, load_ms=fr["load_ms"],
                                            predict_ms=fr.get("predict_ms"), error=fr.get("error"),
                                            footprint_max=r["footprint_max"])
                    for units in cfg.get("sharing_units", [self.units]):
                        for i in range(self.runs(cfg["sharing_runs"])):
                            for st, names in fns:
                                if len(names) < 2:
                                    continue
                                model = d / sidecar(self.m, self.models, v, s, st).name
                                self.record("sharing", **{**common, "units": units}, run=i, stage=st,
                                            **self._sharing(host, model, names, units))
                    shutil.rmtree(d, ignore_errors=True)

    @staticmethod
    def _sharing(host, model: Path, names, units: str) -> dict:
        """Every function of a stage loaded and held in one process. Per metric,
        the increment after each load + prediction and for the last function
        loaded alone. phys_footprint is what fit charges but misses file-backed
        weights (resident) and what the ANE keeps wired outside the process
        (host wired); host available memory is the noisy catch-all."""
        r = run_json([host, "load", "--model", model, "--units", units, "--predict", "1", "--keep",
                      *sum((["--function", n] for n in names), [])])
        single = run_json([host, "load", "--model", model, "--units", units, "--predict", "1", "--function",
                           names[-1]])
        for fr in r["functions"] + single["functions"]:
            if fr.get("error"):
                raise RuntimeError(f"sharing {model.name} {fr['function']} ({units}): {fr['error']}")
        weight = (model / "weights" / "weight.bin").stat().st_size
        out = {"functions": names, "weight_bytes": weight, "footprint_max": r["footprint_max"], "metrics": {}}
        for metric, sign in SHARING_METRICS.items():
            prev, inc = r[f"{metric}_base"], []
            for fr in r["functions"]:
                inc.append(sign * (fr[f"{metric}_after_predict"] - prev))
                prev = fr[f"{metric}_after_predict"]
            alone = sign * (single["functions"][0][f"{metric}_after_predict"] - single[f"{metric}_base"])
            out["metrics"][metric] = {"increments": inc, "alone": alone, "verdict": sharing_verdict(inc, weight)}
        return out

    # ---- time to ready ----
    def ttr(self, cfg):
        self.need(self.driver, "driver (build-driver)")
        for v in self.variants(cfg):
            for c in cfg["configs"]:
                fns = set_functions(self.m, v, c["set"])
                kind = "mlpackage" if c["route"] == "generate" else "mlmodelc"
                for i in range(self.runs(c["runs"])):
                    with fresh_identity(self.driver, self.hosts) as ident:
                        d = self.stage_dir(v, c["set"], [st for st, _ in fns], kind)
                        arts = []
                        for st, names in fns:
                            path = d / sidecar(self.m, self.models, v, c["set"], st, kind).name
                            arts += ["--artifact", f"{st}={path}:{','.join(names)}"]
                        cmd = [ident["binary"], "ttr", "--mode", c["mode"], "--route", c["route"], *arts,
                               "--work-dir", d / "work", "--cache-dir", d / "cache", "--units", self.units,
                               "--timeout-s", cfg.get("timeout_s", 1800)]
                        if cfg.get("memory_admission"):
                            cmd.append("--memory-admission")
                        waited = wait_quiet(self.proto, self.quiet)
                        r = run_json(cmd, timeout=cfg.get("timeout_s", 1800) + 120, ok_codes=(0, 3))
                        summary = self._ttr_summary(r)
                        self.record("ttr", variant=v, config=c["name"], route=c["route"], mode=c["mode"], set=c["set"],
                                    run=i, identity=ident["id"], quiet_wait_s=waited, **summary)
                        # Cached startup needs every plan of the priming run in the cache.
                        cached = c.get("cached_startup_runs", 0) if c["route"] == "prebuilt" else 0
                        if cached and not summary["all_ready"]:
                            print(f"[ttr] {v}/{c['name']} run {i}: not every function was ready; "
                                  "skipping cached startup", flush=True)
                            cached = 0
                        for j in range(cached):
                            r2 = run_json(cmd, timeout=cfg.get("timeout_s", 1800) + 120, ok_codes=(0, 3))
                            self.record("ttr_cached", variant=v, config=c["name"], set=c["set"], run=i, rep=j,
                                        identity=ident["id"], **self._ttr_summary(r2))
                        shutil.rmtree(d, ignore_errors=True)

    @staticmethod
    def _ttr_summary(r: dict) -> dict:
        slots = r.get("detail", {}).get("slots", [])
        # Ready means every model is loaded, warmed and still resident when the
        # final readings are taken (the driver counts them then).
        live = r.get("models_live")
        all_ready = r["time_to_ready_ms"] >= 0 and live == len(slots)
        return {"time_to_ready_ms": r["time_to_ready_ms"], "all_ready": all_ready, "models_live": live,
                "available_bytes_at_start": r.get("available_bytes_at_start"),
                "deferred_memory_ms": sum(s.get("deferred_memory_ms", 0) or 0 for s in slots),
                "prepare_waits_ms": prepare_waits(r), "load_gate_waits_ms": load_gate_waits(r),
                "footprint_max": r.get("footprint_max"), "timeline": r.get("timeline"),
                "slots": [{"slot": s.get("slot"), "ready_ms": s.get("ready_ms")} for s in slots],
                "stats": r.get("detail", {}).get("stats")}

    # ---- tts-cli ----
    def tts(self, binary: Path, v: str, text: str, max_chars: int, s3gen: Path, coreml: bool, tag: str) -> dict:
        var = self.m["variants"][v]
        wav = self.work / f"{tag}-{uuid.uuid4().hex[:8]}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["/usr/bin/time", "-l", str(binary), "--model", str(resolve(var["t3"], self.models)),
               "--s3gen-gguf", str(s3gen), "--n-gpu-layers", "999", "--seed", str(self.proto["seed"]),
               "--text", text, "--out", str(wav), "--verbose", "--max-sentence-chars", str(max_chars)]
        if var.get("language"):
            cmd += ["--language", var["language"]]
        env = dict(os.environ, CHATTERBOX_CHUNK_PROFILE="1", CHATTERBOX_COREML_COMPUTE_UNITS=self.units)
        env.pop("CHATTERBOX_COREML_DISABLE", None)
        env.pop("CHATTERBOX_COREML_STRICT", None)
        if not coreml:
            env["CHATTERBOX_COREML_DISABLE"] = "1"
        t0 = time.monotonic()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
        wall = (time.monotonic() - t0) * 1000
        if p.returncode != 0:
            raise RuntimeError(f"tts-cli exited {p.returncode}: {p.stderr[-2000:]}")
        out = parse_tts_stderr(p.stderr)
        out["wall_ms"] = wall
        wav.unlink(missing_ok=True)
        return out

    def metal_cost(self, cfg):
        self.need(self.tts_cli, "--tts-cli")
        sw = self.m["workloads"][cfg["workload"]]
        for v in self.variants(cfg):
            s3gen = resolve(self.m["variants"][v]["s3gen"], self.models)
            for i in range(self.runs(cfg["runs"])):
                for k, sentence in enumerate(sw["sentences"]):
                    r = self.tts(self.tts_cli, v, f"{sw['warmup']} {sentence}", len(sentence), s3gen, False, "metal-cost")
                    bad = check_labels(r["labels"], ())
                    if len(r["chunks"]) != 2 or bad:
                        raise RuntimeError(f"metal_cost {v} sentence {k}: {len(r['chunks'])} S3Gen calls, {bad}")
                    self.record("metal_cost", variant=v, run=i, sentence=k, chars=len(sentence), **r["chunks"][1])

    def synthesis(self, cfg):
        self.need(self.tts_cli, "--tts-cli")
        wl = self.m["workloads"][cfg["workload"]]
        text = resolve(wl["text"], self.models).read_text().strip()
        for v in self.variants(cfg):
            var = self.m["variants"][v]
            for c in cfg["configs"]:
                stages = c.get("stages", []) if c.get("coreml") else []
                with contextlib.ExitStack() as stack:
                    if c.get("coreml"):
                        ident = stack.enter_context(fresh_identity(self.tts_cli, self.hosts))
                        binary, identity = Path(ident["binary"]), ident["id"]
                        d = self.stage_dir(v, c["set"], stages, "mlmodelc")
                        # Absolute target: a relative one would resolve inside d.
                        gguf = resolve(var["s3gen"], self.models).resolve()
                        (d / gguf.name).symlink_to(gguf)
                        s3gen = d / gguf.name
                        waited = wait_quiet(self.proto, self.quiet)
                    else:
                        binary, identity, d, waited = self.tts_cli, None, None, 0.0
                        s3gen = resolve(var["s3gen"], self.models)
                    prime = self.tts(binary, v, text, wl["max_sentence_chars"], s3gen, bool(c.get("coreml")), "prime")
                    self.record("synthesis_prime", variant=v, config=c["name"], identity=identity,
                                quiet_wait_s=waited, wall_ms=prime["wall_ms"], labels=prime["labels"])
                    for i in range(self.runs(cfg["runs"])):
                        r = self.tts(binary, v, text, wl["max_sentence_chars"], s3gen, bool(c.get("coreml")), c["name"])
                        bad = check_labels(r["labels"], stages)
                        if bad:
                            raise RuntimeError(f"synthesis {v}/{c['name']} run {i}: {bad}")
                        self.record("synthesis", variant=v, config=c["name"], run=i, identity=identity,
                                    wall_ms=r["wall_ms"], t3_infer_ms=sum(r["t3_infer_ms"]),
                                    s3gen_infer_ms=sum(r["s3gen_infer_ms"]),
                                    load_ms=sum(r["t3_load_ms"]) + sum(r["s3gen_load_ms"]),
                                    segments=len(r["chunks"]), chunks=r["chunks"], labels=r["labels"],
                                    peak_footprint=r["peak_footprint"], max_rss=r["max_rss"])
                    if d is not None:
                        shutil.rmtree(d, ignore_errors=True)


# ---- report -------------------------------------------------------------------


def read_results(path: Path) -> list:
    f = path / "results.jsonl" if path.is_dir() else path
    if not f.exists():
        return []
    return [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]


def _f(x, digits=0, unit=""):
    if x is None:
        return "—"
    return f"{x:,.{digits}f}{unit}"


def _mb(x):
    return "—" if x is None else f"{x / 1e6:,.2f}"


def summarize(recs: list) -> dict:
    by = {}
    for r in recs:
        by.setdefault(r["kind"], []).append(r)
    s: dict = {"env": (by.get("env") or [{}])[-1]}

    # Results appended by several runs into one --out: one row per artifact.
    s["sizes"] = list({(r["variant"], r["set"], r["stage"]): r for r in by.get("sizes", [])}.values())

    mem = by.get("memory", [])
    s["memory"] = mem[-1] if mem else None

    # A cold total is per identity (one fresh-identity run of the set); several
    # runs into one --out give one total each, reported as their median.
    cold = {}
    for r in by.get("load_cold", []):
        cold.setdefault((r["variant"], r["set"]), {}).setdefault(r.get("identity"), []).append(r)
    s["cold"] = {}
    for k, runs in cold.items():
        per_fn = {}
        for rows in runs.values():
            for x in rows:
                per_fn.setdefault((x["stage"], x["function"]), []).append(x["load_ms"])
        s["cold"][k] = {"total_ms": median([sum(x["load_ms"] for x in rows) for rows in runs.values()]),
                        "runs": len(runs), "functions": sorted(per_fn),
                        "load_ms": {fn: median(ms) for fn, ms in per_fn.items()}}

    warm = {}
    for r in by.get("load_warm", []):
        if r.get("error"):
            continue
        key = (r["variant"], r["set"], r["stage"], r["function"])
        first = (r.get("predict_ms") or [0])[0]
        e = warm.setdefault(key, {"load": [], "warm": [], "predict": []})
        e["load"].append(r["load_ms"])
        e["warm"].append(r["load_ms"] + first)
        e["predict"].extend((r.get("predict_ms") or [])[1:] or (r.get("predict_ms") or []))
    s["warm"] = warm

    shar = {}
    for r in by.get("sharing", []):
        shar.setdefault((r["variant"], r["set"], r["stage"], r.get("units", "—")), []).append(r)
    s["sharing"] = shar

    ttr = {}
    for r in by.get("ttr", []):
        ttr.setdefault((r["variant"], r["config"]), []).append(r)
    s["ttr"] = ttr
    cached = {}
    for r in by.get("ttr_cached", []):
        cached.setdefault((r["variant"], r["config"]), []).append(r)
    s["ttr_cached"] = cached

    fits = {}
    pts = {}
    for r in by.get("metal_cost", []):
        pts.setdefault(r["variant"], []).append(r)
    for v, rows in pts.items():
        fits[v] = {}
        # The encoder and CFM run over the whole mel (voice prompt + new frames,
        # T_mu); HiFT only over the new frames (T_mel = T_mu - prompt).
        for stage, x in (("enc_ms", "T_mu"), ("cfm_ms", "T_mu"), ("hift_ms", "T_mel"), ("pipeline_ms", "T_mu")):
            xs = [p[x] for p in rows]
            ys = [p[stage] for p in rows]
            try:
                a, b, r2 = linfit(xs, ys)
            except ValueError:
                fits[v][stage] = None
                continue
            worst = max(y / (a + b * xx) for xx, y in zip(xs, ys) if a + b * xx > 0)
            fits[v][stage] = {"a": a, "b": b, "r2": r2, "n": len(xs), "x": x, "x_min": min(xs), "x_max": max(xs),
                              "worst_ratio": worst}
    s["metal_fit"] = fits

    syn = {}
    for r in by.get("synthesis", []):
        syn.setdefault((r["variant"], r["config"]), []).append(r)
    s["synthesis"] = syn
    return s


def render(s: dict) -> str:
    env = s["env"]
    L = []
    w = L.append

    # Each generated section sits between <!-- report:NAME --> and
    # <!-- /report:NAME -->; `report --out` on an existing file replaces only these.
    def begin(name):
        w(f"<!-- report:{name} -->")

    def end(name):
        w(f"<!-- /report:{name} -->")
        w("")

    w("# Core ML load time measurements")
    w("")
    w("Measured with `scripts/coreml/bench-load-time.py` on the workloads pinned in "
      "`scripts/coreml/bench-manifest.json`. These numbers set the Chatterbox Core ML runtime's constants "
      "(Metal cost model, warm-load hints, width set, weight-sharing accounting); the last section says where each one goes.")
    w("")
    if env:
        w(f"- Host: {env.get('chip', '—')}, {(env.get('memsize') or 0) / 2**30:.0f} GB, macOS "
          f"{env.get('os_version', '—')} ({env.get('os_build', '—')}), AC power: {env.get('ac_power')}.")
        w(f"- Commit `{(env.get('git') or '—')[:12]}`, manifest sha256 `{(env.get('manifest_sha256') or '—')[:16]}…`, "
          f"compute units `{env.get('units', '—')}`.")
        w("- Sidecars exported locally with `tts/scripts/export-chatterbox-coreml.py` from the pinned f32 "
          "GGUFs; tree digests are in `results.jsonl` (`env.sidecars`).")
    else:
        w("- Not measured yet: run `bench-load-time.py run` on an otherwise idle machine, then regenerate this file "
          "with `bench-load-time.py report`.")
    w("")

    begin("setup")
    w("## Setup and usage")
    w("")
    w("Requirements:")
    w("")
    w("- Apple silicon, macOS 15 or later, Xcode command-line tools (`xcrun clang++`) and Python 3.9 or later. "
      "The published numbers come from an otherwise idle M2 on macOS 15.7.3; `run` refuses battery power unless "
      "`--allow-battery` is given.")
    w("- The Chatterbox GGUFs pinned in `bench-manifest.json` (`variants`, with SHA-256 and Hugging Face "
      "revisions) under the models root: `--models-root DIR`, or `QVAC_BENCH_MODELS`.")
    w("- The Core ML sidecars for each width set in `sidecar_sets`, exported from the pinned f32 GGUFs with "
      "`tts/scripts/export-chatterbox-coreml.py` as `exporter` in the manifest describes "
      "(`--frames` selects the widths, `--compile` writes the `.mlmodelc` next to the `.mlpackage`).")
    w("- A `tts-cli` built with `-DTTS_CPP_COREML=ON` (see `tts/docs/chatterbox.md`).")
    w("")
    w("Commands:")
    w("")
    w("```bash")
    w("python3 -B scripts/coreml/test_bench_load_time.py     # hermetic tests, no Core ML needed")
    w("scripts/coreml/bench-load-time.py --dry-run --check-models   # manifest, pinned files and the run plan")
    w("scripts/coreml/bench-load-time.py build-driver")
    w("scripts/coreml/bench-load-time.py smoke                # fresh-host.sh identity and e5rt cache clean-up")
    w("scripts/coreml/bench-load-time.py run --tts-cli <build>/tts-cli --out <dir>   # all measurements")
    w("scripts/coreml/bench-load-time.py run ttr --variant turbo --out <dir>   # one measurement, one variant")
    w("scripts/coreml/bench-load-time.py report --results <dir> --out docs/COREML-LOAD-TIME.md")
    w("scripts/coreml/bench-load-time.py cleanup              # remove leftover bench identities and caches")
    w("```")
    w("")
    w("`run` writes one JSON object per line to `<dir>/results.jsonl`; `report` renders the measurements from it. "
      "Cold runs use a fresh `fresh-host.sh` identity and a fresh copy of the sidecars, and delete both "
      "afterwards.")
    w("")
    w("`report --out` on a new file writes the whole document. On an existing file it replaces only the generated "
      "sections, each between a `<!-- report:NAME -->` and a `<!-- /report:NAME -->` line, and keeps everything "
      "else (the header and the hand-written reading of the numbers); it refuses a file without those markers.")
    w("")
    end("setup")

    begin("sizes")
    w("## Multifunction spec sizes")
    w("")
    w("What a template would embed is the program (`model.mil` compiled, `model.mlmodel` in the package), not the "
      "weights. MB = 10^6 bytes.")
    w("")
    w("| Variant | Set | Stage | Functions | model.mil | zlib | LZFSE | model.mlmodel | zlib | LZFSE | weight.bin |")
    w("|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in s["sizes"] or []:
        w(f"| {r['variant']} | {r['set']} | {r['stage']} | {r['functions']} | {_mb(r['mil_bytes'])} | "
          f"{_mb(r['mil_zlib'])} | {_mb(r.get('mil_lzfse'))} | {_mb(r.get('mlmodel_bytes'))} | "
          f"{_mb(r.get('mlmodel_zlib'))} | {_mb(r.get('mlmodel_lzfse'))} | {_mb(r['weight_bytes'])} |")
    if not s["sizes"]:
        w("| — | | | | | | | | | | |")
    w("")
    end("sizes")

    begin("width-set")
    w("## Width set: 3 vs 7")
    w("")
    w("Cold = sum of first loads on a fresh identity (ANE compile included), all functions of the set incl. HiFT; "
      "with several cold runs, the median of their sums. Warm = median per-function load + first prediction in a "
      "new process. Synthesis = warm `tts-cli` on the benchmark prompt (median of the runs).")
    w("")
    w("| Variant | Set | Functions | Cold total (s) | Warm load+predict, median over functions (ms) | "
      "Synthesis wall (ms) | S3Gen (ms) |")
    w("|---|---|--:|--:|--:|--:|--:|")
    for (v, st), c in sorted(s["cold"].items()):
        ws = [median(e["warm"]) for (vv, ss, _, _), e in s["warm"].items() if vv == v and ss == st]
        syn = s["synthesis"].get((v, f"coreml-{st}"), [])
        w(f"| {v} | {st} | {len(c['functions'])} | {_f(c['total_ms'] / 1000, 1)} | {_f(median(ws), 0)} | "
          f"{_f(median([r['wall_ms'] for r in syn]))} | {_f(median([r['s3gen_infer_ms'] for r in syn]))} |")
    if not s["cold"]:
        w("| — | | | | | | |")
    w("")
    end("width-set")

    begin("warm-load")
    w("## `expected_warm_load_ms` per function")
    w("")
    w("A warm load is a new process on an identity whose ANE plans are cached; the hint's `warm_load_ms` is load "
      "plus the warm-up prediction, which is what this column reports. Prediction = median of the later predictions.")
    w("")
    w("| Variant | Set | Stage | Function | Cold load (ms) | Warm load (ms) | expected_warm_load_ms (median) | Max | "
      "Prediction (ms) |")
    w("|---|---|---|---|--:|--:|--:|--:|--:|")
    cold_fn = {(v, st, *fn): ms for (v, st), c in s["cold"].items() for fn, ms in c["load_ms"].items()}
    for key in sorted(s["warm"]):
        e = s["warm"][key]
        w(f"| {' | '.join(key)} | {_f(cold_fn.get(key))} | {_f(median(e['load']))} | {_f(median(e['warm']))} | "
          f"{_f(max(e['warm']))} | {_f(median(e['predict']), 1)} |")
    if not s["warm"]:
        w("| — | | | | | | | | |")
    w("")
    end("warm-load")

    begin("sharing")
    w("## Do loaded functions share weight memory?")
    w("")
    w(f"All functions of one stage loaded and held in one process, per compute-unit policy. Per metric: the "
      f"increment of the first function, the median of each later one, and the last function loaded alone (MB). "
      f"`footprint` = phys_footprint (what fit charges); `resident` adds file-backed weight pages; `wired` = "
      f"host wired memory, where the ANE keeps a loaded program; `available` = the host's available-memory drop "
      f"(noisy). 'shared' when functions 2..n add less than {SHARED_FRACTION:.0%} of a weight.bin each.")
    w("")
    w("| Variant | Set | Stage | Units | weight.bin (MB) | Metric | First | Each later | Alone | Verdict |")
    w("|---|---|---|---|--:|---|--:|--:|--:|---|")
    for key in sorted(s["sharing"]):
        rows = s["sharing"][key]
        for metric in SHARING_METRICS:
            ms = [r["metrics"][metric] for r in rows if metric in r.get("metrics", {})]
            if not ms:
                continue
            verdict = ", ".join(sorted({x["verdict"] for x in ms}))
            w(f"| {' | '.join(key)} | {_mb(rows[0]['weight_bytes'])} | {metric} | "
              f"{_mb(median([x['increments'][0] for x in ms]))} | "
              f"{_mb(median([y for x in ms for y in x['increments'][1:]]))} | "
              f"{_mb(median([x['alone'] for x in ms]))} | {verdict} |")
    if not s["sharing"]:
        w("| — | | | | | | | | | |")
    w("")
    end("sharing")

    begin("metal-cost")
    w("## Per-variant Metal cost constants for `s3gen_estimate_metal_ms`")
    w("")
    w("Least-squares `ms = a + b × frames` over the length sweep on Metal (`CHATTERBOX_COREML_DISABLE=1`), one S3Gen "
      "call per sentence after a discarded warm-up call. Encoder, CFM and the whole pipeline against `T_mu` "
      "(prompt + new frames), HiFT against `T_mel` (new frames). "
      "Worst ratio = the largest measured / fitted.")
    w("")
    w("| Variant | Stage | a (ms) | b (ms/frame) | R² | Points | Frames | Worst ratio |")
    w("|---|---|--:|--:|--:|--:|---|--:|")
    for v in sorted(s["metal_fit"]):
        for stage, fit in s["metal_fit"][v].items():
            if fit is None:
                continue
            w(f"| {v} | {stage[:-3]} | {fit['a']:.1f} | {fit['b']:.4f} | {fit['r2']:.4f} | {fit['n']} | "
              f"{fit['x']} {fit['x_min']}–{fit['x_max']} | {fit['worst_ratio']:.2f} |")
    if not s["metal_fit"]:
        w("| — | | | | | | | |")
    w("")
    end("metal-cost")

    begin("ttr")
    w("## Time to ready, with and without the admission gate")
    w("")
    w("Every function (encoder and CFM widths plus HiFT) requested at `minimal` in one construction group with "
      "`wait=construct`, on a fresh identity with the artifacts at a fresh path. `generate` = a stand-in prepare "
      "job per artifact (copy the package, `compileModelAtURL`), as on-device generation will; gated = the shared "
      "runtime's Manager, ungated = the same two queues without the gate. Prepare wait = issue to prepare start; "
      "load gate wait = artifact available to load start.")
    w("")
    w("| Variant | Config | Runs | All ready | Time to ready median (s) | MAD (s) | Min–max (s) | "
      "Worst prepare wait (s) | Worst load wait (s) | deferred(memory) (s) | Cached startup (s) |")
    w("|---|---|--:|---|--:|--:|---|--:|--:|--:|--:|")
    for key in sorted(s["ttr"]):
        rows = s["ttr"][key]
        ok = [r["time_to_ready_ms"] / 1000 for r in rows if r["all_ready"]]
        pw = [x for r in rows for x in r["prepare_waits_ms"]]
        lw = [x for r in rows for x in r["load_gate_waits_ms"]]
        cached = [r["time_to_ready_ms"] / 1000 for r in s["ttr_cached"].get(key, []) if r["all_ready"]]
        w(f"| {key[0]} | {key[1]} | {len(rows)} | {len(ok)}/{len(rows)} | {_f(median(ok), 1)} | {_f(mad(ok), 1)} | "
          f"{_f(min(ok) if ok else None, 1)}–{_f(max(ok) if ok else None, 1)} | "
          f"{_f(max(pw) / 1000 if pw else None, 1)} | {_f(max(lw) / 1000 if lw else None, 1)} | "
          f"{_f(sum(r['deferred_memory_ms'] for r in rows) / 1000, 1)} | {_f(median(cached), 2)} |")
    if not s["ttr"]:
        w("| — | | | | | | | | | | |")
    w("")
    end("ttr")

    begin("memory")
    w("## Available-memory reading under normal load")
    w("")
    mem = s["memory"]
    if mem:
        w(f"{len(mem['samples'])} samples of free + inactive + purgeable + speculative pages "
          f"(`host_statistics64`, the runtime's own reading): min {_mb(mem['min'])} MB, median "
          f"{_mb(mem['median'])} MB, max {_mb(mem['max'])} MB; low power {mem['low_power']}.")
        w("")
        w("| Artifact | ane_compile_reserve_bytes (MB) | Deferred at the minimum reading? |")
        w("|---|--:|---|")
        for k, res in sorted(mem["reserves"].items()):
            w(f"| {k} | {_mb(res)} | {'yes' if mem['min'] < res else 'no'} |")
    else:
        w("Not measured yet.")
    w("")
    end("memory")

    begin("synthesis")
    w("## Warm synthesis and encoder-fallback contention")
    w("")
    w("`tts-cli`, benchmark prompt, built-in voice, seed 42, q8_0, a new process per run after one priming run. "
      "`coreml-no-encoder` keeps CFM and HiFT on Core ML (so S3Gen still overlaps T3) and the encoder on Metal: "
      "the T3 slowdown against `coreml-w7` is the contention the runtime pays when the encoder falls back.")
    w("")
    w("| Variant | Config | Runs | Wall (ms) | T3 (ms) | S3Gen (ms) | Peak footprint (MB) |")
    w("|---|---|--:|--:|--:|--:|--:|")
    for key in sorted(s["synthesis"]):
        rows = s["synthesis"][key]
        w(f"| {key[0]} | {key[1]} | {len(rows)} | {_f(median([r['wall_ms'] for r in rows]))} | "
          f"{_f(median([r['t3_infer_ms'] for r in rows]))} | {_f(median([r['s3gen_infer_ms'] for r in rows]))} | "
          f"{_mb(median([r['peak_footprint'] for r in rows]))} |")
    if not s["synthesis"]:
        w("| — | | | | | | |")
    w("")
    end("synthesis")

    begin("destinations")
    w("## Where each number goes")
    w("")
    w("| Number | Where it goes |")
    w("|---|---|")
    w("| Spec sizes | The multifunction template size budget; LZFSE compression if a template is over it |")
    w("| 3 vs 7 widths (cold total, warm load, synthesis) | The width set the Chatterbox generation templates ship |")
    w("| `expected_warm_load_ms` per function | Each template's `template.json`; the bounded wait before a hint "
      "exists |")
    w("| Weight sharing verdict | How resident functions are charged against the Core ML memory envelope |")
    w("| Metal cost constants | `s3gen_estimate_metal_ms` in the Chatterbox Core ML runtime |")
    w("| Time to ready with / without the gate, worst prepare wait | Whether the admission gate stays on; the "
      "Chatterbox \"time to all 15 ready\" target |")
    w("| Available-memory reading | Confirms memory admission does not defer cold loads on the bench host |")
    w("| Encoder-fallback contention | Expected overlap when the Chatterbox encoder runs on Metal |")
    w("")
    end("destinations")
    return "\n".join(L)


# ---- CLI ----------------------------------------------------------------------


def cmd_dry_run(args, m) -> int:
    errors = validate(m, REPO, models_root(args.models_root), args.check_models)
    if errors:
        for e in errors:
            print(f"manifest error: {e}", file=sys.stderr)
        return 1
    selected = getattr(args, "measurements", None) or list(MEASUREMENTS)
    steps = plan(m, selected, getattr(args, "variant", None))
    print(f"manifest ok: {args.manifest} ({len(steps)} steps)")
    for name, what in steps:
        print(f"  {name:11s} {what}")
    return 0


def cmd_build_driver(args) -> int:
    out = Path(args.out or DEFAULT_BUILD / "bench-load-driver")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["xcrun", "clang++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-fobjc-arc",
           "-DQVAC_COREML_NS=qvac_coreml_bench", "-I", RUNTIME / "include", "-I", RUNTIME / "src",
           "-I", RUNTIME / "third_party", DRIVER_SRC, *[RUNTIME / s for s in RUNTIME_SOURCES],
           "-framework", "CoreML", "-framework", "Foundation", "-framework", "Security", "-o", out]
    print(" ".join(map(str, cmd)))
    return subprocess.run([str(c) for c in cmd]).returncode


def cmd_run(args, m) -> int:
    unknown = [x for x in args.measurements if x not in MEASUREMENTS]
    if unknown:
        print(f"unknown measurement(s) {unknown}; choose from {MEASUREMENTS}", file=sys.stderr)
        return 64
    if args.dry_run:
        return cmd_dry_run(args, m)
    errors = validate(m, REPO, models_root(args.models_root), check_models=True)
    if errors:
        for e in errors:
            print(f"manifest error: {e}", file=sys.stderr)
        return 1
    selected = args.measurements or list(MEASUREMENTS)
    r = Runner(args, m)
    r.environment()
    try:
        for name in selected:
            getattr(r, name)(m["measurements"][name])
    finally:
        subprocess.run(["bash", str(FRESH_HOST), "cleanup-all"], capture_output=True,
                       env=dict(os.environ, QVAC_BENCH_HOSTS=str(r.hosts)))
        shutil.rmtree(r.work, ignore_errors=True)
    print(f"results: {r.results}")
    return 0


def cmd_smoke(args, m) -> int:
    """Device smoke for fresh-host.sh: a re-signed copy reports a new
    host identity, a plain copy does not, a cold load creates the identity's
    e5rt cache, and cleanup deletes it."""
    driver = Path(args.driver or DEFAULT_BUILD / "bench-load-driver").resolve()
    if not driver.exists():
        print(f"smoke: driver not found: {driver} (run build-driver)", file=sys.stderr)
        return 1
    models = models_root(args.models_root)
    variant = next(iter(m["variants"]))
    model = sidecar(m, models, variant, "w7", "encoder")
    fn = function_names("encoder", m["sidecar_sets"]["w7"]["frames"][:1], m["variants"][variant]["token_mel_ratio"])[0]
    checks = {}
    with tempfile.TemporaryDirectory() as tmp:
        hosts = Path(tmp) / "hosts"
        plain = Path(tmp) / "plain-copy"
        shutil.copy2(driver, plain)
        base = run_json([driver, "identity"])["host_id"]
        checks["plain_copy_keeps_identifier"] = run_json([plain, "identity"])["host_id"] == base
        env = dict(os.environ, QVAC_BENCH_HOSTS=str(hosts))
        with fresh_identity(driver, hosts) as ident:
            got = run_json([ident["binary"], "identity"])["host_id"]
            checks["fresh_identity_reported"] = got == ident["id"] != base
            if model.exists():
                r = run_json([ident["binary"], "load", "--model", model, "--function", fn, "--units",
                              m["protocol"]["units"], "--predict", "1"])
                checks["cold_load_ok"] = not r["functions"][0].get("error")
                caches = subprocess.run(["bash", str(FRESH_HOST), "caches", ident["id"]], capture_output=True,
                                        text=True, env=env).stdout.splitlines()
                checks["e5rt_cache_created"] = any((Path(c) / "com.apple.e5rt.e5bundlecache").exists()
                                                   for c in caches)
                cache_dir = Path(ident["cache"])
            else:
                print(f"smoke: {model} not found; skipping the cache checks", file=sys.stderr)
                cache_dir = None
        if cache_dir is not None:
            checks["e5rt_cache_removed"] = not cache_dir.exists()
        checks["copy_removed"] = not Path(ident["binary"]).exists()
    print(json.dumps({"smoke": checks, "identity": ident["id"]}))
    return 0 if checks and all(checks.values()) else 1


REPORT_BLOCK_RE = re.compile(r"^<!-- report:([a-z0-9-]+) -->\n.*?^<!-- /report:\1 -->$", re.S | re.M)


class ReportError(Exception):
    pass


def merge_report(existing: str, rendered: str) -> str:
    """`existing` with each generated block replaced by the same block of
    `rendered`; text outside the blocks is kept as it is."""
    new = {m.group(1): m.group(0) for m in REPORT_BLOCK_RE.finditer(rendered)}
    old = [m.group(1) for m in REPORT_BLOCK_RE.finditer(existing)]
    if not old:
        raise ReportError("no generated sections (<!-- report:NAME --> ... <!-- /report:NAME -->); refusing to "
                          "replace hand-written text. Write the report to a new file instead")
    dup = sorted({n for n in old if old.count(n) > 1})
    missing = [n for n in new if n not in old]
    unknown = [n for n in old if n not in new]
    if dup or missing or unknown:
        raise ReportError(f"generated sections do not match the report: duplicated {dup}, missing {missing}, "
                          f"unknown {unknown}; add or remove the markers by hand")
    return REPORT_BLOCK_RE.sub(lambda m: new[m.group(1)], existing)


def cmd_report(args) -> int:
    text = render(summarize(read_results(Path(args.results)))) + "\n"
    if not args.out:
        print(text, end="")
        return 0
    out = Path(args.out)
    if out.exists():
        try:
            text = merge_report(out.read_text(), text)
        except ReportError as e:
            print(f"report: {out}: {e}", file=sys.stderr)
            return 1
    out.write_text(text)
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--dry-run", action="store_true", help="validate the manifest and print the plan; run nothing")
    ap.add_argument("--check-models", action="store_true", help="also hash the pinned model files (--dry-run)")
    ap.add_argument("--models-root", help="default $QVAC_BENCH_MODELS or ~/tether/models")
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build-driver", help="compile bench-load-driver with the runtime sources")
    b.add_argument("--out")
    r = sub.add_parser("run", help="run measurements (on an otherwise idle machine)")
    r.add_argument("measurements", nargs="*", metavar="MEASUREMENT", help=f"any of {', '.join(MEASUREMENTS)} (default all)")
    r.add_argument("--variant", action="append", choices=("turbo", "mtl"))
    r.add_argument("--out", help="results directory (default build-bench/results/<UTC time>)")
    r.add_argument("--driver", help="bench-load-driver (default build-bench/bench-load-driver)")
    r.add_argument("--tts-cli", help="tts-cli built with TTS_CPP_COREML=ON (metal_cost, synthesis)")
    r.add_argument("--units", choices=UNITS)
    r.add_argument("--runs", type=int, help="override every run count (smoke tests only)")
    r.add_argument("--no-quiet-check", action="store_true", help="skip the ANECompilerService wait (smoke only)")
    r.add_argument("--allow-battery", action="store_true", help="run on battery power (smoke runs only; numbers are not comparable)")
    rp = sub.add_parser("report", help="render the measurements doc from results")
    rp.add_argument("--results", required=True)
    rp.add_argument("--out", help="a new file gets the whole document; an existing one keeps its text and has only "
                                  "the generated sections between <!-- report:NAME --> markers replaced")
    sub.add_parser("cleanup", help="delete every qvac-bench-* identity copy and cache")
    sm = sub.add_parser("smoke", help="fresh-host.sh identity smoke and e5rt cache cleanup (needs Core ML)")
    sm.add_argument("--driver")
    args = ap.parse_args(argv)

    if args.cmd == "build-driver":
        return cmd_build_driver(args)
    if args.cmd == "report":
        return cmd_report(args)
    if args.cmd == "cleanup":
        return subprocess.run(["bash", str(FRESH_HOST), "cleanup-all"]).returncode
    try:
        m = load_manifest(args.manifest)
    except ManifestError as e:
        print(f"manifest error: {e}", file=sys.stderr)
        return 1
    if args.cmd == "run":
        return cmd_run(args, m)
    if args.cmd == "smoke":
        return cmd_smoke(args, m)
    if args.dry_run:
        return cmd_dry_run(args, m)
    ap.print_help()
    return 64


if __name__ == "__main__":
    sys.exit(main())
