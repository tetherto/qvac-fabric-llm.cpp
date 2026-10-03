"""Render the explicitly selected DFlash2 campaign evidence as Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from manifest import runtime_fingerprint

BASE_SHA = "20e89b5a0b87f9f79e640e0bc3e209b9e192bc52"
TF_SHA = "6ea5ade26c4335491c50275af0be32b81f75f525"
MODEL_SHA = "ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d"
DRAFT_SHA = "ef6a1340ed018cc58f53efd3613312c719b9ebf72b5586a046214c7ba366b297"
FIXTURE_SHA = "cf991cd976528946657d122b6315ef73acd1f800ac68f07221b64e45833d5a83"
LANES = {"metal": ("mac", "metal"), "rtx5090-cuda": ("rtx5090", "cuda"),
         "rtx5090-vulkan": ("rtx5090", "vulkan"), "spark-cuda": ("spark", "cuda"),
         "spark-vulkan": ("spark", "vulkan"), "strix-vulkan": ("strix", "vulkan")}
METRICS = ("server_prefill_tps", "decode_tps")
ARMS = ("base", "candidate", "serial_base", "serial_candidate", "tf")
CONFIG_FIELDS = ("source_sha", "model_sha256", "fixture_sha256", "context", "batch", "ubatch",
                 "cache_k", "cache_v", "threads", "threads_batch", "flash_attn", "offload_layers",
                 "parallel", "greedy", "gpu_id", "backend_config", "toolchain", "evidence")
DRAFT_FIELDS = ("draft_sha256", "cache_k_draft", "cache_v_draft")
TF_CONFIG_FIELDS = ("source_sha", "model_sha256", "draft_sha256", "fixture_sha256", "context",
                    "parallel", "greedy", "gpu_id", "backend_config", "evidence", "drafting_active")
RUNTIME_FIELDS = ("target_gpu_layers", "target_layers", "draft_gpu_layers", "draft_layers",
                  "target_n_rs_seq", "draft_n_rs_seq")


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def digest(value, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None


def load_json(path: Path):
    return json.loads(path.read_text())


def resolve(root: Path, value: str) -> Path:
    require(isinstance(value, str) and bool(value), "artifact path must be a nonempty string")
    return (root / value).resolve()


def cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def link(path: Path, label: str | None = None) -> str:
    return f"[{cell(label or path.name)}](<{quote(str(path), safe='/:-._~')}>)"


def markdown_table(headers: list, rows: list) -> list[str]:
    return ["| " + " | ".join(map(cell, headers)) + " |", "|" + "---|" * len(headers),
            *["| " + " | ".join(map(cell, row)) + " |" for row in rows]]


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def bootstrap(log_blocks: list[float], seed: int, resamples: int) -> dict:
    """Resample independent ABBA blocks, never prompts or streamed tokens."""
    estimate = math.exp(statistics.median(log_blocks))
    if len(log_blocks) < 3:
        return {"median": estimate, "ci": None, "blocks": [math.exp(x) for x in log_blocks]}
    rng = random.Random(seed)
    samples = [math.exp(statistics.median(rng.choices(log_blocks, k=len(log_blocks)))) for _ in range(resamples)]
    return {"median": estimate, "ci": [percentile(samples, 0.025), percentile(samples, 0.975)],
            "blocks": [math.exp(x) for x in log_blocks]}


def aggregate(log_lanes: list[list[float]], seed: int, resamples: int) -> dict:
    require(len(log_lanes) == 6 and all(len(x) >= 3 for x in log_lanes), "aggregate needs all six lanes and >=3 blocks each")
    rng = random.Random(seed)
    estimate = math.exp(statistics.mean(statistics.median(x) for x in log_lanes))
    samples = [math.exp(statistics.mean(statistics.median(rng.choices(x, k=len(x))) for x in log_lanes))
               for _ in range(resamples)]
    return {"median": estimate, "ci": [percentile(samples, 0.025), percentile(samples, 0.975)]}


def optional_rate(value) -> str:
    return f"{value:.3f}" if positive(value) else "unreported"


def ratio_cell(value: dict | None) -> str:
    if value is None:
        return "UNMEASURED"
    interval = value["ci"]
    suffix = f" [{interval[0]:.4f}, {interval[1]:.4f}]" if interval else " [CI withheld: <3 blocks]"
    return f"{value['median']:.4f}x" + suffix


def verdict(value: dict | None) -> str:
    if value is None or value["ci"] is None:
        return "UNMEASURED"
    if value["median"] < 1:
        return "REGRESSION: reject or gather more blocks"
    if value["ci"][0] > 1:
        return "GAIN"
    return "UNCHANGED: interval includes 1"


def validate_config(data: dict, arm: str, lane_id: str, index: dict, root: Path) -> dict:
    manifest = data["provenance"]["manifest"]
    config = manifest["lane"]["report_config"]
    tf = arm == "tf"
    serial = arm.startswith("serial_")
    fields = TF_CONFIG_FIELDS if tf else CONFIG_FIELDS
    for field in fields + (() if tf or serial else DRAFT_FIELDS):
        require(field in config and config[field] is not None, f"missing report_config.{field}")
    device, backend = LANES[lane_id]
    require(manifest["device"] == device, "device mismatch")
    require(manifest["backend"] == ("mlx" if tf and backend == "metal" else backend), "backend mismatch")
    require(not tf or backend != "vulkan", "TensorFold has no Vulkan comparison")
    require(digest(manifest.get("binary_sha256")), "missing binary SHA256")
    require(isinstance(manifest.get("command"), list) and manifest["command"], "missing exact launch command")
    require(positive(manifest.get("created_unix_s")), "missing launch timestamp")
    require(isinstance(manifest.get("environment"), dict), "missing backend environment")
    if not tf:
        artifacts = manifest.get("runtime_artifacts")
        require(isinstance(artifacts, dict) and artifacts, "missing executable/shared-library identities")
        require(all(isinstance(name, str) and isinstance(item, dict) and digest(item.get("sha256"))
                    and isinstance(item.get("path"), str) and item["path"]
                    for name, item in artifacts.items()), "invalid runtime artifact identity")
        launcher = Path(manifest["command"][0]).name
        require(artifacts.get(launcher, {}).get("sha256") == manifest["binary_sha256"],
                "runtime launcher identity mismatch")
        require(manifest.get("runtime_sha256") == runtime_fingerprint(artifacts),
                "runtime fingerprint mismatch")
    source = TF_SHA if tf else index["baseline_sha" if arm.endswith("base") else "candidate_sha"]
    require(config["source_sha"] == source, "source pin mismatch")
    require(config["fixture_sha256"] == data["provenance"].get("fixture_sha256") == FIXTURE_SHA, "fixture hash mismatch")
    require(digest(data["provenance"].get("client_sha256")), "missing client hash")
    require(config["context"] == 16384 and config["parallel"] == 1 and config["greedy"] is True, "headline context/parallel/greedy mismatch")
    require(isinstance(config["backend_config"], dict) and config["backend_config"], "missing backend configuration")
    require(bool(config["gpu_id"]), "missing physical GPU identity")
    require(isinstance(config["evidence"], list) and config["evidence"], "missing configuration log evidence")
    for artifact in config["evidence"]:
        require(resolve(root, artifact).is_file(), f"missing configuration evidence: {artifact}")
    if tf:
        require(digest(config["model_sha256"]) and digest(config["draft_sha256"]), "missing TensorFold weight hashes")
        require(config.get("drafting_active") is True, "TensorFold drafting not evidenced")
    else:
        require(config["model_sha256"] == MODEL_SHA, "target model hash mismatch")
        require(config["batch"] == 2048 and config["ubatch"] == 512, "headline batch/ubatch mismatch")
        require(config["cache_k"] == config["cache_v"] == "f16", "headline cache precision mismatch")
        require(config["flash_attn"] == "on" and config["offload_layers"] == 99, "headline attention/offload mismatch")
        require(all(type(config[k]) is int and config[k] > 0 for k in ("threads", "threads_batch")), "actual CPU threads unreported")
        require(bool(config["toolchain"]), "missing toolchain")
        width = 0 if serial else index["lanes"][lane_id]["draft_width"]
        require(manifest["draft_width"] == width, "draft width mismatch")
        if serial:
            flags = {arg.split("=")[0] for arg in manifest["command"]}
            require(not flags.intersection(("-md", "--model-draft", "--spec-type", "--spec-draft-n-max")),
                    "serial launch still enables a drafter")
        runtime = manifest["lane"]["runtime_evidence"]
        require(all(type(runtime.get(k)) is int and runtime[k] >= 0 for k in RUNTIME_FIELDS),
                "missing numeric actual offload/state evidence")
        require(runtime["target_gpu_layers"] == runtime["target_layers"] > 0, "target is not fully GPU offloaded")
        require(runtime["draft_gpu_layers"] == runtime["draft_layers"] and
                (runtime["draft_layers"] == 0 if serial else runtime["draft_layers"] > 0),
                "draft offload does not match execution mode")
        require(runtime["target_n_rs_seq"] == width and runtime["draft_n_rs_seq"] == 0,
                "actual rollback state configuration mismatch")
        require(isinstance(runtime.get("evidence"), list) and runtime["evidence"], "missing runtime log evidence")
        for artifact in runtime["evidence"]:
            require(resolve(root, artifact).is_file(), f"missing runtime evidence: {artifact}")
        if not serial:
            require(config["draft_sha256"] == DRAFT_SHA, "draft model hash mismatch")
            require(config["cache_k_draft"] == config["cache_v_draft"] == "f16", "draft cache precision mismatch")
    return config


def load_record(path: Path, arm: str, lane_id: str, index: dict, root: Path) -> dict:
    record = {"path": path, "arm": arm, "lane": lane_id, "error": None, "proof_error": None}
    try:
        data = load_json(path)
        require(isinstance(data, dict), "result must be a JSON object")
        record["data"] = data
        require(isinstance(data.get("label"), str) and data["label"], "missing explicit run label")
        require(data["engine"] == ("tf" if arm == "tf" else "fabric"), "engine mismatch")
        require(data.get("complete") is True and not data.get("error"), "incomplete/error result")
        require(data["summary"].get("all_ok") is True, "summary gate failed")
        runs = data["runs"]
        require(isinstance(runs, list) and bool(runs), "empty result")
        require([r["prompt_index"] for r in runs] == index["prompt_indices"], "prompt indices mismatch or partial rows")
        for run in runs:
            require(run["prompt_tokens"] == 10000 and run["completion_tokens"] == 1024, "headline token count mismatch")
            require(run.get("ok") is True and isinstance(run.get("checks"), dict) and run["checks"]
                    and all(v is True for v in run["checks"].values()), "run checks failed")
            for check in ("prompt_tokens", "completion_tokens", "cached_zero", "server_error_free", "positive_intervals"):
                require(run["checks"].get(check) is True, f"missing/failed check: {check}")
            cached = run.get("cached_tokens")
            reported = type(cached) is int and cached == 0 and run.get("cache_status") == "reported"
            evidence = run.get("cache_evidence") or {}
            evidenced = (arm == "tf" and cached is None and run.get("cache_status") == "evidenced"
                         and evidence.get("uncached") is True and evidence.get("source_revision") == TF_SHA
                         and isinstance(evidence.get("artifact"), str) and resolve(root, evidence["artifact"]).is_file())
            require(reported or evidenced, "cached or unproven uncached result")
            request = run["request"]
            require(request.get("temperature") == 0 and request.get("max_tokens") == 1024
                    and request.get("ignore_eos") is True and request.get("stream") is True, "request settings mismatch")
            require(isinstance(request.get("prompt"), str) and request["prompt"], "missing rendered prompt")
            require(arm == "tf" or request.get("cache_prompt") is False, "prompt reuse not disabled")
            require(digest(run.get("text_sha256")), "missing full output hash")
            require(isinstance(run.get("text"), str) and bool(run["text"]), "missing full output text")
            require(hashlib.sha256(run["text"].encode()).hexdigest() == run["text_sha256"],
                    "full output hash does not match saved text")
            require(positive(run.get("ttft_s")) and positive(run.get("decode_s")), "invalid client timing")
            require(not (run.get("extras") or {}).get("error"), "native server error")
            for metric, expected in (("decode_tps", 1023 / run["decode_s"]), ("prefill_tps", 10000 / run["ttft_s"])):
                require(positive(run.get(metric)) and math.isclose(run[metric], expected, rel_tol=1e-8), f"inconsistent {metric}")
            require(positive(run["server"].get("prefill_tps")), "missing engine prefill rate")
        rates = {"server_prefill_tps": [r["server"]["prefill_tps"] for r in runs],
                 "decode_tps": [r["decode_tps"] for r in runs], "prefill_tps": [r["prefill_tps"] for r in runs]}
        for metric, values in rates.items():
            require(positive(data["summary"].get(metric))
                    and math.isclose(data["summary"][metric], statistics.median(values), rel_tol=1e-8), f"summary {metric} disagrees with rows")
    except (ValueError, KeyError, TypeError, AttributeError, OSError, ZeroDivisionError) as error:
        record["error"] = str(error)
    if record["error"] is None:
        try:
            record["config"] = validate_config(data, arm, lane_id, index, root)
            require(data["provenance"]["manifest"].get("label") == data["label"], "manifest label mismatch")
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
            record["proof_error"] = str(error)
    return record


def check_cross_engine_sessions(records: list[dict]) -> None:
    first_arm = records[0]["arm"]
    prompts = {row["prompt_index"]: row["request"]["prompt"] for record in records
               if record["arm"] == first_arm for row in record["data"]["runs"]}
    for record in records:
        for row in record["data"]["runs"]:
            require(prompts.get(row["prompt_index"]) == row["request"]["prompt"],
                    "cross-engine rendered prompt mismatch")
    for arm in {record["arm"] for record in records}:
        selected = [record["data"] for record in records if record["arm"] == arm]
        counts = Counter(row["prompt_index"] for data in selected for row in data["runs"])
        require(set(counts) == set(prompts) and len(set(counts.values())) == 1,
                "cross-engine sessions need equal coverage of every selected prompt")
        launches = [data["provenance"]["manifest"]["created_unix_s"] for data in selected]
        require(len(set(launches)) == len(launches), "reused cross-engine server launch")
        require(len({data["label"] for data in selected}) == len(selected), "reused cross-engine launch label")


def comparable(records: list[dict], cross_engine: bool = False) -> None:
    require(all(r["error"] is None for r in records), "invalid selected result")
    require(all(r["proof_error"] is None for r in records), "missing or incompatible configuration/runtime proof")
    first = records[0]
    cross_keys = ("fixture_sha256", "context", "parallel", "greedy", "gpu_id")
    def config_key(record, across_engines=cross_engine):
        if across_engines:
            return {k: record["config"][k] for k in cross_keys}
        fields = TF_CONFIG_FIELDS if record["arm"] == "tf" else CONFIG_FIELDS
        if record["arm"] != "tf" and record["data"]["provenance"]["manifest"]["draft_width"] != 0:
            fields += DRAFT_FIELDS
        normalized = {k: record["config"][k] for k in fields if k not in ("source_sha", "evidence")}
        if record["arm"] != "tf":
            runtime = record["data"]["provenance"]["manifest"]["lane"]["runtime_evidence"]
            normalized["runtime_evidence"] = {k: runtime[k] for k in RUNTIME_FIELDS}
        return normalized
    if cross_engine:
        check_cross_engine_sessions(records)
    for record in records[1:]:
        require(config_key(record) == config_key(first), "workload/backend configuration mismatch")
        a, b = first["data"], record["data"]
        require(a["provenance"]["client_sha256"] == b["provenance"]["client_sha256"], "client revision mismatch")
        if not cross_engine:
            require([r["prompt_index"] for r in a["runs"]] == [r["prompt_index"] for r in b["runs"]], "paired prompt mismatch")
            for left, right in zip(a["runs"], b["runs"]):
                require(left["request"] == right["request"], "paired request mismatch")
    for arm in {r["arm"] for r in records}:
        identity_key = "binary_sha256" if arm == "tf" else "runtime_sha256"
        binaries = {r["data"]["provenance"]["manifest"][identity_key] for r in records if r["arm"] == arm}
        require(len(binaries) == 1, "mixed executable/shared-library runtimes within an arm")
        if cross_engine:
            same_arm = [r for r in records if r["arm"] == arm]
            require(all(config_key(record, False) == config_key(same_arm[0], False) for record in same_arm),
                    "workload/backend configuration mismatch within an engine")


def block_logs(records: list[dict], metric: str) -> float:
    def rate(record, position):
        run = record["data"]["runs"][position]
        return run["server"]["prefill_tps"] if metric == "server_prefill_tps" else run[metric]
    return statistics.mean((math.log(rate(records[1], i)) - math.log(rate(records[0], i))
                            + math.log(rate(records[2], i)) - math.log(rate(records[3], i))) / 2
                           for i in range(len(records[0]["data"]["runs"])))


def comparison(lane: dict, records: dict[Path, dict], root: Path, arms: tuple[str, str], block_key: str,
               seed: int, resamples: int) -> dict:
    result = {"error": None, "metrics": {}, "logs": {}, "block_ids": []}
    try:
        blocks = lane.get(block_key, [])
        require(isinstance(blocks, list) and blocks, "no independent ABBA blocks")
        selected = [r for r in records.values() if r["arm"] in arms]
        require(selected and all(r["error"] is None for r in selected), "missing/invalid selected results")
        comparable(selected)
        used = []
        timestamps = []
        labels = []
        for block in blocks:
            require(isinstance(block["id"], str) and block["id"] and block["id"] not in result["block_ids"], "duplicate/missing block id")
            result["block_ids"].append(block["id"])
            paths = [resolve(root, p) for p in block["launches"]]
            require(len(paths) == 4, "each block needs four launches")
            group = [records[p] for p in paths]
            require([r["arm"] for r in group] == [arms[0], arms[1], arms[1], arms[0]], "block order is not A/B/B/A")
            require(all(len(r["data"]["runs"]) >= 3 for r in group), "ABBA launches need >=3 prompt fixtures")
            times = [r["data"]["provenance"]["manifest"]["created_unix_s"] for r in group]
            require(all(a < b for a, b in zip(times, times[1:])), "launch timestamps do not prove ABBA order")
            timestamps.extend(times)
            labels.extend(r["data"]["label"] for r in group)
            used.extend(paths)
            for metric in METRICS:
                result["logs"].setdefault(metric, []).append(block_logs(group, metric))
        require(len(set(used)) == len(used) and set(used) == {r["path"] for r in selected}, "blocks must use every selection exactly once")
        require(len(set(timestamps)) == len(timestamps) and len(set(labels)) == len(labels), "launches are not independent/distinct")
        require(all(a < b for a, b in zip(timestamps, timestamps[1:])), "blocks must be listed in independent launch order")
        result["metrics"] = {metric: bootstrap(values, seed, resamples) for metric, values in result["logs"].items()}
    except (ValueError, KeyError, TypeError) as error:
        result["error"] = str(error)
        result["logs"] = {}
        result["metrics"] = {}
    return result


def selected_sessions(lane: dict, arm: str, prompts: list[int]) -> list[tuple[str, list[int]]]:
    selections = lane.get(arm, [])
    require(isinstance(selections, list), f"{arm} must be an explicit selection list")
    if arm != "tf":
        return [(path, prompts) for path in selections]
    result = []
    for selection in selections:
        require(isinstance(selection, dict), "TensorFold selection needs path and prompt_indices")
        selected = selection.get("prompt_indices")
        require(isinstance(selected, list) and selected and all(type(p) is int and p in prompts for p in selected)
                and len(set(selected)) == len(selected), "invalid TensorFold session prompt_indices")
        result.append((selection.get("path"), selected))
    counts = Counter(prompt for _, selected in result for prompt in selected)
    require(not selections or (set(counts) == set(prompts) and len(set(counts.values())) == 1),
            "TensorFold selections need equal coverage of every selected prompt")
    return result


def analyze(index_path: Path) -> dict:
    index_path = index_path.resolve()
    root = index_path.parent
    index = load_json(index_path)
    require(index.get("schema_version") == 1, "unsupported index schema_version")
    require(index.get("baseline_sha") == BASE_SHA, "baseline must be pinned PR325")
    require(digest(index.get("candidate_sha"), 40), "candidate_sha must be a full source pin")
    require(isinstance(index.get("campaign"), str) and index["campaign"], "missing campaign name")
    prompts = index.get("prompt_indices")
    require(isinstance(prompts, list) and prompts and all(type(p) is int and 0 <= p < 12 for p in prompts)
            and len(set(prompts)) == len(prompts), "prompt_indices must explicitly select distinct canonical fixtures")
    require(isinstance(index.get("lanes"), dict) and not set(index["lanes"]) - set(LANES), "unknown lane id")
    boot = index["bootstrap"]
    require(type(boot.get("seed")) is int and type(boot.get("resamples")) is int and boot["resamples"] >= 1000,
            "bootstrap needs integer seed and >=1000 resamples")
    result = {"index": index, "index_path": index_path, "lanes": {}, "aggregate": {}, "aggregate_error": None}
    seen = set()
    for lane_id in LANES:
        lane = index["lanes"].get(lane_id, {})
        require(isinstance(lane, dict), "lane must be an object")
        if lane:
            require(type(lane.get("draft_width")) is int and lane["draft_width"] in (3, 5, 7), "baseline selected draft_width must be 3, 5 or 7")
        records = {}
        for arm in ARMS:
            for path, selected_prompts in selected_sessions(lane, arm, prompts):
                resolved = resolve(root, path)
                require(resolved not in seen, f"result selected more than once: {path}")
                seen.add(resolved)
                record_index = dict(index, prompt_indices=selected_prompts)
                records[resolved] = load_record(resolved, arm, lane_id, record_index, root)
        result["lanes"][lane_id] = {"records": records, "spec": comparison(lane, records, root, ("base", "candidate"), "blocks", **boot),
                                    "serial": comparison(lane, records, root, ("serial_base", "serial_candidate"), "serial_blocks", **boot)}
    failures = [lane_id for lane_id, lane in result["lanes"].items()
                if lane["spec"]["error"] or any(len(x) < 3 for x in lane["spec"]["logs"].values())]
    if failures:
        result["aggregate_error"] = "UNMEASURED: all six valid lanes with >=3 independent blocks required; withheld: " + ", ".join(failures)
    else:
        result["aggregate"] = {metric: aggregate([lane["spec"]["logs"][metric] for lane in result["lanes"].values()], **boot)
                               for metric in METRICS}
    return result


def rates(records: list[dict], arm: str) -> str:
    selected = [r for r in records if r["arm"] == arm]
    if not selected or any(r["error"] for r in selected):
        return "UNMEASURED"
    suffix = " (PROVISIONAL: missing/incompatible proof)" if any(r["proof_error"] for r in selected) else ""
    return " / ".join(f"{statistics.median(r['data']['summary'][m] for r in selected):.2f}" for m in METRICS) + suffix


def cross_engine_rates(records: list[dict], arm: str) -> list[float]:
    rows = [row for record in records if record["arm"] == arm for row in record["data"]["runs"]]
    prompts = sorted({row["prompt_index"] for row in rows})
    rates_by_metric = []
    for metric in METRICS:
        per_prompt = []
        for prompt in prompts:
            values = [row["server"]["prefill_tps"] if metric == "server_prefill_tps" else row[metric]
                      for row in rows if row["prompt_index"] == prompt]
            per_prompt.append(statistics.median(values))
        rates_by_metric.append(statistics.median(per_prompt))
    return rates_by_metric


def evidence_sections(index: dict, root: Path) -> list[str]:
    lines = ["", "## Teacher-forced quality evidence", "",
             "Timing rows and output hashes do not establish numerical fidelity. Indexed artifacts below are not independently validated by this report."]
    if not index.get("quality_index"):
        lines.append("UNMEASURED: no teacher-forced quality evidence index selected.")
    else:
        path = resolve(root, index["quality_index"])
        lines.append("Evidence index: " + link(path))
        try:
            rows = []
            for entry in load_json(path)["entries"]:
                artifact = resolve(path.parent, entry["artifact"])
                rows.append([entry["lane"], entry["schedule"], entry["reference_sha"], entry["candidate_sha"],
                             link(artifact), "INDEXED ONLY" if artifact.is_file() else "UNMEASURED: missing artifact"])
            lines += markdown_table(["Lane", "Schedule", "Reference source", "Candidate source", "Raw evidence", "Status"], rows)
        except (ValueError, KeyError, TypeError, OSError) as error:
            lines.append("UNMEASURED: invalid quality index: " + str(error))
    lines += ["", "## Top operation evidence", ""]
    if not index.get("profile_index"):
        return lines + ["UNMEASURED: no production profile or exact-shape proxy index selected."]
    path = resolve(root, index["profile_index"])
    lines.append("Evidence index: " + link(path))
    try:
        for entry in load_json(path)["entries"]:
            require(entry["lane"] in LANES and entry["stage"] in ("prefill", "decode"), "unknown profile lane/stage")
            require(entry["kind"] in ("production", "proxy"), "profile must label production or proxy")
            artifact = resolve(path.parent, entry["artifact"])
            require(artifact.is_file(), "missing profile artifact")
            production = entry["kind"] == "production"
            denominator = entry.get("denominator_ms")
            require(not production or positive(denominator), "production profile needs measured stage denominator_ms")
            require(isinstance(entry["ops"], list) and entry["ops"], "empty profile")
            require(all(positive(op["time_ms"]) for op in entry["ops"]), "invalid operation time")
            residual = entry.get("residual_cpu_sync_ms")
            require(residual is None or type(residual) in (int, float) and math.isfinite(residual) and residual >= 0,
                    "invalid residual CPU/sync time")
            lines += ["", f"### {entry['lane']} / {entry['stage']} / {entry['kind']}", "",
                      "Raw: " + link(artifact),
                      f"Production stage denominator: {denominator} ms" if production else "Proxy only: no production share or summed stage-time claim.",
                      f"Residual CPU/sync: {entry['residual_cpu_sync_ms']} ms" if entry.get("residual_cpu_sync_ms") is not None else "Residual CPU/sync: UNMEASURED"]
            rows = [[op["family"], op["shape"], op["dtype"], f"{op['time_ms']:.6f}",
                     f"{100 * op['time_ms'] / denominator:.2f}%" if production else "n/a (proxy)"]
                    for op in sorted(entry["ops"], key=lambda op: op["time_ms"], reverse=True)[:10]]
            lines += markdown_table(["Family", "Shape", "Dtype", "Time ms", "Production share"], rows)
    except (ValueError, KeyError, TypeError, OSError) as error:
        lines.append("UNMEASURED: invalid profile index: " + str(error))
    return lines


def report(result: dict) -> str:
    index = result["index"]
    lines = [f"# {index['campaign']}", "", "Index: " + link(result["index_path"]), "",
             f"Fabric base: `{index['baseline_sha']}`; candidate: `{index['candidate_sha']}`.",
             f"TensorFold required reference: unmodified v0.6.4 `{TF_SHA}`. Cross-engine ratios are engine-plus-format comparisons, not kernel-only gains.",
             "Headline: context 16384, prompt 10000, completion 1024, greedy, one request, uncached; Q4_0 target / Q8_0 DFlash2.",
             "Prefill = engine prompt rate; decode = client (completion-1)/(last-first). Client TTFT includes different stage boundaries.",
             "Medians below use selected launches only. Paired ratios use matched prompts inside ordered independent A/B/B/A launch blocks.",
             f"95% percentile block bootstrap: seed={index['bootstrap']['seed']}, resamples={index['bootstrap']['resamples']}; >=3 blocks required.",
             "", "## Six-lane scoreboard", "", "Rates are prefill / decode tok/s. Ratios are candidate / baseline median paired block ratios [95% CI]."]
    rows = []
    for lane_id, lane in result["lanes"].items():
        records = list(lane["records"].values())
        metrics = lane["spec"]["metrics"]
        rows.append([lane_id, index["lanes"].get(lane_id, {}).get("draft_width", "UNMEASURED"), rates(records, "base"), rates(records, "candidate"),
                     ratio_cell(metrics.get(METRICS[0])), ratio_cell(metrics.get(METRICS[1])),
                     "; ".join(verdict(metrics.get(m)) for m in METRICS)])
    lines += markdown_table(["Lane", "Frozen n_max", "Baseline pp / tg", "Candidate pp / tg", "Prefill ratio", "Decode ratio", "Prefill / decode status"], rows)
    lines += ["", "## Equal-weight six-lane aggregate", ""]
    if result["aggregate_error"]:
        lines.append(result["aggregate_error"])
    else:
        rows = []
        for metric, target in zip(METRICS, (1.50, 1.35)):
            value = result["aggregate"][metric]
            no_regression = all(lane["spec"]["metrics"][metric]["median"] >= 1 for lane in result["lanes"].values())
            passed = value["ci"][0] >= target and no_regression
            rows.append([metric, ratio_cell(value), f">={target:.2f}x", "PERFORMANCE TARGET MET" if passed else "TARGET NOT MET"])
        lines += markdown_table(["Metric", "Geometric mean of lane medians [95% CI]", "Target", "Performance only"], rows)
    lines += ["Quality acceptance remains separate; this report does not infer a quality pass.", "", "## Serial regression control", ""]
    rows = []
    for lane_id, lane in result["lanes"].items():
        metrics = lane["serial"]["metrics"]
        rows.append([lane_id, rates(list(lane["records"].values()), "serial_base"), rates(list(lane["records"].values()), "serial_candidate"),
                     ratio_cell(metrics.get(METRICS[0])), ratio_cell(metrics.get(METRICS[1])),
                     "; ".join(verdict(metrics.get(m)) for m in METRICS)])
    lines += markdown_table(["Lane", "Base pp / tg", "Candidate pp / tg", "Prefill ratio", "Decode ratio", "Prefill / decode status"], rows)
    lines += ["", "## TensorFold comparison (engine plus format)", "",
              "Descriptive selected-session ratios only, not paired performance claims. No Vulkan/TF ratios; never included in the six-lane aggregate."]
    rows = []
    for lane_id, lane in result["lanes"].items():
        if LANES[lane_id][1] == "vulkan":
            rows.append([lane_id, "n/a: no TensorFold Vulkan backend", "n/a", "n/a"])
            continue
        records = list(lane["records"].values())
        for arm in ("base", "candidate"):
            selected = [r for r in records if r["arm"] in (arm, "tf")]
            try:
                require(any(r["arm"] == arm for r in selected) and any(r["arm"] == "tf" for r in selected), "missing selection")
                comparable(selected, cross_engine=True)
                tf_rates = cross_engine_rates(selected, "tf")
                fabric_rates = cross_engine_rates(selected, arm)
                ratios = [fabric / tf for fabric, tf in zip(fabric_rates, tf_rates)]
                rows.append([lane_id + " / " + arm, " / ".join(f"{value:.2f}" for value in tf_rates),
                             f"{ratios[0]:.4f}x", f"{ratios[1]:.4f}x"])
            except (ValueError, KeyError, TypeError) as error:
                rows.append([lane_id + " / " + arm, rates(records, "tf"), "UNMEASURED: " + str(error), "UNMEASURED"])
    lines += markdown_table(["Lane / Fabric arm", "TF pp / tg", "Prefill Fabric/TF", "Decode Fabric/TF"], rows)
    lines += ["", "## Block evidence and exclusions", ""]
    rows = []
    for lane_id, lane in result["lanes"].items():
        for mode in ("spec", "serial"):
            comparison_result = lane[mode]
            if comparison_result["error"]:
                rows.append([lane_id, mode, "UNMEASURED", comparison_result["error"]])
            else:
                for i, block_id in enumerate(comparison_result["block_ids"]):
                    rows.append([lane_id, mode, block_id,
                                 " / ".join(f"{comparison_result['metrics'][m]['blocks'][i]:.8f}x" for m in METRICS)])
    lines += markdown_table(["Lane", "Mode", "Block", "Prefill / decode ratios or exclusion"], rows)
    lines += ["", "## Raw runs and stage-boundary cross-check", "",
              "Each raw link includes the full manifest, exact request/launch, native metadata and output. Missing rates are not zero; multi-token chunks can affect client boundaries."]
    rows = []
    for lane_id, lane in result["lanes"].items():
        for record in lane["records"].values():
            data = record.get("data", {})
            raw = link(record["path"], data.get("label", record["path"].name))
            if record["proof_error"]:
                raw += " (PROVISIONAL: " + record["proof_error"] + ")"
            if record["error"]:
                rows.append([lane_id, record["arm"], raw, "INVALID: " + record["error"], "UNMEASURED", "UNMEASURED", "UNMEASURED", "UNMEASURED"])
                continue
            summary = data["summary"]
            rows.append([lane_id, record["arm"], raw, optional_rate(summary["server_prefill_tps"]), optional_rate(summary["prefill_tps"]),
                         optional_rate(summary.get("server_decode_tps")), optional_rate(summary["decode_tps"]),
                         f"{statistics.median(r['ttft_s'] for r in data['runs']):.4f}"])
    lines += markdown_table(["Lane", "Arm", "Exact raw artifact", "Engine pp", "Client pp (prompt/TTFT)", "Engine tg", "Client tg", "TTFT s"], rows)
    lines += ["", "## Build and configuration evidence", ""]
    rows = []
    for lane_id, lane in result["lanes"].items():
        for record in lane["records"].values():
            if record["error"] or record["proof_error"]:
                continue
            manifest = record["data"]["provenance"]["manifest"]
            config = record["config"]
            rows.append([lane_id, record["data"]["label"], config["source_sha"], manifest["binary_sha256"],
                         config["model_sha256"], config.get("draft_sha256", "n/a: serial"),
                         ", ".join(link(resolve(result["index_path"].parent, p)) for p in
                                   config["evidence"] + manifest["lane"].get("runtime_evidence", {}).get("evidence", []))])
    lines += markdown_table(["Lane", "Run", "Source", "Binary SHA256", "Target SHA256", "Draft SHA256", "Verified config logs"], rows)
    lines += evidence_sections(index, result["index_path"].parent)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--require-complete", action="store_true", help="exit 1 if the six-lane aggregate is withheld")
    args = parser.parse_args()
    try:
        result = analyze(args.index)
        print(report(result), end="")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    if args.require_complete and result["aggregate_error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
