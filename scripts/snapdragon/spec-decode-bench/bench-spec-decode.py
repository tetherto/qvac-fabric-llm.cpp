#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

DEFAULT_DEVICE_DIR = "/data/local/tmp/llama.cpp"
DEFAULT_DEVICE_PORT = 8080
DEFAULT_HOST_PORT = 18080
DEFAULT_BUILDS = "base=base,pr=pr"
DEFAULT_MODELS = "2B=models/Qwen3.5-2B-Q4_K_M.gguf,4B=models/Qwen3.5-4B-Q4_K_M.gguf"
DEFAULT_CONFIGS = "nospec,mtp1,mtp3,mtp7"
DEFAULT_PASSES = 2
DEFAULT_N_PREDICT = 256
DEFAULT_CTX = 4096
DEFAULT_RESULTS_DIR = "spec-decode-results"
KGSL = "/sys/class/kgsl/kgsl-3d0"
VENDOR_LIB = "/vendor/lib64"
HEALTH_TIMEOUT_S = 300
HEALTH_POLL_S = 2
COMPLETION_TIMEOUT_S = 1800
COOL_MAX_WAIT_S = 900
COOL_POLL_S = 15
COOL_SETTLE_S = 20
COOL_MAX_PWRLEVEL = 2
COOL_MAX_GPU_TEMP_C = 48.0
COOL_MAX_SKIN_C = 36.5
BETWEEN_LAUNCH_S = 5
OTHER_BENCH_POLL_S = 60
SERVER_START_TIMEOUT_S = 20
PWR_POLL_S = 2

CONFIGS = {
    "nospec": [],
    "mtp1": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"],
    "mtp3": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
    "mtp7": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "7"],
}
PROMPTS = [
    "Write a detailed essay about the history of the Roman Empire, covering its rise, its peak under the Antonines, and the causes of its fall.",
    "Explain step by step how a transformer neural network processes a sentence, from tokenization through attention to the output logits, for a software engineer who has never studied machine learning.",
    "Write a short story about a lighthouse keeper on a remote island who finds a message in a bottle that was written by her grandmother fifty years earlier.",
]


class Device:
    def __init__(self, serial, device_dir, device_port, host_port):
        self.serial = serial
        self.device_dir = device_dir
        self.device_port = device_port
        self.host_port = host_port
        self.server_url = f"http://127.0.0.1:{host_port}"

    def adb_args(self):
        return ["adb", "-s", self.serial] if self.serial else ["adb"]

    def shell(self, cmd, timeout=60, check=True):
        return sh([*self.adb_args(), "shell", cmd], timeout=timeout, check=check).stdout

    def forward(self):
        sh([*self.adb_args(), "forward", f"tcp:{self.host_port}", f"tcp:{self.device_port}"])


def sh(args, timeout=None, check=True):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=check)


def read_thermal(dev):
    raw = dev.shell(f"cat {KGSL}/thermal_pwrlevel {KGSL}/max_gpuclk {KGSL}/temp; "
                    "dumpsys thermalservice | grep -m1 'Thermal Status' | tr -dc '0-9'; echo; "
                    "dumpsys thermalservice | sed -n '/Current temperatures from HAL/,/Current cooling/p' | grep -m1 'mName=SKIN' | sed 's/.*mValue=\\([0-9.]*\\).*/\\1/'")
    parts = raw.split()
    return {
        "gpu_pwrlevel": int(parts[0]),
        "gpu_max_clk_mhz": int(parts[1]) // 1_000_000,
        "gpu_temp_c": int(parts[2]) / 1000.0,
        "thermal_status": int(parts[3]) if len(parts) > 3 and parts[3] else -1,
        "skin_temp_c": float(parts[4]) if len(parts) > 4 else -1.0,
    }


def read_pwrlevel(dev):
    return int(dev.shell(f"cat {KGSL}/thermal_pwrlevel").strip())


class PwrSampler:
    def __init__(self, dev):
        self.dev = dev
        self.max_pwr = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                self.max_pwr = max(self.max_pwr, read_pwrlevel(self.dev))
            except Exception:
                pass
            self.stop.wait(PWR_POLL_S)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def is_cool(t):
    return (t["thermal_status"] == 0
            and t["gpu_pwrlevel"] <= COOL_MAX_PWRLEVEL
            and t["gpu_temp_c"] <= COOL_MAX_GPU_TEMP_C
            and t["skin_temp_c"] <= COOL_MAX_SKIN_C)


def wait_for_cool(dev, enabled):
    if not enabled:
        return 0, read_thermal(dev)
    waited = 0
    t = read_thermal(dev)
    while not is_cool(t) and waited < COOL_MAX_WAIT_S:
        time.sleep(COOL_POLL_S)
        waited += COOL_POLL_S
        t = read_thermal(dev)
    time.sleep(COOL_SETTLE_S)
    return waited, read_thermal(dev)


def other_bench_running(dev, pattern):
    host = sh(["pgrep", "-f", pattern], check=False).stdout.strip()
    device = dev.shell(f"ps -A | grep -c '[{pattern[0]}]{pattern[1:]}' || true", check=False).strip()
    return bool(host) or device not in ("", "0")


def wait_for_other_bench(dev, pattern):
    while pattern and other_bench_running(dev, pattern):
        print(f"process matching '{pattern}' is still running, waiting", flush=True)
        time.sleep(OTHER_BENCH_POLL_S)


def stop_server(dev):
    dev.shell("pkill llama-server || true", check=False)
    time.sleep(1)


def server_command(dev, build_dir, model_path, server_args, cfg_args, log_name, env):
    env_str = " ".join(f"{k}={v}" for k, v in env.items())
    args = " ".join([f"./{build_dir}/llama-server", "-m", model_path, *server_args, *cfg_args])
    return (f"cd {dev.device_dir} && mkdir -p logs && LD_LIBRARY_PATH={VENDOR_LIB}:{dev.device_dir} {env_str} "
            f"nohup {args} > logs/{log_name}.log 2>&1 < /dev/null &")


def start_server(dev, build_dir, model_path, server_args, cfg_args, log_name, env):
    stop_server(dev)
    try:
        dev.shell(server_command(dev, build_dir, model_path, server_args, cfg_args, log_name, env),
                  timeout=SERVER_START_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        pass
    dev.forward()


def http_json(dev, path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(dev.server_url + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def health_ok(dev):
    try:
        return http_json(dev, "/health", timeout=HEALTH_POLL_S).get("status") == "ok"
    except Exception:
        return False


def wait_for_health(dev, log_name):
    waited = 0
    while waited < HEALTH_TIMEOUT_S:
        if health_ok(dev):
            return waited
        time.sleep(HEALTH_POLL_S)
        waited += HEALTH_POLL_S
    tail = dev.shell(f"tail -c 2000 {dev.device_dir}/logs/{log_name}.log", check=False)
    raise RuntimeError(f"server {log_name} not healthy after {HEALTH_TIMEOUT_S}s:\n{tail}")


def render_prompt(dev, text):
    return http_json(dev, "/apply-template", {"messages": [{"role": "user", "content": text}]})["prompt"]


def run_completion(dev, prompt, n_predict):
    payload = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": False}
    return http_json(dev, "/completion", payload, timeout=COMPLETION_TIMEOUT_S)


def timings_row(resp):
    t = resp.get("timings", {})
    return {
        "prompt_n": t.get("prompt_n"),
        "prompt_tps": t.get("prompt_per_second"),
        "predicted_n": t.get("predicted_n"),
        "predicted_ms": t.get("predicted_ms"),
        "decode_tps": t.get("predicted_per_second"),
        "draft_n": t.get("draft_n", 0),
        "draft_accepted": t.get("draft_n_accepted", 0),
        "content_head": resp.get("content", "")[:80],
    }


def print_prompt_result(idx, row, before):
    print(f"    p{idx}: {row['decode_tps']:.2f} tok/s  draft {row['draft_accepted']}/{row['draft_n']}  "
          f"pwr {before['gpu_pwrlevel']}->{row['thermal_after']['gpu_pwrlevel']} max {row['gpu_pwrlevel_max']}", flush=True)


def run_prompts(ctx, base_row, out):
    for idx, text in enumerate(ctx.prompts):
        wait_for_cool(ctx.dev, ctx.cool_per_prompt)
        before = read_thermal(ctx.dev)
        prompt = render_prompt(ctx.dev, text)
        with PwrSampler(ctx.dev) as sampler:
            resp = run_completion(ctx.dev, prompt, ctx.n_predict)
        row = {**base_row, "prompt_idx": idx, "thermal_before": before,
               "thermal_after": read_thermal(ctx.dev), "gpu_pwrlevel_max": sampler.max_pwr, **timings_row(resp)}
        out.write(json.dumps(row) + "\n")
        out.flush()
        print_prompt_result(idx, row, before)


def run_cell(ctx, build, model, cfg_name, pass_idx, out):
    log_name = f"{build}-{model}-{cfg_name}-p{pass_idx}"
    cool_wait, thermal = wait_for_cool(ctx.dev, ctx.cool_gate)
    print(f"  [{build} {model} {cfg_name} pass {pass_idx}] cooled {cool_wait}s, "
          f"pwr {thermal['gpu_pwrlevel']} temp {thermal['gpu_temp_c']:.1f}C skin {thermal['skin_temp_c']:.1f}C", flush=True)
    start_server(ctx.dev, ctx.builds[build], ctx.models[model], ctx.server_args, CONFIGS[cfg_name], log_name, ctx.env)
    load_s = wait_for_health(ctx.dev, log_name)
    base_row = {"build": build, "model": model, "config": cfg_name, "pass": pass_idx,
                "env": ctx.env, "load_s": load_s, "cool_wait_s": cool_wait, "ts": time.time()}
    run_prompts(ctx, base_row, out)
    stop_server(ctx.dev)
    time.sleep(BETWEEN_LAUNCH_S)


def build_order(row_idx, pass_idx, builds):
    return builds if (row_idx + pass_idx) % 2 == 0 else list(reversed(builds))


def run_passes(ctx, row_idx, model, cfg_name, out):
    for pass_idx in range(ctx.passes):
        for build in build_order(row_idx, pass_idx, list(ctx.builds)):
            run_cell(ctx, build, model, cfg_name, pass_idx, out)


def run_rows(ctx, out):
    row_idx = 0
    for model in ctx.models:
        for cfg_name in ctx.configs:
            run_passes(ctx, row_idx, model, cfg_name, out)
            row_idx += 1


class RunContext:
    def __init__(self, args):
        self.dev = Device(args.serial, args.device_dir, args.device_port, args.host_port)
        self.builds = parse_pairs(args.builds)
        self.models = parse_pairs(args.models)
        self.configs = args.configs.split(",")
        self.passes = args.passes
        self.env = parse_pairs(args.env)
        self.n_predict = args.n_predict
        self.prompts = load_prompts(args.prompts_file)
        self.cool_gate = not args.no_cool_gate
        self.cool_per_prompt = self.cool_gate and args.cool_per_prompt
        self.server_args = ["-ngl", "99", "-c", str(args.ctx), "-np", "1", "--no-cache-prompt",
                            "--port", str(args.device_port), "--host", "127.0.0.1"]


def parse_pairs(spec):
    return dict(kv.split("=", 1) for kv in spec.split(",") if kv)


def load_prompts(path):
    if not path:
        return PROMPTS
    return [line for line in Path(path).read_text().splitlines() if line.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Interleaved llama-server decode benchmark on an adb device")
    p.add_argument("--serial", default="", help="adb serial; default is the only attached device")
    p.add_argument("--device-dir", default=DEFAULT_DEVICE_DIR)
    p.add_argument("--device-port", type=int, default=DEFAULT_DEVICE_PORT)
    p.add_argument("--host-port", type=int, default=DEFAULT_HOST_PORT)
    p.add_argument("--builds", default=DEFAULT_BUILDS, help="name=subdir pairs under --device-dir")
    p.add_argument("--models", default=DEFAULT_MODELS, help="name=path pairs relative to --device-dir")
    p.add_argument("--configs", default=DEFAULT_CONFIGS, help=f"subset of {','.join(CONFIGS)}")
    p.add_argument("--passes", type=int, default=DEFAULT_PASSES, help="ABBA passes per row")
    p.add_argument("--env", default="", help="KEY=VALUE pairs exported to llama-server")
    p.add_argument("--n-predict", type=int, default=DEFAULT_N_PREDICT)
    p.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    p.add_argument("--prompts-file", default="", help="one prompt per line; default: three built-in prompts")
    p.add_argument("--no-cool-gate", action="store_true", help="do not wait for the GPU to cool before each launch")
    p.add_argument("--cool-per-prompt", action="store_true", help="also wait for the GPU to cool before each prompt")
    p.add_argument("--wait-for-process", default="", help="pgrep -f pattern to wait on before starting (shared devices)")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--tag", default="run")
    return p.parse_args()


def main():
    args = parse_args()
    ctx = RunContext(args)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.tag}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    print(f"results -> {out_path}", flush=True)
    wait_for_other_bench(ctx.dev, args.wait_for_process)
    with out_path.open("w") as out:
        run_rows(ctx, out)
    stop_server(ctx.dev)
    print(f"done -> {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
