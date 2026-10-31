"""Summarise a held-out gate run: usage analyze_gate.py <output dir> <offloaded layers>. Writes <output dir>/gate.json."""
import glob, json, math, os, random, re, sys
if len(sys.argv) != 3:
    sys.exit(__doc__)
out = sys.argv[1]; layers = int(sys.argv[2])
rows = {}
for log in glob.glob(os.path.join(out, "*.log")):
    mode, window = os.path.basename(log)[:-4].split("-", 1) if not os.path.basename(log).startswith("quality-metal") else ("quality-metal", os.path.basename(log)[len("quality-metal-"):-4])
    for line in open(log):
        if line.startswith("quality "):
            kv = {k: v for k, v in (x.split("=", 1) for x in line.split()[1:])}
            rows.setdefault(window, {})[mode] = kv
problems, windows = [], []
for w, r in sorted(rows.items()):
    if "quality" not in r or "quality-metal" not in r:
        problems.append(f"missing run for {w}"); continue
    n = int(r["quality"]["prompt_tokens"])
    for mode in ("quality", "quality-metal"):
        if int(r[mode]["offload_calls"]) != layers * n // 512:
            problems.append(f"{w} {mode} offload_calls={r[mode]['offload_calls']}")
    d = float(r["quality"]["split_nll"]) - float(r["quality"]["metal_nll"])
    dc = float(r["quality-metal"]["split_nll"]) - float(r["quality-metal"]["metal_nll"])
    windows.append(dict(window=w, tokens=n, dnll=d, dnll_control=dc, kl=float(r["quality"]["mean_kl"]), kl_control=float(r["quality-metal"]["mean_kl"]),
                        top1=float(r["quality"]["top1_agreement"])))
def wmean(xs, key):
    t = sum(x["tokens"] for x in xs); return sum(x[key] * x["tokens"] for x in xs) / t
random.seed(11259)
boot = sorted(wmean([random.choice(windows) for _ in windows], "dnll") for _ in range(10000))
mean_d, upper = wmean(windows, "dnll"), boot[int(0.975 * len(boot))]
kl, klc = wmean(windows, "kl"), wmean(windows, "kl_control")
kl_limit = max(0.002, 2 * klc)
def domain(name):
    return {"cpp": "code", "py": "code", "doc": "docs", "prose": "prose", "math": "math"}[name.split("-")[0]]
domains = {}
for x in windows:
    domains.setdefault(domain(x["window"]), []).append(x)
per_domain = {d: dict(tokens=sum(x["tokens"] for x in xs), mean_dnll=wmean(xs, "dnll"), mean_kl=wmean(xs, "kl"),
                      kl_limit=max(0.003, 2 * wmean(xs, "kl_control"))) for d, xs in sorted(domains.items())}
for d in per_domain.values():
    d["pass"] = d["mean_dnll"] <= 0.003 and d["mean_kl"] <= d["kl_limit"]
worst = max(windows, key=lambda x: x["dnll"])
for x in windows:
    print(f"{x['window']:16s} tokens={x['tokens']:5d} dnll={x['dnll']:+.5f} (control {x['dnll_control']:+.5f}) kl={x['kl']:.6f} (control {x['kl_control']:.6f}) top1={x['top1']:.4f}")
result = dict(windows=len(windows), tokens=sum(x["tokens"] for x in windows), mean_dnll=mean_d, dnll_upper95=upper, ppl_change_percent=100 * (math.exp(mean_d) - 1),
              mean_kl=kl, mean_kl_control=klc, kl_limit=kl_limit, problems=problems,
              per_domain=per_domain, worst_window=worst["window"], worst_window_dnll=worst["dnll"],
              pass_primary=mean_d <= 0.002 and upper <= 0.005, pass_kl_screen=kl <= kl_limit, pass_hard=not problems,
              pass_domains=all(d["pass"] for d in per_domain.values()), pass_windows=worst["dnll"] <= 0.005)
result["pass"] = all(result[k] for k in ("pass_primary", "pass_kl_screen", "pass_hard", "pass_domains", "pass_windows"))
print(json.dumps(result, indent=1))
json.dump(dict(result, rows=windows), open(os.path.join(out, "gate.json"), "w"), indent=1)
