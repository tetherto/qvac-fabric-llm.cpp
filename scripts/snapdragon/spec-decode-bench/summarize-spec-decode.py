#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from statistics import mean, median

CONFIG_ORDER = ["nospec", "mtp1", "mtp3", "mtp7"]
CONFIG_LABEL = {"nospec": "no spec", "mtp1": "MTP n_max 1", "mtp3": "MTP n_max 3", "mtp7": "MTP n_max 7"}
NO_SPEC = "nospec"
THROTTLED_PWRLEVEL = 3
THROTTLED_TPS_FRACTION = 0.7
HEADER = ("| Model | Decode | {a} (tok/s) | {b} (tok/s) | {b}/{a} | {a} vs its no-spec | {b} vs its no-spec "
          "| Accepted / drafted ({a}) | Accepted / drafted ({b}) | Clean samples ({a}/{b}) | Throttled samples ({a}/{b}) |")
SEPARATOR = "|---|---|---|---|---|---|---|---|---|---|---|"


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def group_cells(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["config"], r["build"])].append(r)
    return cells


def is_throttled(r, cell_median):
    return (r["thermal_before"]["gpu_pwrlevel"] >= THROTTLED_PWRLEVEL
            or r["thermal_after"]["gpu_pwrlevel"] >= THROTTLED_PWRLEVEL
            or r.get("gpu_pwrlevel_max", 0) >= THROTTLED_PWRLEVEL
            or r["decode_tps"] < THROTTLED_TPS_FRACTION * cell_median)


def clean_samples(rs):
    cell_median = median(r["decode_tps"] for r in rs)
    return [r for r in rs if not is_throttled(r, cell_median)]


def cell_stats(rs):
    clean = clean_samples(rs) if rs else []
    if not clean:
        return None
    tps = [r["decode_tps"] for r in clean]
    return {"tps": mean(tps), "tps_min": min(tps), "tps_max": max(tps), "n": len(clean),
            "throttled": len(rs) - len(clean),
            "drafted": sum(r["draft_n"] for r in clean), "accepted": sum(r["draft_accepted"] for r in clean)}


def ratio(a, b):
    return f"{a / b:.2f}x" if a and b else "-"


def tps_str(st):
    return f"{st['tps']:.2f} ({st['tps_min']:.1f}-{st['tps_max']:.1f})" if st else "-"


def accept_str(st):
    if not st or not st["drafted"]:
        return "-"
    return f"{st['accepted']}/{st['drafted']} ({100.0 * st['accepted'] / st['drafted']:.0f}%)"


def row_line(model, cfg, a, b, a_nospec, b_nospec):
    return "| {m} | {cfg} | {at} | {bt} | {sp} | {an} | {bn} | {aa} | {ba} | {cnt} | {pw} |".format(
        m=model, cfg=CONFIG_LABEL.get(cfg, cfg), at=tps_str(a), bt=tps_str(b),
        sp=ratio(b["tps"] if b else None, a["tps"] if a else None),
        an=ratio(a["tps"] if a else None, a_nospec["tps"] if a_nospec else None),
        bn=ratio(b["tps"] if b else None, b_nospec["tps"] if b_nospec else None),
        aa=accept_str(a), ba=accept_str(b),
        cnt=f"{a['n'] if a else 0}/{b['n'] if b else 0}",
        pw=f"{a['throttled'] if a else 0}/{b['throttled'] if b else 0}")


def model_lines(cells, model, build_a, build_b):
    a_nospec = cell_stats(cells.get((model, NO_SPEC, build_a)))
    b_nospec = cell_stats(cells.get((model, NO_SPEC, build_b)))
    lines = []
    for cfg in CONFIG_ORDER:
        a = cell_stats(cells.get((model, cfg, build_a)))
        b = cell_stats(cells.get((model, cfg, build_b)))
        if a or b:
            lines.append(row_line(model, cfg, a, b, a_nospec, b_nospec))
    return lines


def table_lines(cells, build_a, build_b):
    lines = [HEADER.format(a=build_a, b=build_b), SEPARATOR]
    for model in sorted({k[0] for k in cells}):
        lines.extend(model_lines(cells, model, build_a, build_b))
    return lines


def parse_args():
    p = argparse.ArgumentParser(description="Markdown table from bench-spec-decode.py results")
    p.add_argument("results", nargs="+", help="jsonl files written by bench-spec-decode.py")
    p.add_argument("--build-a", default="base")
    p.add_argument("--build-b", default="pr")
    return p.parse_args()


def main():
    args = parse_args()
    cells = group_cells(load_rows(args.results))
    print("\n".join(table_lines(cells, args.build_a, args.build_b)))


if __name__ == "__main__":
    main()
