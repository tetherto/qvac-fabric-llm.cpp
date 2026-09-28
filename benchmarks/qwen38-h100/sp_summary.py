#!/usr/bin/env python3
"""Summarize pd_bench result files: per shape and rep, the server prompt rate, the client TTFT and the decode rate.

Usage: sp_summary.py results/fabric/K3-sp4-b1.json [more.json ...]
"""
import json
import sys
from collections import defaultdict


def main():
    for path in sys.argv[1:]:
        d = json.load(open(path))
        print(f"== {d['label']}")
        means = defaultdict(list)
        for s in d["shapes"]:
            st = s["server_timings"]
            print(f"  rep {s['rep']} {s['shape']:<14} prompt {st['prompt_per_second']:8.1f} tok/s  ttft {s['ttft_s']:6.2f} s  "
                  f"decode {st['predicted_per_second']:6.2f} tok/s  cached {s['cached_tokens']}")
            means[s["shape"]].append((st["prompt_per_second"], st["predicted_per_second"], s["ttft_s"]))
        for shape, v in means.items():
            n = len(v)
            print(f"  mean  {shape:<14} prompt {sum(x[0] for x in v)/n:8.1f}  decode {sum(x[1] for x in v)/n:6.2f}  ttft {sum(x[2] for x in v)/n:6.2f}")


if __name__ == "__main__":
    main()
