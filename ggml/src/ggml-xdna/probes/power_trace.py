#!/usr/bin/env python3
# power_trace.py <trace file> -- cmd...: package power per profiler section.
#
# Samples the RAPL package counter every 2 ms while cmd runs with
# GGML_XDNA_PROF=1 GGML_XDNA_PROF_TRACE=<trace file> (xdna-prof.h), then
# gives each interval to the section covering its midpoint: seconds, W and
# share of the energy per section. Reading intel-rapl:0 needs no root here.
#
#   python3 power_trace.py /tmp/pt.tsv -- build/bin/llama-bench -m model.gguf \
#       -ngl 99 -fa 1 -p 4096 -ub 4096 -b 4096 -n 0 -r 3
import sys, time, subprocess, os, bisect, collections
trace = sys.argv[1]; cmd = sys.argv[sys.argv.index("--") + 1:]
def rapl(): return int(open("/sys/class/powercap/intel-rapl:0/energy_uj").read())
env = dict(os.environ, GGML_XDNA_PROF="1", GGML_XDNA_PROF_TRACE=trace)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
S = []
while p.poll() is None:
    S.append((time.monotonic_ns(), rapl())); time.sleep(0.002)
out = p.communicate()[0]
secs = []
for l in open(trace):
    k, a, b = l.rstrip("\n").split("\t"); secs.append((int(a), int(b), k))
secs.sort(); starts = [s[0] for s in secs]
t_lo, t_hi = secs[0][0], secs[-1][1]
E = collections.defaultdict(float); T = collections.defaultdict(float)
for (ta, ea), (tb, eb) in zip(S, S[1:]):
    if eb == ea and tb - ta < 5e6: continue          # RAPL not yet updated: merge
    m = (ta + tb) // 2
    if m < t_lo or m > t_hi: continue
    i = bisect.bisect_right(starts, m) - 1
    k = secs[i][2] if i >= 0 and secs[i][1] >= m else "(between)"
    E[k] += (eb - ea) / 1e6; T[k] += (tb - ta) / 1e9
tot_e = sum(E.values()); tot_t = sum(T.values())
print(f"window {tot_t:.2f} s  {tot_e / tot_t:.1f} W avg")
for k in sorted(T, key=lambda k: -E[k]):
    print(f"  {k:34s} {T[k]:6.2f} s  {E[k] / T[k]:5.1f} W  {100 * E[k] / tot_e:4.1f}% of energy")
for l in out.splitlines():
    if "pp" in l or "tg" in l: print("  ", l.split("|")[-3].strip(), l.split("|")[-2].strip())
