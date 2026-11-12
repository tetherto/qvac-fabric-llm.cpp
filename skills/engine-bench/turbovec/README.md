# TurboVec CPU bench

A weekly CPU benchmark of qvac-fabric's vector index (`ggml-vector-index`, TurboVec 2- and 4-bit) against the upstream `turbovec` package from PyPI. It runs on the Mac, the Strix Halo box and the DGX Spark right after the Saturday LLM engine bench, so the two never share the machine.

## What it measures

Data: 100,000 random unit vectors at dim 1024 and 1,000 queries (seed 0, written once to `data/`). Every sample is a fresh process, so a cold add includes rotation and codebook setup.

| job | driver | what |
| --- | --- | --- |
| batch | `bench/run_bench.py` | cold add of 100k vectors, then one 1,000-query search (k 64), on 1 and 8 threads; median of 5 processes |
| single | `bench/run_single.py` | one query per call (as RAG does), k 10 and 64, p50 and p95 per call, on 1 and 8 threads; fabric with staged search on and off; median of 3 processes |
| recall | `bench/recall.py` | recall@1/10/64 against an exact float32 search (1 thread, k 10 and 64), and the share of top-k ids two implementations have in common |

`bench/bench_fabric.cpp` drives fabric through the C API; `bench/bench_upstream.py` drives turbovec through Python, so upstream latencies include the Python call. Staged search (both defaults) is faster per query but has lower recall at small k than a full scan (fabric with staged search off).

## Files

| path | what it is |
| --- | --- |
| `turbovec.sh` | entry point: gate, CPU-quiet wait, update, the three jobs, `summary.json`, `DONE` or `FAILED` |
| `update.sh` | builds fabric from the newest `temp-<n>` branch and installs the newest turbovec, writes `engines.env` |
| `summarize.py` | one run directory -> `summary.json` (schema 1); Python 3.9 compatible |
| `status.sh` | lock, a run in progress, builds, latest runs, trigger |
| `conf/<machine>.conf` | per-machine settings (`MACHINE`, `HW`, `BUILD_JOBS`, `PATH`), deployed as `machine.conf` |

Environment: `TV_DIR` (this folder on the machine, default `~/bench-turbovec`), `BENCH_DIR` (the LLM bench folder, default `~/bench-weekly`), `GPU_LOCK` (the lock the LLM bench uses, default `/tmp/gpu-bench.lock`).

## Builds

- **qvac-fabric:** `update.sh` picks the `temp-<n>` branch with the highest n, shallow-clones it into `src/fabric`, builds only the `ggml-vector-index` target (Release, every GPU backend off) and compiles `bench_fabric.cpp` against it into `bin/fabric-<sha>`. It adds `-DBENCH_HAS_THREADS` and `-DBENCH_HAS_CALIBRATE` when the header has those calls; a build without them runs only the 1-thread full-scan rows. It rebuilds only when the branch tip moves and keeps the current and one previous build.
- **turbovec:** the newest stable PyPI release, with numpy, in `venvs/turbovec-<ver>` (uv when present, else venv + pip). Its python runs the drivers.

A failed update keeps the previous build; `runs/<date>/updates.txt` says what changed.

## Deploy

From the repository root (about 0.7 GB on the machine: 0.4 GB data, the fabric source, the venv):

```sh
H=<host>; M=<mac-m3-ultra|strix-halo|dgx-spark>; S=skills/engine-bench/turbovec
ssh $H 'mkdir -p ~/bench-turbovec'
scp -r $S/turbovec.sh $S/update.sh $S/status.sh $S/summarize.py $S/bench $H:bench-turbovec/
scp $S/conf/$M.conf $H:bench-turbovec/machine.conf
ssh $H 'cd ~/bench-turbovec && mkdir -p setup && ./update.sh setup'   # first build, a few minutes
```

## Run

- `turbovec.sh`: full run now, into `runs/<date>`.
- `turbovec.sh --quick`: pipeline test, no updates, one sample, 1 thread, into `runs/<date>-quick`.
- `turbovec.sh --after-llm`: the weekly trigger. It runs only on a Saturday (Zurich), once the LLM run `$BENCH_DIR/runs/<date>` has `DONE` or `FAILED`, and once per date. Chain it after `weekly.sh` in the LLM bench's trigger (the cron line, the Mac's `ssh-entry.sh`, the Strix `scheduler.sh`).

Start it detached (`nohup ./turbovec.sh < /dev/null > /dev/null 2>&1 &`). One run at a time (a lock directory, as macOS has no `flock`). It waits until `GPU_LOCK` is free and the 1-minute load is under max(2, cores / 8), at most 6 hours, then holds `GPU_LOCK` for the run. A full run takes about 11 min on the Mac, 5 min on the Spark and 3 min on Strix, plus the build when the branch moved.

## Outputs

`runs/<date>/`:
- `batch.md`, `single.md`, `recall.md` (readable tables) and their `.json` rows;
- `summary.json`: `machine`, `kind` (`weekly` or `test`), `status` (`ok`, `partial` when a job failed, `failed`), `versions` (fabric's `api` flag says whether threads and staged search exist), `batch` (bits, threads, impl, search `staged` or `full scan`, add_ms, search_ms, qps, top1), `single` (bits, k, threads, impl, search, p50_us, p95_us), `recall` (bits, k, impl, search, r1, r10, r64), `same` (top-k overlap per pair), `load` (max and mean 1-minute load during the run, and the threshold), `errors`, `jobs`;
- `turbovec.log`, `load.log` (load every 30 s), `logs/<job>.log`, `DONE` or `FAILED`.

A `load.max` above the threshold means something else ran meanwhile: re-run before trusting the timings. The top-1 id of query 0 should match across implementations.
