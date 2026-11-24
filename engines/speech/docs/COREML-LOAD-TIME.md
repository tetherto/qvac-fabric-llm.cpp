# Core ML load time measurements

Measured with `scripts/coreml/bench-load-time.py` on the workloads pinned in `scripts/coreml/bench-manifest.json`. These numbers set the Chatterbox Core ML runtime's constants (Metal cost model, warm-load hints, width set, weight-sharing accounting); the last section says where each one goes.

- Host: Apple M2, 24 GB, macOS 15.7.3 (24G419), Low Power Mode off.
- Power: the run started on AC power, but the machine was on battery from 12:35 to 14:41 UTC (`pmset -g log`). That stretch covers the second Turbo load pass, every Multilingual load row and every time-to-ready row; the sizes, memory, first Turbo load pass, Metal cost and synthesis rows ran on AC. A re-run on AC of every Multilingual load row and of one Turbo time-to-ready run per configuration matched to within 1 % (AC/battery: cold-load sum of medians 0.991, warm-load 0.996, time to ready 0.992–0.998, cached startup 0.993–0.996), so the battery stretch did not move the numbers.
- Commit `42ebde209e7f`, manifest sha256 `051eaea968c57763…`, compute units `cpu_and_ane`.
- Sidecars exported locally with `tts/scripts/export-chatterbox-coreml.py` from the pinned f32 GGUFs; tree digests are in `results.jsonl` (`env.sidecars`).

## Reading the numbers

Written by hand from the tables below (`report` regenerates only the sections between `<!-- report:NAME -->` markers and keeps this text). Spreads are MAD; N is in each table. Synthesis rows are N=3 warm runs, so treat differences under about 5 % as noise.

- **Warm synthesis against Metal** (median wall, benchmark prompt):

  | Variant | Config | Median wall (ms) | MAD (ms) | N | vs Metal |
  |---|---|--:|--:|--:|--:|
  | turbo | metal | 6,014 | 89 | 3 | - |
  | turbo | coreml-w7 | 3,295 | 28 | 3 | -45.2 % |
  | turbo | coreml-w3 | 3,590 | 28 | 3 | -40.3 % |
  | turbo | coreml-no-encoder | 3,412 | 3 | 3 | -43.3 % |
  | mtl | metal | 9,010 | 383 | 3 | - |
  | mtl | coreml-w7 | 6,181 | 892 | 3 | -31.4 % |
  | mtl | coreml-w3 | 6,553 | 130 | 3 | -27.3 % |
  | mtl | coreml-no-encoder | 5,637 | 189 | 3 | -37.4 % |

  Multilingual with 7 widths clears a 30 % warm gain only narrowly and with a wide spread (5,289 to 7,847 ms); with 3 widths it does not (-27.3 %), because S3Gen takes 2,920 ms against 1,811 ms when long sentences are padded to the 1536-frame function. Turbo clears 30 % with either set.
- **Width set.** Three widths cut the cold total from 201.5 s to 78.4 s on Turbo and from 235.8 s to 97.7 s on Multilingual (about 2.5 times), halve the warm load of each CFM function (0.47 s against 1.08 s) and shrink the CFM `model.mil` from 9.2 MB to 3.9 MB. The price is warm synthesis: +9 % wall on Turbo, +6 % on Multilingual, and the S3Gen regression above on long Multilingual sentences.
- **Warm load is paid per function and scales with the multifunction size.** A seven-width CFM function costs about 1.08 s to load against 0.47 s in the three-width package, whatever its width; the prediction time on top is 23 to 450 ms. `expected_warm_load_ms` runs from 94 ms (smallest encoder) to 1.53 s (largest CFM).
- **Weight sharing.** `phys_footprint` grows by 0.5 to 4 MB for each additional function of a stage, so the fit charge is small, but that counter does not see weights. On `cpu_and_ane`, additional **CFM** functions add about 158 to 175 MB each of host wired memory (a CFM `weight.bin` is 139.9 MB), so the ANE does not share weights between CFM functions; additional **encoder** functions add 0 MB, so the encoder functions do share (the first one costs 95 to 133 MB, with one run reading 0 and another 497 MB because the first-function reading is noisy). On `cpu_only`, resident size grows by 156 to 233 MB for each additional function, about one `weight.bin` each, which also reads as not shared (resident counts mappings, so this is an upper bound on physical memory). Available-memory drops are noisy (one run reads -1,680 MB for the first function) and support the same CFM picture only loosely.
  - Consequence for the memory envelope: charge every resident CFM function about one `weight.bin` of ANE memory, and treat the encoder family as one weight copy.
- **Admission gate.** With everything requested at once, the gate costs +4.6 s on Turbo (206.0 s against 201.4 s, +2.3 %) and +3.2 s on Multilingual (238.6 s against 235.4 s, +1.4 %); the spread is under 1 s, so the difference is real but small. Memory admission never deferred a load (0.0 s `deferred(memory)` in every run). The worst prepare wait is 5.1 s gated and 5.5 s ungated; the worst load wait (186 to 232 s) is the serial ANE compile of the functions ahead in line, not the gate.
- **Time to ready is dominated by the cold ANE compile**: the prebuilt route (202.5 s Turbo, 233.9 s Multilingual) is no faster than the generate stand-in, and no prepare job waited more than 5.5 s for its turn, so the serial ANE compile and load of the 15 functions, not preparation, sets the time. Three widths bring the prebuilt time to ready to 79.5 s and 99.0 s. Once the plans are cached, all 15 functions are ready 9.9 s (Turbo) and 10.6 s (Multilingual) after a new process starts (2.3 s and 2.7 s for three widths).
- **Encoder fallback.** With the encoder on Metal, Turbo's T3 takes 2,600 ms against 2,386 ms with the encoder on Core ML (+9 %) and the wall time is +3.6 %; Multilingual shows no such penalty (T3 4,330 against 4,950 ms, within its run-to-run spread). Warm T3 under overlap is slower than on Metal alone for Multilingual with the encoder on Core ML (4,950 against 4,259 ms).
- **Metal cost constants** fit with R² 0.94 to 0.997 and a worst measured-to-fitted ratio of at most 1.18, over the frame ranges shown (T_mu 376 to 820 on Multilingual, 566 to 1,200 on Turbo). Several intercepts are negative, so the estimate must be clamped at zero outside those ranges.
- **Available memory** on the idle bench host stayed at 11.3 to 11.4 GB for 120 s, well above every artifact's compile reserve, which is why nothing deferred.

Limits of this run: the generate route is a stand-in (copy and compile), not template-driven generation; the bytes of the e5rt CPU and GPU bundles are not measured (the ANE cache is not measurable); and the first-function wired and available readings are noisy at N=3.

<!-- report:setup -->
## Setup and usage

Requirements:

- Apple silicon, macOS 15 or later, Xcode command-line tools (`xcrun clang++`) and Python 3.9 or later. The published numbers come from an otherwise idle M2 on macOS 15.7.3; `run` refuses battery power unless `--allow-battery` is given.
- The Chatterbox GGUFs pinned in `bench-manifest.json` (`variants`, with SHA-256 and Hugging Face revisions) under the models root: `--models-root DIR`, or `QVAC_BENCH_MODELS`.
- The Core ML sidecars for each width set in `sidecar_sets`, exported from the pinned f32 GGUFs with `tts/scripts/export-chatterbox-coreml.py` as `exporter` in the manifest describes (`--frames` selects the widths, `--compile` writes the `.mlmodelc` next to the `.mlpackage`).
- A `tts-cli` built with `-DTTS_CPP_COREML=ON` (see `tts/docs/chatterbox.md`).

Commands:

```bash
python3 -B scripts/coreml/test_bench_load_time.py     # hermetic tests, no Core ML needed
scripts/coreml/bench-load-time.py --dry-run --check-models   # manifest, pinned files and the run plan
scripts/coreml/bench-load-time.py build-driver
scripts/coreml/bench-load-time.py smoke                # fresh-host.sh identity and e5rt cache clean-up
scripts/coreml/bench-load-time.py run --tts-cli <build>/tts-cli --out <dir>   # all measurements
scripts/coreml/bench-load-time.py run ttr --variant turbo --out <dir>   # one measurement, one variant
scripts/coreml/bench-load-time.py report --results <dir> --out docs/COREML-LOAD-TIME.md
scripts/coreml/bench-load-time.py cleanup              # remove leftover bench identities and caches
```

`run` writes one JSON object per line to `<dir>/results.jsonl`; `report` renders the measurements from it. Cold runs use a fresh `fresh-host.sh` identity and a fresh copy of the sidecars, and delete both afterwards.

`report --out` on a new file writes the whole document. On an existing file it replaces only the generated sections, each between a `<!-- report:NAME -->` and a `<!-- /report:NAME -->` line, and keeps everything else (the header and the hand-written reading of the numbers); it refuses a file without those markers.

<!-- /report:setup -->

<!-- report:sizes -->
## Multifunction spec sizes

What a template would embed is the program (`model.mil` compiled, `model.mlmodel` in the package), not the weights. MB = 10^6 bytes.

| Variant | Set | Stage | Functions | model.mil | zlib | LZFSE | model.mlmodel | zlib | LZFSE | weight.bin |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| turbo | w7 | encoder | 7 | 1.20 | 0.09 | 0.07 | 1.12 | 0.11 | 0.07 | 165.87 |
| turbo | w7 | cfm | 7 | 9.19 | 0.77 | 0.95 | 9.07 | 0.83 | 0.93 | 139.90 |
| turbo | w7 | hift | 1 | 0.48 | 0.04 | 0.05 | 0.45 | 0.04 | 0.05 | 35.48 |
| turbo | w3 | encoder | 3 | 0.51 | 0.04 | 0.04 | 0.48 | 0.05 | 0.04 | 114.53 |
| turbo | w3 | cfm | 3 | 3.94 | 0.33 | 0.41 | 3.89 | 0.36 | 0.40 | 139.90 |
| mtl | w7 | encoder | 7 | 1.20 | 0.09 | 0.07 | 1.12 | 0.11 | 0.07 | 165.87 |
| mtl | w7 | cfm | 7 | 9.19 | 0.77 | 0.94 | 9.07 | 0.83 | 0.94 | 139.90 |
| mtl | w7 | hift | 1 | 0.48 | 0.04 | 0.05 | 0.45 | 0.04 | 0.05 | 35.48 |
| mtl | w3 | encoder | 3 | 0.51 | 0.04 | 0.04 | 0.48 | 0.05 | 0.04 | 114.53 |
| mtl | w3 | cfm | 3 | 3.94 | 0.33 | 0.41 | 3.89 | 0.36 | 0.40 | 139.90 |

<!-- /report:sizes -->

<!-- report:width-set -->
## Width set: 3 vs 7

Cold = sum of first loads on a fresh identity (ANE compile included), all functions of the set incl. HiFT; with several cold runs, the median of their sums. Warm = median per-function load + first prediction in a new process. Synthesis = warm `tts-cli` on the benchmark prompt (median of the runs).

| Variant | Set | Functions | Cold total (s) | Warm load+predict, median over functions (ms) | Synthesis wall (ms) | S3Gen (ms) |
|---|---|--:|--:|--:|--:|--:|
| mtl | w3 | 7 | 97.7 | 236 | 6,553 | 2,920 |
| mtl | w7 | 15 | 235.8 | 316 | 6,181 | 1,811 |
| turbo | w3 | 7 | 78.4 | 234 | 3,590 | 989 |
| turbo | w7 | 15 | 201.5 | 318 | 3,295 | 855 |

<!-- /report:width-set -->

<!-- report:warm-load -->
## `expected_warm_load_ms` per function

A warm load is a new process on an identity whose ANE plans are cached; the hint's `warm_load_ms` is load plus the warm-up prediction, which is what this column reports. Prediction = median of the later predictions.

| Variant | Set | Stage | Function | Cold load (ms) | Warm load (ms) | expected_warm_load_ms (median) | Max | Prediction (ms) |
|---|---|---|---|--:|--:|--:|--:|--:|
| mtl | w3 | cfm | frames_1024 | 24,799 | 474 | 672 | 676 | 195.5 |
| mtl | w3 | cfm | frames_1536 | 25,834 | 473 | 924 | 953 | 448.8 |
| mtl | w3 | cfm | frames_512 | 27,313 | 481 | 541 | 573 | 54.0 |
| mtl | w3 | encoder | tokens_256 | 2,869 | 74 | 95 | 95 | 16.7 |
| mtl | w3 | encoder | tokens_512 | 3,132 | 79 | 150 | 174 | 67.5 |
| mtl | w3 | encoder | tokens_768 | 4,184 | 87 | 236 | 254 | 145.4 |
| mtl | w3 | hift | main | 9,577 | 73 | 119 | 146 | 43.0 |
| mtl | w7 | cfm | frames_1024 | 28,775 | 1,085 | 1,283 | 1,319 | 195.4 |
| mtl | w7 | cfm | frames_1280 | 28,883 | 1,080 | 1,395 | 1,404 | 308.9 |
| mtl | w7 | cfm | frames_1536 | 29,051 | 1,083 | 1,531 | 1,555 | 447.5 |
| mtl | w7 | cfm | frames_512 | 30,957 | 1,083 | 1,146 | 1,163 | 54.0 |
| mtl | w7 | cfm | frames_640 | 31,279 | 1,096 | 1,165 | 1,176 | 62.7 |
| mtl | w7 | cfm | frames_768 | 26,037 | 1,088 | 1,212 | 1,222 | 118.3 |
| mtl | w7 | cfm | frames_896 | 26,480 | 1,085 | 1,242 | 1,258 | 153.4 |
| mtl | w7 | encoder | tokens_256 | 3,068 | 152 | 176 | 180 | 16.7 |
| mtl | w7 | encoder | tokens_320 | 3,085 | 154 | 186 | 187 | 25.2 |
| mtl | w7 | encoder | tokens_384 | 3,241 | 156 | 197 | 200 | 38.4 |
| mtl | w7 | encoder | tokens_448 | 3,486 | 156 | 211 | 213 | 51.0 |
| mtl | w7 | encoder | tokens_512 | 3,456 | 170 | 238 | 246 | 67.7 |
| mtl | w7 | encoder | tokens_640 | 3,995 | 161 | 264 | 266 | 101.0 |
| mtl | w7 | encoder | tokens_768 | 4,354 | 168 | 316 | 341 | 146.4 |
| mtl | w7 | hift | main | 9,614 | 74 | 120 | 122 | 42.6 |
| turbo | w3 | cfm | frames_1024 | 19,401 | 472 | 579 | 605 | 96.4 |
| turbo | w3 | cfm | frames_1536 | 19,612 | 481 | 686 | 715 | 201.9 |
| turbo | w3 | cfm | frames_512 | 19,864 | 478 | 512 | 513 | 22.8 |
| turbo | w3 | encoder | tokens_256 | 2,769 | 74 | 94 | 94 | 16.5 |
| turbo | w3 | encoder | tokens_512 | 3,170 | 78 | 146 | 151 | 67.2 |
| turbo | w3 | encoder | tokens_768 | 3,964 | 85 | 234 | 237 | 145.2 |
| turbo | w3 | hift | main | 9,574 | 73 | 119 | 146 | 42.8 |
| turbo | w7 | cfm | frames_1024 | 22,736 | 1,075 | 1,180 | 1,196 | 96.6 |
| turbo | w7 | cfm | frames_1280 | 22,700 | 1,091 | 1,240 | 1,293 | 141.5 |
| turbo | w7 | cfm | frames_1536 | 23,363 | 1,085 | 1,291 | 1,319 | 202.8 |
| turbo | w7 | cfm | frames_512 | 23,188 | 1,080 | 1,110 | 1,129 | 22.9 |
| turbo | w7 | cfm | frames_640 | 25,178 | 1,074 | 1,119 | 1,121 | 35.1 |
| turbo | w7 | cfm | frames_768 | 24,369 | 1,075 | 1,126 | 1,159 | 45.4 |
| turbo | w7 | cfm | frames_896 | 26,087 | 1,082 | 1,142 | 1,146 | 52.2 |
| turbo | w7 | encoder | tokens_256 | 3,015 | 153 | 176 | 178 | 16.6 |
| turbo | w7 | encoder | tokens_320 | 3,030 | 153 | 181 | 186 | 25.2 |
| turbo | w7 | encoder | tokens_384 | 3,234 | 156 | 199 | 211 | 37.7 |
| turbo | w7 | encoder | tokens_448 | 3,390 | 157 | 214 | 221 | 52.1 |
| turbo | w7 | encoder | tokens_512 | 3,456 | 158 | 228 | 229 | 67.3 |
| turbo | w7 | encoder | tokens_640 | 3,872 | 162 | 265 | 286 | 98.9 |
| turbo | w7 | encoder | tokens_768 | 4,241 | 171 | 318 | 337 | 146.4 |
| turbo | w7 | hift | main | 9,648 | 73 | 118 | 141 | 42.6 |

<!-- /report:warm-load -->

<!-- report:sharing -->
## Do loaded functions share weight memory?

All functions of one stage loaded and held in one process, per compute-unit policy. Per metric: the increment of the first function, the median of each later one, and the last function loaded alone (MB). `footprint` = phys_footprint (what fit charges); `resident` adds file-backed weight pages; `wired` = host wired memory, where the ANE keeps a loaded program; `available` = the host's available-memory drop (noisy). 'shared' when functions 2..n add less than 25% of a weight.bin each.

| Variant | Set | Stage | Units | weight.bin (MB) | Metric | First | Each later | Alone | Verdict |
|---|---|---|---|--:|---|--:|--:|--:|---|
| mtl | w3 | cfm | cpu_and_ane | 139.90 | footprint | 12.01 | 3.55 | 14.45 | shared |
| mtl | w3 | cfm | cpu_and_ane | 139.90 | resident | 40.58 | 14.76 | 44.22 | shared |
| mtl | w3 | cfm | cpu_and_ane | 139.90 | wired | 41.12 | 174.82 | 179.73 | not shared |
| mtl | w3 | cfm | cpu_and_ane | 139.90 | available | 54.44 | 53.90 | 79.76 | not shared |
| mtl | w3 | cfm | cpu_only | 139.90 | footprint | 12.58 | 1.14 | 9.96 | shared |
| mtl | w3 | cfm | cpu_only | 139.90 | resident | 194.58 | 178.46 | 209.44 | not shared |
| mtl | w3 | cfm | cpu_only | 139.90 | wired | -5.11 | -3.78 | 37.63 | shared |
| mtl | w3 | cfm | cpu_only | 139.90 | available | 21.54 | 1.10 | 30.49 | not shared, shared |
| mtl | w3 | encoder | cpu_and_ane | 114.53 | footprint | 9.59 | 3.89 | 8.98 | shared |
| mtl | w3 | encoder | cpu_and_ane | 114.53 | resident | 17.10 | 5.20 | 18.68 | shared |
| mtl | w3 | encoder | cpu_and_ane | 114.53 | wired | 132.92 | 0.00 | 0.00 | not shared, shared |
| mtl | w3 | encoder | cpu_and_ane | 114.53 | available | 61.52 | 0.00 | 0.00 | not shared, shared |
| mtl | w3 | encoder | cpu_only | 114.53 | footprint | 8.88 | 2.52 | 8.98 | shared |
| mtl | w3 | encoder | cpu_only | 114.53 | resident | 115.18 | 232.75 | 300.47 | not shared |
| mtl | w3 | encoder | cpu_only | 114.53 | wired | 0.00 | 0.00 | 0.00 | shared |
| mtl | w3 | encoder | cpu_only | 114.53 | available | 0.00 | 0.00 | 0.00 | shared |
| mtl | w7 | cfm | cpu_and_ane | 139.90 | footprint | 34.93 | 2.90 | 37.73 | shared |
| mtl | w7 | cfm | cpu_and_ane | 139.90 | resident | 73.68 | 10.30 | 74.55 | shared |
| mtl | w7 | cfm | cpu_and_ane | 139.90 | wired | 56.46 | 170.19 | 162.04 | not shared |
| mtl | w7 | cfm | cpu_and_ane | 139.90 | available | -128.79 | 182.35 | -165.54 | not shared |
| mtl | w7 | cfm | cpu_only | 139.90 | footprint | 38.83 | 0.89 | 34.18 | shared |
| mtl | w7 | cfm | cpu_only | 139.90 | resident | 219.59 | 168.24 | 240.63 | not shared |
| mtl | w7 | cfm | cpu_only | 139.90 | wired | -3.78 | 5.96 | -8.85 | shared |
| mtl | w7 | cfm | cpu_only | 139.90 | available | 203.37 | 7.07 | 4.78 | not shared, shared |
| mtl | w7 | encoder | cpu_and_ane | 165.87 | footprint | 10.65 | 2.09 | 10.24 | shared |
| mtl | w7 | encoder | cpu_and_ane | 165.87 | resident | 22.99 | 3.53 | 20.64 | shared |
| mtl | w7 | encoder | cpu_and_ane | 165.87 | wired | 95.29 | 0.00 | 0.00 | not shared |
| mtl | w7 | encoder | cpu_and_ane | 165.87 | available | 137.99 | 0.00 | 0.00 | not shared |
| mtl | w7 | encoder | cpu_only | 165.87 | footprint | 12.75 | 1.22 | 7.80 | shared |
| mtl | w7 | encoder | cpu_only | 165.87 | resident | 122.49 | 164.54 | 302.63 | not shared |
| mtl | w7 | encoder | cpu_only | 165.87 | wired | 0.00 | 0.00 | 0.00 | shared |
| mtl | w7 | encoder | cpu_only | 165.87 | available | 0.00 | 0.00 | 0.00 | shared |
| turbo | w3 | cfm | cpu_and_ane | 139.90 | footprint | 9.54 | 1.57 | 11.76 | shared |
| turbo | w3 | cfm | cpu_and_ane | 139.90 | resident | 39.55 | 19.97 | 46.94 | shared |
| turbo | w3 | cfm | cpu_and_ane | 139.90 | wired | 0.00 | 157.88 | 129.11 | not shared |
| turbo | w3 | cfm | cpu_and_ane | 139.90 | available | 0.00 | 53.49 | 31.13 | not shared |
| turbo | w3 | cfm | cpu_only | 139.90 | footprint | 9.57 | 0.45 | 12.21 | shared |
| turbo | w3 | cfm | cpu_only | 139.90 | resident | 184.71 | 168.24 | 194.10 | not shared |
| turbo | w3 | cfm | cpu_only | 139.90 | wired | -46.04 | -6.16 | 34.29 | not shared, shared |
| turbo | w3 | cfm | cpu_only | 139.90 | available | 10.37 | -3.74 | 25.17 | shared |
| turbo | w3 | encoder | cpu_and_ane | 114.53 | footprint | 7.11 | 1.97 | 8.11 | shared |
| turbo | w3 | encoder | cpu_and_ane | 114.53 | resident | 16.65 | 4.91 | 17.58 | shared |
| turbo | w3 | encoder | cpu_and_ane | 114.53 | wired | 0.00 | 0.00 | 496.84 | not shared, shared |
| turbo | w3 | encoder | cpu_and_ane | 114.53 | available | 0.00 | 0.00 | 435.08 | not shared, shared |
| turbo | w3 | encoder | cpu_only | 114.53 | footprint | 8.65 | 3.60 | 7.59 | shared |
| turbo | w3 | encoder | cpu_only | 114.53 | resident | 115.46 | 233.38 | 299.45 | not shared |
| turbo | w3 | encoder | cpu_only | 114.53 | wired | 0.00 | 0.00 | 0.00 | shared |
| turbo | w3 | encoder | cpu_only | 114.53 | available | 0.00 | 0.00 | 0.00 | shared |
| turbo | w7 | cfm | cpu_and_ane | 139.90 | footprint | 36.34 | 1.65 | 36.37 | shared |
| turbo | w7 | cfm | cpu_and_ane | 139.90 | resident | 75.51 | 8.27 | 77.23 | shared |
| turbo | w7 | cfm | cpu_and_ane | 139.90 | wired | -1,679.79 | 158.26 | 148.49 | not shared |
| turbo | w7 | cfm | cpu_and_ane | 139.90 | available | -1,631.55 | 160.02 | -225.44 | not shared |
| turbo | w7 | cfm | cpu_only | 139.90 | footprint | 31.38 | 0.61 | 34.88 | shared |
| turbo | w7 | cfm | cpu_only | 139.90 | resident | 216.88 | 156.53 | 220.55 | not shared |
| turbo | w7 | cfm | cpu_only | 139.90 | wired | -25.15 | 5.78 | 4.34 | shared |
| turbo | w7 | cfm | cpu_only | 139.90 | available | 94.19 | 8.51 | 16.30 | not shared, shared |
| turbo | w7 | encoder | cpu_and_ane | 165.87 | footprint | 9.45 | 2.38 | 13.30 | shared |
| turbo | w7 | encoder | cpu_and_ane | 165.87 | resident | 20.51 | 4.31 | 23.38 | shared |
| turbo | w7 | encoder | cpu_and_ane | 165.87 | wired | 0.00 | 0.00 | 0.00 | not shared |
| turbo | w7 | encoder | cpu_and_ane | 165.87 | available | 0.00 | 0.00 | 0.00 | not shared |
| turbo | w7 | encoder | cpu_only | 165.87 | footprint | 11.16 | 0.62 | 12.24 | shared |
| turbo | w7 | encoder | cpu_only | 165.87 | resident | 122.47 | 164.51 | 306.20 | not shared |
| turbo | w7 | encoder | cpu_only | 165.87 | wired | 37.63 | 0.00 | -17.69 | shared |
| turbo | w7 | encoder | cpu_only | 165.87 | available | 23.67 | 0.00 | -67.52 | shared |

<!-- /report:sharing -->

<!-- report:metal-cost -->
## Per-variant Metal cost constants for `s3gen_estimate_metal_ms`

Least-squares `ms = a + b × frames` over the length sweep on Metal (`CHATTERBOX_COREML_DISABLE=1`), one S3Gen call per sentence after a discarded warm-up call. Encoder, CFM and the whole pipeline against `T_mu` (prompt + new frames), HiFT against `T_mel` (new frames). Worst ratio = the largest measured / fitted.

| Variant | Stage | a (ms) | b (ms/frame) | R² | Points | Frames | Worst ratio |
|---|---|--:|--:|--:|--:|---|--:|
| mtl | enc | -28.6 | 0.1619 | 0.9574 | 18 | T_mu 376–820 | 1.18 |
| mtl | cfm | -916.6 | 5.6612 | 0.9825 | 18 | T_mu 376–820 | 1.18 |
| mtl | hift | 16.7 | 3.3881 | 0.9969 | 18 | T_mel 62–506 | 1.11 |
| mtl | pipeline | -1992.6 | 9.2146 | 0.9907 | 18 | T_mu 376–820 | 1.17 |
| turbo | enc | -48.3 | 0.1601 | 0.9402 | 18 | T_mu 566–1200 | 1.17 |
| turbo | cfm | -76.6 | 0.3832 | 0.9908 | 18 | T_mu 566–1200 | 1.06 |
| turbo | hift | -2.5 | 2.4405 | 0.9957 | 18 | T_mel 66–700 | 1.09 |
| turbo | pipeline | -1347.8 | 2.9867 | 0.9964 | 18 | T_mu 566–1200 | 1.06 |

<!-- /report:metal-cost -->

<!-- report:ttr -->
## Time to ready, with and without the admission gate

Every function (encoder and CFM widths plus HiFT) requested at `minimal` in one construction group with `wait=construct`, on a fresh identity with the artifacts at a fresh path. `generate` = a stand-in prepare job per artifact (copy the package, `compileModelAtURL`), as on-device generation will; gated = the shared runtime's Manager, ungated = the same two queues without the gate. Prepare wait = issue to prepare start; load gate wait = artifact available to load start.

| Variant | Config | Runs | All ready | Time to ready median (s) | MAD (s) | Min–max (s) | Worst prepare wait (s) | Worst load wait (s) | deferred(memory) (s) | Cached startup (s) |
|---|---|--:|---|--:|--:|---|--:|--:|--:|--:|
| mtl | generate-gated | 5 | 5/5 | 238.6 | 0.4 | 238.0–239.6 | 5.1 | 224.6 | 0.0 | — |
| mtl | generate-ungated | 5 | 5/5 | 235.4 | 1.0 | 233.4–236.4 | 5.5 | 221.0 | 0.0 | — |
| mtl | prebuilt-gated | 5 | 5/5 | 233.9 | 0.7 | 233.3–241.8 | — | 232.1 | 0.0 | 10.64 |
| mtl | prebuilt-gated-w3 | 3 | 3/3 | 99.0 | 0.2 | 98.6–99.2 | — | 89.5 | 0.0 | 2.70 |
| turbo | generate-gated | 5 | 5/5 | 206.0 | 0.2 | 205.5–206.5 | 5.1 | 191.5 | 0.0 | — |
| turbo | generate-ungated | 5 | 5/5 | 201.4 | 0.0 | 201.2–201.4 | 5.5 | 186.2 | 0.0 | — |
| turbo | prebuilt-gated | 5 | 5/5 | 202.5 | 2.0 | 200.2–209.6 | — | 200.0 | 0.0 | 9.88 |
| turbo | prebuilt-gated-w3 | 3 | 3/3 | 79.5 | 0.1 | 79.5–81.5 | — | 71.8 | 0.0 | 2.33 |

<!-- /report:ttr -->

<!-- report:memory -->
## Available-memory reading under normal load

120 samples of free + inactive + purgeable + speculative pages (`host_statistics64`, the runtime's own reading): min 11,305.47 MB, median 11,377.08 MB, max 11,429.90 MB; low power False.

| Artifact | ane_compile_reserve_bytes (MB) | Deferred at the minimum reading? |
|---|--:|---|
| mtl/cfm | 223.84 | no |
| mtl/encoder | 265.39 | no |
| mtl/hift | 56.77 | no |
| turbo/cfm | 223.84 | no |
| turbo/encoder | 265.39 | no |
| turbo/hift | 56.77 | no |

<!-- /report:memory -->

<!-- report:synthesis -->
## Warm synthesis and encoder-fallback contention

`tts-cli`, benchmark prompt, built-in voice, seed 42, q8_0, a new process per run after one priming run. `coreml-no-encoder` keeps CFM and HiFT on Core ML (so S3Gen still overlaps T3) and the encoder on Metal: the T3 slowdown against `coreml-w7` is the contention the runtime pays when the encoder falls back.

| Variant | Config | Runs | Wall (ms) | T3 (ms) | S3Gen (ms) | Peak footprint (MB) |
|---|---|--:|--:|--:|--:|--:|
| mtl | coreml-no-encoder | 3 | 5,637 | 4,330 | 1,851 | 1,761.99 |
| mtl | coreml-w3 | 3 | 6,553 | 4,357 | 2,920 | 1,652.64 |
| mtl | coreml-w7 | 3 | 6,181 | 4,950 | 1,811 | 1,753.21 |
| mtl | metal | 3 | 9,010 | 4,259 | 4,190 | 1,825.40 |
| turbo | coreml-no-encoder | 3 | 3,412 | 2,600 | 887 | 1,649.81 |
| turbo | coreml-w3 | 3 | 3,590 | 2,659 | 989 | 1,434.13 |
| turbo | coreml-w7 | 3 | 3,295 | 2,386 | 855 | 1,599.31 |
| turbo | metal | 3 | 6,014 | 2,996 | 2,502 | 1,610.34 |

<!-- /report:synthesis -->

<!-- report:destinations -->
## Where each number goes

| Number | Where it goes |
|---|---|
| Spec sizes | The multifunction template size budget; LZFSE compression if a template is over it |
| 3 vs 7 widths (cold total, warm load, synthesis) | The width set the Chatterbox generation templates ship |
| `expected_warm_load_ms` per function | Each template's `template.json`; the bounded wait before a hint exists |
| Weight sharing verdict | How resident functions are charged against the Core ML memory envelope |
| Metal cost constants | `s3gen_estimate_metal_ms` in the Chatterbox Core ML runtime |
| Time to ready with / without the gate, worst prepare wait | Whether the admission gate stays on; the Chatterbox "time to all 15 ready" target |
| Available-memory reading | Confirms memory admission does not defer cold loads on the bench host |
| Encoder-fallback contention | Expected overlap when the Chatterbox encoder runs on Metal |

<!-- /report:destinations -->

