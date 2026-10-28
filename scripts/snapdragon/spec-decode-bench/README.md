# Speculative-decoding benchmark on an Adreno phone

`bench-spec-decode.py` runs `llama-server` builds on an adb-attached phone and measures the decode rate with and without MTP speculative decoding. `summarize-spec-decode.py` turns the results into the Markdown table used in PR descriptions.

Two builds are compared in the same session, ABBA-interleaved per row with the order flipped every row, so the GPU thermal drift hits both builds the same way. Each launch waits for the phone to cool (thermal service status 0, kgsl `thermal_pwrlevel` at most 2, GPU below 48 C, skin sensor below 36.5 C) before the server starts, and records the thermal state around every prompt.

## Device layout

```
/data/local/tmp/llama.cpp/
  base/llama-server        one build per subdirectory
  pr/llama-server
  models/*.gguf
  libc++_shared.so         from the NDK sysroot
  libomp.so                from the NDK clang runtime (lib/clang/*/lib/linux/aarch64)
```

The server runs with `LD_LIBRARY_PATH=/vendor/lib64:<device dir>` so the vendor OpenCL driver is found. Build with `-DGGML_OPENCL=ON -DLLAMA_CURL=OFF -DANDROID_STL=c++_shared` and the Khronos headers and ICD loader, see `docs/backend/OPENCL.md`.

## Usage

```sh
python3 scripts/snapdragon/spec-decode-bench/bench-spec-decode.py \
  --builds base=base,pr=pr \
  --models 2B=models/Qwen3.5-2B-Q4_K_M.gguf,4B=models/Qwen3.5-4B-Q4_K_M.gguf \
  --configs nospec,mtp1,mtp3,mtp7 --passes 2 --tag main

python3 scripts/snapdragon/spec-decode-bench/summarize-spec-decode.py spec-decode-results/main-*.jsonl
```

Server settings follow the Adreno MTP measurements: `-ngl 99 -c 4096 -np 1 --no-cache-prompt`, three prompts rendered through `/apply-template`, 256 generated tokens at temperature 0, `--spec-type draft-mtp --spec-draft-n-max N` for the MTP rows. `--env KEY=VALUE,...` exports backend switches such as `GGML_OPENCL_KQUANT_MC_GEMV=1` to the server. `--wait-for-process PATTERN` holds the run while another benchmark matching the pattern is active on the host or the phone. `--no-cool-gate` reproduces uncontrolled back-to-back runs.

Every prompt writes one JSON line with the server timings (`predicted_per_second`, `draft_n`, `draft_n_accepted`), the thermal state before and after, the load time and the cool-down wait. A sampler thread records the peak `thermal_pwrlevel` during each prompt. The summary averages only the prompts whose peak stayed below level 3 and whose rate is at least 70 % of the cell median, and reports how many prompts per cell were set aside as throttled.
