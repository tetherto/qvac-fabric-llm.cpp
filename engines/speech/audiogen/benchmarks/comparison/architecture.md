# ACE-Step comparability notes

QVAC `audiogen` is a ggml port of acestep.cpp, not an independent
reimplementation of the Python ACE-Step 1.5 stack. NOTICE maps files such as
`philox.h`, `lm_pipeline`, BPE, FSM, Qwen3 graphs, DiT, and the FSQ
detokenizer to the upstream C++ sources. Stage GGUFs are produced by
acestep.cpp `convert.py`.

## What can be matched

| Item | Match strategy |
|---|---|
| Architecture | ACE-Step 1.5 turbo text2music: LM codes -> FSQ detok -> text/cond encoders -> Euler DiT -> Oobleck VAE |
| Weights | Same four GGUFs on disk for both CLIs (0.6B LM Q8_0, turbo DiT Q8_0, embedding Q8_0, VAE BF16) |
| Output | 48 kHz stereo PCM. QVAC writes PCM16 WAV. acestep.cpp is forced to `output_format=wav16` and `peak_clip=0` so it does not MP3-encode or percentile-clip |
| Seed | QVAC uses one `seed` for LM `mt19937` and DiT Philox. acestep.cpp JSON sets both `seed` and `lm_seed` to the manifest value |
| Duration, lyrics, bpm, key, language | Explicit in every prompt so Phase 1 lyric generation is skipped |
| LM sampling | temperature 0.85, top_p 0.9, top_k 0, cfg 2.0 |
| DiT | 8 steps, shift 3.0, guidance 1.0, Euler. QVAC turbo has no DiT CFG; acestep.cpp also resolves turbo guidance to 1.0 |
| DCW | QVAC defaults to Haar double mode (0.05 / 0.02). acestep.cpp JSON defaults disable DCW (`dcw_scaler=0`). The harness enables the QVAC scalers on both sides |
| Threads | `--threads N` on QVAC; acestep.cpp uses ggml defaults unless a CLI flag exists — recorded as an asymmetry if logs show a different thread count |

## What cannot be matched

| Asymmetry | Effect |
|---|---|
| ggml pin | QVAC links the fabric ggml (`qvac-fabric` port, the `ggml/` tree of qvac-fabric-llm.cpp). acestep.cpp vendors its own ggml submodule with SNAKE and COL2IM_1D. Kernel-level equivalence is not claimed |
| Process shape | QVAC: one `music-cli` process, lazy stage load/unload. acestep.cpp: `ace-lm` then `ace-synth`, each loading its modules |
| LM size defaults | acestep.cpp README downloads 4B LM; this harness uses 0.6B so the weights are shared |
| `timesignature` spelling | acestep.cpp documents the CoT field as a numerator (`"4"`). QVAC CLI examples use `"4/4"`. The manifest uses `"4"` for both |
| Metal on macOS | Default acestep.cpp builds enable Metal. CPU runs require `GGML_METAL=OFF` for **both** engines. GPU fallback is a hard failure |
| Stage placement | QVAC allowlists Metal LM + detokenizer on GPU. acestep.cpp placement may differ. Logs are the source of truth |
| Cover/lego/XL | Out of scope. QVAC has MiniMax and a subset of edit ops; acestep.cpp has lego/extract/complete, LoRA, HTTP server |
| Quality vs speed | FAD is not computed without a licensed reference corpus. CLAP is an optional post-pass (`score-clap.js`) over saved WAVs and is never added to generation time or RTF |

## Comparison class

Use **implementation-level** language. Do not claim bit-identical kernels or
identical GGUF provenance with QVAC registry artifacts: registry GGUFs are
QVAC-built; this harness uses the shared Serveurperso files so both engines
read the same bytes.

Licensing: both engines are MIT. ACE-Step 1.5 weights follow the upstream
model card (MIT for the 1.5 line). Qwen3-Embedding is Apache-2.0. Generated
audio follows the model card; prompts in this suite are original and do not
request living-artist imitation.
