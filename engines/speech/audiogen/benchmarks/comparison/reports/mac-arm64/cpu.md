# ACE-Step engine comparison (cpu)

Generated: 2026-08-20T08:10:26.607Z

Comparison class: **implementation-level**. Platform: `darwin-arm64`. Threads: 4. Warm-ups: 1. Timed runs: 3. Order: `alternate`. Cooldown: 5000 ms.

Both engines used the same GGUF files, prompt manifest, duration, seed, LM sampling defaults, Euler sampler, and Haar DCW double-mode scalers. QVAC `music-cli` is a single process. `acestep.cpp` is `ace-lm` then `ace-synth`. QVAC lazy-loads stages inside `generate()`; acestep.cpp loads modules in each CLI process.

## Overall

| Engine | Success | Fail | Median gen ms | Median e2e ms | Median RTF | Peak RSS | Unique WAV hashes | Silent | Median CLAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acestep.cpp | 30 | 0 | 15895.3 | 15895.3 | 1.740 | 3276.7 MiB | 10 | 0 | 0.267 |
| qvac | 30 | 0 | 17765.5 | 17970.8 | 1.931 | 1709.1 MiB | 10 | 0 | 0.368 |

## Visualizations

```mermaid
xychart-beta
  title "Median generation time (lower is better)"
  x-axis ["acestep.cpp","qvac"]
  y-axis "Milliseconds" 0 --> 19542.1
  bar [15895.3, 17765.5]
```

```mermaid
xychart-beta
  title "Median real-time factor (lower is better)"
  x-axis ["acestep.cpp","qvac"]
  y-axis "RTF" 0 --> 2.124
  bar [1.74, 1.931]
```

```mermaid
xychart-beta
  title "Median peak memory (lower is better)"
  x-axis ["acestep.cpp","qvac"]
  y-axis "MiB" 0 --> 3604.4
  bar [3276.7, 1709.1]
```

```mermaid
xychart-beta
  title "Median CLAP score (higher is better)"
  x-axis ["acestep.cpp","qvac"]
  y-axis "Cosine similarity" 0 --> 0.405
  bar [0.267, 0.368]
```

### Median generation time by prompt

```mermaid
xychart-beta
  title "acestep.cpp by prompt"
  x-axis ["acoustic-chamber","adversarial-clash","dense-electronic","dense-orchestral","instrumental-folk","instrumental-house","short-drone","unusual-prepared","vocal-jazz","vocal-pop"]
  y-axis "Milliseconds" 0 --> 26122.3
  bar [12772, 13444.4, 15603.9, 23747.5, 13372.4, 19350.7, 10099.6, 16292.6, 19638.8, 16234.3]
```

```mermaid
xychart-beta
  title "qvac by prompt"
  x-axis ["acoustic-chamber","adversarial-clash","dense-electronic","dense-orchestral","instrumental-folk","instrumental-house","short-drone","unusual-prepared","vocal-jazz","vocal-pop"]
  y-axis "Milliseconds" 0 --> 31948.4
  bar [14549, 14315, 17872, 29044, 13900, 23934, 9834, 17650, 22221, 19579]
```

## acoustic-chamber

| Engine | Success | Fail | Median gen ms | Median e2e ms | Median RTF | Peak RSS | Unique WAV hashes | Silent | Median CLAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acestep.cpp | 3 | 0 | 12772.0 | 12772.0 | 1.774 | 3276.2 MiB | 1 | 0 | 0.225 |
| qvac | 3 | 0 | 14549.0 | 14753.5 | 1.956 | 1710.2 MiB | 1 | 0 | 0.408 |

## adversarial-clash

| Engine | Success | Fail | Median gen ms | Median e2e ms | Median RTF | Peak RSS | Unique WAV hashes | Silent | Median CLAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acestep.cpp | 3 | 0 | 13444.4 | 13444.4 | 1.867 | 3276.1 MiB | 1 | 0 | 0.206 |
| qvac | 3 | 0 | 14315.0 | 14516.1 | 1.988 | 1705.9 MiB | 1 | 0 | 0.154 |

## dense-electronic

| Engine | Success | Fail | Median gen ms | Median e2e ms | Median RTF | Peak RSS | Unique WAV hashes | Silent | Median CLAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acestep.cpp | 3 | 0 | 15603.9 | 15603.9 | 1.696 | 3268.6 MiB | 1 | 0 | 0.259 |
| qvac | 3 | 0 | 17872.0 | 18081.1 | 1.893 | 1709.9 MiB | 1 | 0 | 0.368 |

## dense-orchestral

| Engine | Success | Fail | Median gen ms | Median e2e ms | Median RTF | Peak RSS | Unique WAV hashes | Silent | Median CLAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acestep.cpp | 3 | 0 | 23747.5 | 23747.5 | 1.562 | 3274.3 MiB | 1 | 0 | 0.428 |
| qvac | 3 | 0 | 29044.0 | 29262.9 | 1.911 | 2005.9 MiB | 1 | 0 | 0.423 |

## instrumental-folk

| Engine | Success | Fail | Median gen ms | Median e2e ms | Median RTF | Peak RSS | Unique WAV hashes | Silent | Median CLAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acestep.cpp | 3 | 0 | 13372.4 | 13372.4 | 1.797 | 3274.7 MiB | 1 | 0 | 0.256 |
| qvac | 3 | 0 | 13900.0 | 14101.8 | 1.773 | 1707.1 MiB | 1 | 0 | 0.310 |

## instrumental-house

| Engine | Success | Fail | Median gen ms | Median e2e ms | Median RTF | Peak RSS | Unique WAV hashes | Silent | Median CLAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acestep.cpp | 3 | 0 | 19350.7 | 19350.7 | 1.668 | 3282.4 MiB | 1 | 0 | 0.446 |
| qvac | 3 | 0 | 23934.0 | 24148.1 | 2.092 | 1906.1 MiB | 1 | 0 | 0.454 |

## short-drone

| Engine | Success | Fail | Median gen ms | Median e2e ms | Median RTF | Peak RSS | Unique WAV hashes | Silent | Median CLAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acestep.cpp | 3 | 0 | 10099.6 | 10099.6 | 1.804 | 3277.4 MiB | 1 | 0 | 0.012 |
| qvac | 3 | 0 | 9834.0 | 10029.8 | 1.756 | 1702.9 MiB | 1 | 0 | 0.045 |

## unusual-prepared

| Engine | Success | Fail | Median gen ms | Median e2e ms | Median RTF | Peak RSS | Unique WAV hashes | Silent | Median CLAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acestep.cpp | 3 | 0 | 16292.6 | 16292.6 | 1.697 | 3269.1 MiB | 1 | 0 | 0.276 |
| qvac | 3 | 0 | 17650.0 | 17853.7 | 1.918 | 1708.5 MiB | 1 | 0 | 0.333 |

## vocal-jazz

| Engine | Success | Fail | Median gen ms | Median e2e ms | Median RTF | Peak RSS | Unique WAV hashes | Silent | Median CLAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acestep.cpp | 3 | 0 | 19638.8 | 19638.8 | 1.717 | 3278.3 MiB | 1 | 0 | 0.333 |
| qvac | 3 | 0 | 22221.0 | 22434.5 | 1.984 | 1879.8 MiB | 1 | 0 | 0.456 |

## vocal-pop

| Engine | Success | Fail | Median gen ms | Median e2e ms | Median RTF | Peak RSS | Unique WAV hashes | Silent | Median CLAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acestep.cpp | 3 | 0 | 16234.3 | 16234.3 | 1.765 | 3278.5 MiB | 1 | 0 | 0.517 |
| qvac | 3 | 0 | 19579.0 | 19780.2 | 1.990 | 1701.0 MiB | 1 | 0 | 0.517 |

Failed rounds are retained in the JSON. WAV QC and optional CLAP scores are computed from files after generation and are not included in generation time or RTF.
