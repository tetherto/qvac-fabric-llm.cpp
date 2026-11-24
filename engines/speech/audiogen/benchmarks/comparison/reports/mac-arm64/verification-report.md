# macOS ARM64 artifact verification

## Hardware and operating system

The generated reports identify `darwin-arm64`. Exact host model, CPU, memory,
and operating-system version were not retained with this historical run.

## Source revisions

The report predates source-revision capture. The reviewed external revisions
associated with this comparison series are qvac-ext-ggml
`0a76e3ed969781da6de41d6c9a1c3fc471c0978b` and acestep.cpp
`9761469d95fc204b5468623c68a1a2203e50b1f9`; the artifacts do not independently
prove those revisions.

## Toolchain and build flags

The CPU report used Metal-disabled builds. The GPU report identifies Metal.
Compiler, CMake, and detailed build flags were not retained.

## Models

Both JSON reports contain the four expected filenames and matching SHA-256
hashes for the Q8_0 text encoder, Q8_0 0.6B LM, Q8_0 turbo DiT, and BF16 VAE.

## Workload

Both backends used the committed manifest, four threads, one warm-up, three
timed runs, alternating engine order, and a 5000 ms cooldown. CLAP used
`laion/larger_clap_music_and_speech`, caption-only text, and CPU scoring.

## Backend verification

The CPU and Metal artifacts report their requested backends for both engines.
No fallback failure is present in the retained round records.

## Result validation

Each backend contains 30 successful timed rounds per engine and no failed,
malformed, or silent output. Every prompt has one deterministic WAV hash across
its three timed repetitions. CLAP scores are present for timed rounds.

## Limitations and sign-off

This is an implementation-level comparison with the process-shape and ggml-pin
asymmetries documented in `../../architecture.md`. Hardware, toolchain, and
source revisions were not captured, so the reports are retained as historical
results rather than fully reproducible machine provenance.

- Review basis: committed JSON, Markdown, and prompt manifest
- Decision: internally consistent; provenance limitations disclosed
