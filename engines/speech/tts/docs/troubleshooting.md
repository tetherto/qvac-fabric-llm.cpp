# tts engine: troubleshooting

Part of the [tts engine documentation](../README.md).

## Troubleshooting

**`TTS_CPP_USE_SYSTEM_GGML=OFF` reports a missing `tts/ggml`** — run
`bash tts/scripts/setup-ggml.sh` to stage the pinned fabric ggml
there, or install `qvac-fabric`, set
`CMAKE_PREFIX_PATH` or use the vcpkg toolchain, and leave the option enabled.

**A CLI is not under `build/`** — umbrella builds place it under
`build/tts/`. Visual Studio builds add the selected configuration
directory, for example `engines\tts\build\Release\audio8-cli.exe`.

**CTest cannot find a Release executable on Windows** — pass `-C Release` to
CTest when the generator is multi-config.

**Supertonic streaming flags fail in `tts-cli`** — this is intentional. Use
`supertonic-cli`; the umbrella dispatcher supports only batch Supertonic.

**Parler `--greedy` warns and still samples** — argmax does not reach a
terminating sequence for this architecture. Keep sampling and set `--seed` for
reproducible runs.

**CosyVoice3 callback chunks arrive only after generation** — the callback
currently slices completed PCM. It does not provide incremental
first-audio latency.

**A fixture-backed test is disabled** — configure output names each missing
GGUF, WAV, or reference directory. Set `TTS_CPP_TEST_MODEL_DIR`,
`TTS_CPP_TEST_AUDIO_DIR`, and `TTS_CPP_TEST_REF_DIR` to populated roots, then
reconfigure. `ctest -N` lists registrations but does not execute them.

**`error: this GGUF has no embedded tokenizer`** — you're running against
a legacy T3 GGUF built before the tokenizer was embedded. Re-run the
converter to produce a fresh GGUF:

```bash
python scripts/convert-t3-turbo-to-gguf.py --out models/chatterbox-t3-turbo.gguf
```

**`warning: s3gen GGUF lacks variant keys`** — you're running against a
legacy S3Gen GGUF produced before variant metadata was added. The defaults (`meanflow=true, n_timesteps=2, cfg_rate=0`)
match the historical Turbo behaviour, so legacy Turbo GGUFs continue
to work.  For a Multilingual S3Gen GGUF, however, those defaults are
wrong and the output will be garbage — re-run the converter:

```bash
python scripts/convert-s3gen-to-gguf.py --variant mtl --out models/chatterbox-s3gen-mtl.gguf
```

**`error: --min-p must be in [0, 1]`** / `--cfg-weight must be >= 0` /
`--exaggeration must be in [0, 1]` — the MTL sampling knobs reject
out-of-range values up front instead of producing wrong-but-not-crashing
output.  Pass values inside the documented ranges (see [CLI sampling flags](cli.md#useful-flags)).

**`--debug requires --ref-dir`** — debug mode substitutes Python-dumped
random bits to make every intermediate tensor bit-exactly comparable.
Run `python scripts/dump-s3gen-reference.py --out artifacts/s3gen-ref …`
first, then pass `--ref-dir artifacts/s3gen-ref`.

**Output is much louder than the Python reference** — expected: the Python
reference dump uses a very short utterance (mostly silence). Generate a
longer sentence and compare RMS. Differences up to ~2.5 % in spectrogram
magnitude are from the stochastic SineGen excitation (non-bit-exact RNG
between `std::mt19937` and `torch.rand`).

**Slower than real-time** — make sure you built a Release configuration,
requested an available backend, and selected an appropriate `--threads` value.
Several engines cap automatic threads at four, while Supertonic selects its
default by backend and graph path and Pocket uses separate CPU workers. See
[per-surface defaults](api.md#defaults-that-differ-by-surface); more threads are
not automatically faster.
