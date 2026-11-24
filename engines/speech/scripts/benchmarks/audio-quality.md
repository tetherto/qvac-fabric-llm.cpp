# Audio quality benchmarks

`audio8` measures codec reconstruction of the checked-in JFK recording using the encoder and decoder GGUFs, without the text language model. `lavasr` measures denoising and enhancement of the same recording with reproducible additive Gaussian noise (10 dB SNR, seed 42). These workloads replace the former Audio8 text-generation and LavaSR synthetic-tone benchmarks, so their performance baselines reset.

The optional `correctness` declaration supports multiple metrics:

```json
{
  "kind": "audio",
  "metrics": ["sisdr", "stoi"],
  "reference": "parakeet/test/samples/jfk.wav",
  "degradation": {"snr_db": 10, "seed": 42}
}
```

Omit `degradation` for codec reconstruction. Single kinds `sisdr` and `stoi` are also accepted. CLI arguments use `${AUDIO_INPUT}` for the prepared input and `${AUDIO_OUT}` for a unique output path for each warmup and timed run. The final timed output is scored after performance capture. Scores are not medians, and they do not gate the performance status.

Install the benchmark-only dependencies in a Python 3.11 or 3.12 virtual environment on Linux x86_64 or macOS arm64:

```sh
python3 -m venv .venv-audio-quality
.venv-audio-quality/bin/python -m pip install -r scripts/benchmarks/requirements-audio-quality.txt
export AUDIO_QUALITY_PYTHON="$PWD/.venv-audio-quality/bin/python"
```

Dependencies are pinned to official PyPI wheel SHA-256 hashes; pip rejects source distributions and unlisted artifacts. This prevents source-build execution but does not remove capabilities present inside the installed libraries.

SI-SDR uses the standard library for matching sample rates. STOI and resampling use pinned NumPy/SciPy/pystoi.

SI-SDR uses zero-mean projection and is reported in dB. STOI uses standard (not extended) STOI and is displayed as a percentage. Higher values are better for both. Establish baselines from measured outputs rather than assuming codec or enhancement quality is perfect.

The scorer accepts mono integer PCM WAV. Different rates are converted to 16 kHz with polyphase resampling, and resulting sample counts must match. It does not search for the best lag or silently crop signals. Audio8 removes only its known trailing codec padding and preserves input duration. Noise preparation scales the clean/degraded pair together if needed to avoid clipping; its metadata records the scale. Empty/missing/silent audio and undefined or infinite scores produce explicit per-metric statuses and JSON null, never a fabricated finite score.

`result.json` contains `audio_quality`, keyed by metric with `value`, `unit`, `status`, and an optional `reason`. The summary distinguishes measured zero from unavailable metrics. Artifacts include `result.reference.wav`, `result.input.wav`, `result.hypothesis.wav`, `result.audio-quality.json`, and, for degraded input, `result.input-preparation.json`. Inspect these when investigating regressions. Scoring failures preserve available WAVs.

Verification:

```sh
bash scripts/benchmarks/test-run-family.sh
python3 -B scripts/benchmarks/test_audio_quality.py
python3 -B scripts/benchmarks/test-audio-quality-driver.py
```

The desktop workflow runs model-free checks and installs audio dependencies only for affected benchmark cells. Real verification dispatches should select `audio8,lavasr` on `linux,macos` and inspect backend attribution, valid nonempty WAVs, and populated SI-SDR/STOI results. A green performance cell alone is insufficient.

## Dependency capability review

Socket reports shell-access and dynamic-code-execution capabilities in SciPy 1.11.4. These alerts are not CVE findings. The benchmark and pystoi both use `scipy.signal.resample_poly`; removing only the direct requirement would leave SciPy as a transitive dependency.

Examples reviewed in the official wheel:

- `scipy/stats/_generate_pyx.py` launches code-generation scripts with `subprocess.run` during source generation.
- `scipy/_lib/_bunch.py` uses `exec` to generate result-container classes from field names checked by `_validate_names`.
- `scipy/signal/_signaltools.py` uses `eval`/`exec` in `hilbert2`, which this benchmark does not call.

The requirements file permits only hash-verified wheels for the supported benchmark platforms, preventing source-build installation. This does not remove runtime capabilities inside SciPy and does not prove every package path safe. Retaining the dependency with these capability alerts requires security triage; no Socket suppression is included in this change.
