#!/usr/bin/env python3
"""Native-style bench of MLX speech engines for run-family.sh (macOS only).

Loads the model once from a pinned Hugging Face snapshot, decodes the clip with soundfile outside the timed region,
runs the untimed warm-ups and the timed transcriptions, and writes the bench JSON the driver reads for native families
(inference_ms, rtf_median, backend, transcript). With --once it transcribes the clip a single time and prints the text,
for time-wrapped families. The TTS engine (any mlx-audio text-to-speech model: Chatterbox Turbo, Audio8, MOSS-TTS)
runs only with --once: it synthesizes --text into --out, with --ref-audio and --ref-text for voice cloning and the
generate options of --options (a JSON object; its seed, default 42, seeds MLX), for a time-wrapped family. The music
engine (mlx-audio MiniMax Music3) also runs only with --once: it renders --caption and --lyrics for --duration seconds
and writes --out when given.
"""
import argparse
import json
import pathlib
import statistics
import sys
import time

PARAKEET = "parakeet-mlx"
WHISPER = "mlx-whisper"
NEMOTRON = "mlx-audio-nemotron"
MLX_AUDIO_TTS = "mlx-audio-tts"
MLX_AUDIO_MUSIC = "mlx-audio-music"
ENGINES = (PARAKEET, WHISPER, NEMOTRON, MLX_AUDIO_TTS, MLX_AUDIO_MUSIC)
TTS_ENGINES = (MLX_AUDIO_TTS,)
MUSIC_ENGINES = (MLX_AUDIO_MUSIC,)
SEED_OPTION = "seed"
DEFAULT_MUSIC_SECONDS = 30
BACKEND = "Metal"
REVISION_MARKER = ".revision"
MS_PER_S = 1000.0
BYTES_PER_MIB = 1024 * 1024
LANGUAGE = "en"
LANGUAGE_TAG = "en-US"
DEFAULT_RATE = 16000
SEED = 42


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=ENGINES, required=True)
    parser.add_argument("--repo", required=True, help="Hugging Face repo of the MLX conversion")
    parser.add_argument("--revision", required=True, help="commit sha of the conversion")
    parser.add_argument("--model-dir", required=True, help="where the snapshot is kept (downloaded on the first run)")
    parser.add_argument("--wav", help="clip to transcribe (ASR engines)")
    parser.add_argument("--text", help="prompt to synthesize (TTS engines)")
    parser.add_argument("--out", help="WAV written by a TTS engine")
    parser.add_argument("--ref-audio", help="reference recording for voice cloning (TTS engines)")
    parser.add_argument("--ref-text", help="transcript of the reference recording")
    parser.add_argument("--options", default="{}", help="JSON object of generate() options; seed is taken out and used to seed MLX")
    parser.add_argument("--caption", help="style caption of a music request (music engines)")
    parser.add_argument("--lyrics", help="lyrics of a music request; [instrumental] for none")
    parser.add_argument("--duration", type=float, default=DEFAULT_MUSIC_SECONDS, help="requested seconds of music")
    parser.add_argument("--bench-runs", type=int, default=5)
    parser.add_argument("--bench-warmup", type=int, default=1)
    parser.add_argument("--bench-json")
    parser.add_argument("--once", action="store_true", help="transcribe or synthesize once (time-wrapped mode)")
    return check_args(parser.parse_args(argv))


def is_asr(engine):
    return engine not in TTS_ENGINES and engine not in MUSIC_ENGINES


def check_args(args):
    if args.engine in TTS_ENGINES and not (args.once and args.text and args.out):
        raise SystemExit("a TTS engine needs --once, --text and --out")
    if args.engine in MUSIC_ENGINES and not (args.once and args.caption and args.lyrics):
        raise SystemExit("a music engine needs --once, --caption and --lyrics")
    if is_asr(args.engine) and not args.wav:
        raise SystemExit("an ASR engine needs --wav")
    if is_asr(args.engine) and not args.once and not args.bench_json:
        raise SystemExit("--bench-json is required without --once")
    return args


def pinned(model_dir, revision):
    marker = pathlib.Path(model_dir) / REVISION_MARKER
    return marker.is_file() and marker.read_text(encoding="utf-8").strip() == revision


def fetch_snapshot(repo, revision, model_dir):
    if pinned(model_dir, revision):
        return
    from huggingface_hub import snapshot_download
    snapshot_download(repo, revision=revision, local_dir=model_dir)
    (pathlib.Path(model_dir) / REVISION_MARKER).write_text(revision + "\n", encoding="utf-8")


def read_clip(path):
    import soundfile
    samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return samples.mean(axis=1), rate


def check_rate(rate, expected):
    if rate != expected:
        raise SystemExit(f"clip sample rate {rate} differs from the model's {expected}")


def load_parakeet(model_dir):
    from parakeet_mlx import from_pretrained
    return from_pretrained(model_dir)


def parakeet_transcriber(model, samples):
    import mlx.core as mx
    from parakeet_mlx.audio import get_logmel
    audio = mx.array(samples)
    return lambda: model.generate(get_logmel(audio, model.preprocessor_config))[0].text


def whisper_transcriber(model_dir, samples):
    import mlx_whisper
    return lambda: mlx_whisper.transcribe(samples, path_or_hf_repo=model_dir, language=LANGUAGE, verbose=None)["text"]


def whisper_rate():
    from mlx_whisper.audio import SAMPLE_RATE
    return SAMPLE_RATE


def nemotron_transcriber(model_dir, samples, rate):
    import mlx.core as mx
    from mlx_audio.stt import load
    model = load(model_dir)
    check_rate(rate, getattr(getattr(model, "preprocessor_config", None), "sample_rate", DEFAULT_RATE))
    audio = mx.array(samples)
    return lambda: model.generate(audio, language=LANGUAGE_TAG, chunk_duration=None).text


def build_transcriber(engine, model_dir, samples, rate):
    if engine == PARAKEET:
        model = load_parakeet(model_dir)
        check_rate(rate, model.preprocessor_config.sample_rate)
        return parakeet_transcriber(model, samples)
    if engine == NEMOTRON:
        return nemotron_transcriber(model_dir, samples, rate)
    check_rate(rate, whisper_rate())
    return whisper_transcriber(model_dir, samples)


def synthesized_audio(model, text, **options):
    import numpy as np
    return np.concatenate([np.array(result.audio) for result in model.generate(text, **options)])


def resampled(samples, rate, target_rate):
    if rate == target_rate:
        return samples
    import soxr
    return soxr.resample(samples, rate, target_rate)


def reference_options(args, model):
    if not args.ref_audio:
        return {}
    samples, rate = read_clip(args.ref_audio)
    return {"ref_audio": resampled(samples, rate, model.sample_rate), "ref_text": args.ref_text}


def generate_options(args):
    options = dict(json.loads(args.options))
    seed = options.pop(SEED_OPTION, SEED)
    return seed, options


def backend_name():
    import mlx.core as mx
    return BACKEND if mx.default_device().type == mx.DeviceType.gpu else "CPU"


def announce_backend():
    line = f"backend: {backend_name()}"
    print(line, file=sys.stderr)
    print(line)


def synthesize(args):
    import mlx.core as mx
    import soundfile
    from mlx_audio.tts.utils import load_model
    announce_backend()
    model = load_model(args.model_dir)
    seed, options = generate_options(args)
    mx.random.seed(seed)
    audio = synthesized_audio(model, args.text, **options, **reference_options(args, model))
    soundfile.write(args.out, audio, model.sample_rate)
    print(f"wrote {args.out}: {len(audio) / model.sample_rate:.2f} s at {model.sample_rate} Hz")


def timed_call(fn):
    start = time.perf_counter()
    text = fn()
    return (time.perf_counter() - start) * MS_PER_S, text


def warm_up(fn, count):
    for _ in range(count):
        fn()


def timed_runs(fn, count):
    results = [timed_call(fn) for _ in range(count)]
    return [ms for ms, _ in results], results[-1][1] if results else ""


def peak_memory_mib():
    import mlx.core as mx
    return mx.get_peak_memory() / BYTES_PER_MIB


def bench_json(args, load_ms, times, text, audio_seconds):
    median = statistics.median(times)
    return {"engine": args.engine, "backend": BACKEND, "model_repo": args.repo, "model_revision": args.revision, "load_ms": round(load_ms, 1),
            "inference_ms": {"median": round(median, 3), "min": round(min(times), 3), "max": round(max(times), 3), "all": [round(t, 3) for t in times]},
            "rtf_median": median / MS_PER_S / audio_seconds, "audio_seconds": audio_seconds, "runs": len(times), "warmup": args.bench_warmup,
            "transcript": text.strip(), "peak_memory_mib": round(peak_memory_mib(), 1)}


def write_json(path, payload):
    pathlib.Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_once(fn):
    announce_backend()
    print(fn().strip())


def run_bench(args, fn, load_ms, audio_seconds):
    warm_up(fn, args.bench_warmup)
    times, text = timed_runs(fn, args.bench_runs)
    payload = bench_json(args, load_ms, times, text, audio_seconds)
    write_json(args.bench_json, payload)
    print(json.dumps({k: payload[k] for k in ("engine", "inference_ms", "rtf_median", "transcript")}))


def rendered_music(model, args, options):
    import numpy as np
    result = next(model.generate(text=args.caption, lyrics=args.lyrics, duration=args.duration, **options))
    return np.array(result.audio), result.sample_rate


def write_music(out, audio, rate):
    if out:
        import soundfile
        soundfile.write(out, audio, rate)


def render_music(args):
    import mlx.core as mx
    from mlx_audio.music import load
    announce_backend()
    model = load(args.model_dir)
    seed, options = generate_options(args)
    mx.random.seed(seed)
    audio, rate = rendered_music(model, args, {SEED_OPTION: seed, **options})
    write_music(args.out, audio, rate)
    print(f"rendered {len(audio) / rate:.2f} s at {rate} Hz")


def run_asr(args):
    samples, rate = read_clip(args.wav)
    start = time.perf_counter()
    fn = build_transcriber(args.engine, args.model_dir, samples, rate)
    load_ms = (time.perf_counter() - start) * MS_PER_S
    if args.once:
        return run_once(fn)
    run_bench(args, fn, load_ms, len(samples) / rate)


def main(argv=None):
    args = parse_args(argv)
    fetch_snapshot(args.repo, args.revision, args.model_dir)
    if args.engine in TTS_ENGINES:
        return synthesize(args)
    if args.engine in MUSIC_ENGINES:
        return render_music(args)
    run_asr(args)


if __name__ == "__main__":
    main()
