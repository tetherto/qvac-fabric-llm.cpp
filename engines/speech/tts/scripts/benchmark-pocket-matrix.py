#!/usr/bin/env python3
"""Compare pinned upstream Pocket and Fabric CPU inference over an English corpus.

Run from the reference Python environment. Both implementations execute
sequentially, with one warmup and identical text, assets and sampling settings.
Seeds are NOT RNG-equivalent across implementations. Keep the machine idle.
"""
import argparse
import json
import math
import platform
from pathlib import Path
import statistics
import subprocess
import sys

CORPUS = {
    "short": "Hello! This is Pocket TTS. We are bringing natural, local speech generation to Fabric.",
    "numbers": "The temperature is twenty three point five degrees. Your appointment is on September tenth at half past nine. Please bring three copies of the report.",
    "punctuation": "Wait, are you sure? Yes! She said, 'Take a deep breath, then try again.' That's exactly what I'll do.",
    "unicode": "The café opens at eight. Zoë ordered a croissant, and René asked for a cup of tea. It’s a beautiful morning!",
    "long": "Today we are testing speech generation that runs locally on a personal computer. The model turns written text into spoken audio and begins returning sound before the entire passage has finished. This matters for conversations, reading assistance, and applications that need to respond quickly. We also want longer passages to remain clear and consistent. A useful test includes several sentences with different rhythms, short pauses, and familiar words. The listener should be able to follow every sentence without reading the original text. Finally, we will compare the time required by the original implementation with the native Fabric implementation, using the same machine and the same voice.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("native", "bundle", "weights", "config", "tokenizer", "voice", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--cases", nargs="+", choices=CORPUS, default=list(CORPUS))
    args = parser.parse_args()
    if args.runs < 2 or args.runs > 100:
        parser.error("runs must be 2..100 so both implementations include a warmup")
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 10 or not 1 <= args.steps <= 64:
        parser.error("temperature must be finite and 0..10; steps must be 1..64")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"comparison.json").unlink(missing_ok=True)
    report = {"machine": platform.platform(), "runs_per_prompt": args.runs,
              "threads_per_worker": 1, "workers": 2, "temperature": args.temperature, "steps": args.steps,
              "seed": 1234, "rng_equivalent": False, "cases": []}
    for name in dict.fromkeys(args.cases):
        text = CORPUS[name]
        directory = args.output/name
        directory.mkdir(exist_ok=True)
        commands = [
            [sys.executable, str(Path(__file__).with_name("benchmark-pocket-reference.py")),
             "--weights", str(args.weights), "--config", str(args.config),
             "--tokenizer", str(args.tokenizer), "--voice", str(args.voice),
             "--output", str(directory), "--text", text, "--runs", str(args.runs),
             "--temperature", str(args.temperature), "--steps", str(args.steps)],
            [str(args.native), "--model-dir", str(args.bundle), "--text", text,
             "--runs", str(args.runs), "--threads", "1", "--temperature", str(args.temperature),
             "--steps", str(args.steps), "--seed", "1234", "--output", str(directory/"pocket-fabric.wav"),
             "--report", str(directory/"fabric-benchmark.json")],
        ]
        for implementation, command in zip(("upstream", "fabric"), commands):
            print(f"{name}: {implementation}", flush=True)
            with (directory/(implementation+".log")).open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        case = {"name": name, "text": text}
        for implementation, filename in (("upstream", "upstream-benchmark.json"), ("fabric", "fabric-benchmark.json")):
            measured = json.loads((directory/filename).read_text())
            case[implementation] = {key: statistics.median(r[key] for r in measured["runs"])
                                    for key in ("seconds", "first_audio_seconds", "audio_seconds", "real_time_factor")}
        report["cases"].append(case)
        (args.output/"comparison.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
