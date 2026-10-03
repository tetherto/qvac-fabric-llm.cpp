"""Save native server token IDs for identical-prefix cross-build quality replay."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

COMPLETION_TOKENS = 1024
REQUEST_TIMEOUT_S = 3600


def post(base: str, endpoint: str, body: dict) -> dict:
    request = urllib.request.Request(base.rstrip("/") + endpoint, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        result = json.load(response)
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result


def validate_tokens(tokens, count: int) -> list[int]:
    if not isinstance(tokens, list) or len(tokens) != count or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError(f"expected exactly {count} nonnegative integer token IDs")
    return tokens


def collect(args, fixture: dict, index: int, provenance: dict) -> None:
    output = Path(args.output_dir) / f"tokens-{index}.json"
    metadata = output.with_suffix(".meta.json")
    if output.exists() or metadata.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    tokenize_request = {"content": fixture["prompts"][index], "add_special": True, "parse_special": True}
    prefix = validate_tokens(post(args.base, "/tokenize", tokenize_request)["tokens"], fixture["prompt_tokens"])
    request = {"prompt": prefix, "n_predict": COMPLETION_TOKENS, "temperature": 0.0,
               "stream": False, "cache_prompt": False, "return_tokens": True, "ignore_eos": True}
    response = post(args.base, "/completion", request)
    continuation = validate_tokens(response.get("tokens"), COMPLETION_TOKENS)
    timings = response.get("timings", {})
    if timings.get("cache_n") != 0 or timings.get("prompt_n") != len(prefix) or response.get("truncated") is not False:
        raise ValueError("quality prefix was cached, truncated, or not completely evaluated")
    token_bytes = (json.dumps(prefix + continuation) + "\n").encode()
    output.write_bytes(token_bytes)
    metadata.write_text(json.dumps({"prompt_index": index, "prefix_count": len(prefix),
                                   "token_sha256": hashlib.sha256(token_bytes).hexdigest(),
                                   "provenance": provenance, "tokenize_request": tokenize_request,
                                   "request": request, "response": response}, indent=1) + "\n")
    print(json.dumps({"prompt_index": index, "output": str(output), "prefix_count": len(prefix),
                      "completion_count": len(continuation)}), flush=True)


def collect_range(args, fixture: dict, provenance: dict) -> None:
    for index in range(args.prompt_start, args.prompt_start + args.count):
        collect(args, fixture, index, provenance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()
    fixture_bytes = Path(args.prompts).read_bytes()
    fixture = json.loads(fixture_bytes)
    if args.count < 1 or args.prompt_start < 0 or args.prompt_start + args.count > len(fixture["prompts"]):
        parser.error("requested prompt range is outside the fixture")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    provenance = {"manifest": json.loads(Path(args.manifest).read_text()),
                  "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest()}
    collect_range(args, fixture, provenance)


if __name__ == "__main__":
    main()
