#!/usr/bin/env python3
"""
NIAH (Needle In A Haystack) prompt builder.

Produces one prepared prompt per (context_length, depth_percent) cell on
the canonical gkamradt grid. The recipe:

    1. Load the haystack corpus (Paul Graham essays — RULER bundles them at
       tests/ruler/scripts/data/synthetic/json/PaulGrahamEssays.json after
       its first run; we reuse them if present, otherwise download).
    2. Tokenise the corpus and the needle separately.
    3. For each (ctx_len, depth_percent) cell:
         a. Cut a `ctx_len - needle_tokens - question_tokens` slice of the
            haystack starting at offset 0 (deterministic — same content
            for every cell so scores are directly comparable).
         b. Insert the needle's tokens at `depth_percent / 100 * (slice
            length)` along the slice. Round to a token boundary; the
            model sees a perfectly natural-looking haystack + needle.
         c. Decode the spliced token sequence back to text, append the
            question, and apply the chat template.
    4. Write one JSONL row per (ctx_len, depth) cell.

Why deterministic offsets instead of random samples per cell? gkamradt's
original visualisation evaluates *one* needle placement per cell — the
heatmap is a 2-D map of model behaviour, not a noisy expected value. The
randomness in offline NIAH harnesses (e.g., RULER's `niah_single_*`) comes
from running N independent samples at fixed depth=50%; the depth grid
itself is deterministic. We follow gkamradt: one sample per cell. The
orchestrator handles repeats if requested.

Default needle (Greg Kamradt original):
    "The best thing to do in San Francisco is eat a sandwich and sit in
     Dolores Park on a sunny day."

Default question:
    "What is the best thing to do in San Francisco?"

Default answer keys (any one match counts as correct):
    ["dolores park", "sandwich and sit"]
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_NEEDLE = (
    "The best thing to do in San Francisco is eat a sandwich and sit in "
    "Dolores Park on a sunny day."
)
DEFAULT_QUESTION = "What is the best thing to do in San Francisco?"
DEFAULT_ANSWER_KEYS = ["dolores park", "sandwich and sit"]


def _load_haystack(haystack_path: Optional[str], cache_path: Optional[str]) -> str:
    """Load the haystack corpus, preferring an explicit path then a cache.

    The RULER setup downloads Paul Graham essays into a JSON dump at
    tests/ruler/scripts/data/synthetic/json/PaulGrahamEssays.json. If the
    user has run RULER at least once, we reuse that to avoid a second
    network fetch. The JSON's top-level shape is {"text": "..."} for the
    minimal fallback, or a more structured form for the full dump; we
    handle both.
    """
    candidates = [p for p in (haystack_path, cache_path) if p]
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                # Plain text file — return as-is.
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
            if isinstance(data, dict) and "text" in data:
                return data["text"]
            if isinstance(data, list):
                # RULER's full PG essay dump: list of {"text": "..."} dicts.
                parts = []
                for item in data:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
                    elif isinstance(item, str):
                        parts.append(item)
                if parts:
                    return "\n\n".join(parts)
            if isinstance(data, str):
                return data
    raise FileNotFoundError(
        f"No haystack JSON found in any of: {candidates}. "
        "Run tests/ruler-bench.sh first to populate the corpus, or pass "
        "--haystack-path with a UTF-8 text file."
    )


def _insert_needle_at_depth(
    tokenizer,
    haystack_text: str,
    needle_text: str,
    target_tokens: int,
    depth_percent: float,
) -> str:
    """Tokenise haystack + needle, splice in, decode back.

    `target_tokens` is the desired final token budget for the combined
    (haystack + needle) blob — the chat template / question are added on
    top, so the bash runner passes a budget that leaves headroom.
    """
    needle_ids = tokenizer(needle_text, add_special_tokens=False)["input_ids"]
    needle_len = len(needle_ids)
    if needle_len >= target_tokens:
        raise ValueError(
            f"needle ({needle_len} tokens) is longer than target ({target_tokens})"
        )
    haystack_budget = target_tokens - needle_len
    haystack_ids = tokenizer(haystack_text, add_special_tokens=False)["input_ids"]
    if len(haystack_ids) < haystack_budget:
        # Repeat the haystack until we have enough text — Paul Graham
        # essays are ~700k tokens combined, so this only triggers if the
        # fallback minimal corpus is in use.
        repeats = (haystack_budget // max(1, len(haystack_ids))) + 1
        haystack_ids = haystack_ids * repeats
    haystack_ids = haystack_ids[:haystack_budget]

    # Clamp depth into [0, 1] and round to an integer insert position.
    pct = max(0.0, min(1.0, depth_percent / 100.0))
    insert_pos = int(round(pct * haystack_budget))
    spliced = haystack_ids[:insert_pos] + needle_ids + haystack_ids[insert_pos:]
    return tokenizer.decode(spliced, skip_special_tokens=True)


def _build_prompt(haystack_with_needle: str, question: str) -> str:
    # Same shape used by Greg Kamradt's original harness — a brief
    # system-style preamble followed by the question. We wrap the whole
    # thing with the model's chat template (apply_chat_template) further
    # down so the assistant has a clear "user message" boundary.
    return (
        "You are a helpful assistant. Below is a long document. After it, "
        "please answer the question that follows.\n\n"
        f"Document:\n{haystack_with_needle}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def _apply_chat_template(tokenizer, prompt: str) -> str:
    """Wrap the user prompt with the model's chat template.

    enable_thinking=False is a Qwen3-series kwarg that suppresses the
    "<think>...</think>" CoT prefix the model would otherwise emit before
    its answer. NIAH uses a 64-token generation budget, which Qwen3.5
    routinely exhausts inside the thinking phase alone, so its answer
    never reaches the substring matcher. Templates that don't reference
    enable_thinking (Llama, Mistral) silently ignore the kwarg.

    Older `transformers` releases raise TypeError on unknown kwargs to
    apply_chat_template; the try/except keeps us compatible with those.
    """
    msg = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True,
        )


def prepare(
    tokenizer_name: str,
    ctx_lengths: list[int],
    depth_percents: list[float],
    output_file: str,
    haystack_path: Optional[str],
    ruler_haystack_cache: Optional[str],
    needle: str,
    question: str,
    answer_keys: list[str],
) -> int:
    from transformers import AutoTokenizer

    haystack_text = _load_haystack(haystack_path, ruler_haystack_cache)
    logger.info("Loaded haystack: %d chars", len(haystack_text))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    assert tokenizer is not None, f"Failed to load tokenizer: {tokenizer_name}"

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    n_written = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for ctx in ctx_lengths:
            # Leave headroom for: question prompt template (~100 tokens),
            # chat-template overhead (~30 tokens for Llama-3.1), and a small
            # margin. 256 tokens of headroom is generous.
            target_tokens = ctx - 256
            if target_tokens <= len(tokenizer(needle, add_special_tokens=False)["input_ids"]):
                logger.warning("Skipping ctx_len=%d (too small for needle)", ctx)
                continue
            for depth in depth_percents:
                spliced = _insert_needle_at_depth(
                    tokenizer, haystack_text, needle, target_tokens, depth)
                raw_prompt = _build_prompt(spliced, question)
                final_prompt = _apply_chat_template(tokenizer, raw_prompt)
                rec = {
                    "input": final_prompt,
                    "references": list(answer_keys),
                    "ctx_len": ctx,
                    "depth_percent": float(depth),
                    "task": "niah",
                    "max_gen": 64,
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    logger.info("Wrote %d (ctx × depth) cells to %s", n_written, output_file)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[niah-prepare] %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--ctx-lengths", required=True,
                   help="Space-separated context lengths, e.g. '2048 4096 8192 16384'")
    p.add_argument("--depth-percents", required=True,
                   help="Space-separated depths in [0,100], e.g. '0 25 50 75 100'")
    p.add_argument("--output-file", required=True)
    p.add_argument("--haystack-path", default=None,
                   help="Optional explicit haystack file (JSON dump or plain text).")
    p.add_argument("--ruler-haystack-cache", default=None,
                   help="Path to RULER's PaulGrahamEssays.json (auto-used if it exists).")
    p.add_argument("--needle", default=DEFAULT_NEEDLE)
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument("--answer-keys", nargs="*", default=DEFAULT_ANSWER_KEYS)
    args = p.parse_args()

    ctx_lengths = [int(x) for x in args.ctx_lengths.split()]
    depth_percents = [float(x) for x in args.depth_percents.split()]

    return prepare(
        tokenizer_name=args.tokenizer,
        ctx_lengths=ctx_lengths,
        depth_percents=depth_percents,
        output_file=args.output_file,
        haystack_path=args.haystack_path,
        ruler_haystack_cache=args.ruler_haystack_cache,
        needle=args.needle,
        question=args.question,
        answer_keys=args.answer_keys,
    )


if __name__ == "__main__":
    sys.exit(main())
