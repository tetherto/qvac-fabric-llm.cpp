#!/usr/bin/env python3
"""
LongBench-E task data preparation for the llama-server-based runner.

Mirrors the prompt-construction half of upstream LongBench's pred.py:

    1. Load the HF dataset split for the requested task (LongBench-E variant).
    2. For each sample, format the prompt via dataset2prompt[task].
    3. Tokenize and middle-truncate to max_length tokens if oversized (the
       "Lost in the Middle" recipe from upstream pred.py:get_pred lines 60-66).
    4. Apply the HF tokenizer's chat template for QA / summarization /
       synthetic tasks; leave raw text for trec / triviaqa / samsum / lcc /
       repobench-p (matches the upstream rule in pred.py:get_pred line 73).
    5. Write a JSONL with one prepared sample per line:

        {"input": <ready-to-send text>,
         "answers": ["gt1", ...],
         "all_classes": ["c1", ...] | null,
         "length": <upstream length field>,
         "max_gen": <upstream dataset2maxlen[task]>}

The bash runner (longbench-bench.sh) then reads this JSONL line-by-line,
POSTs "input" to llama-server's /completion endpoint with n_predict=max_gen,
and scores the response via longbench-score.py.

Splitting prep from inference keeps the bash runner small and lets us cache
prepared data per-(tokenizer, task, max_length, num_samples) without
re-tokenising on every K/V-cache sweep cell.

Usage:
    python3 tests/longbench-prepare.py \
        --task qasper \
        --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct \
        --max-length 31500 \
        --num-samples 10 \
        --output-file tests/longbench/_data/<tok_slug>/qasper.jsonl \
        --upstream-dir tests/longbench/LongBench \
        --seed 42
"""

import argparse
import json
import logging
import os
import random
import sys

logger = logging.getLogger(__name__)

# Chat-template OFF for these tasks — upstream's pred.py:get_pred line 73:
#     if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
#         prompt = build_chat(...)
# lsht is Chinese; we drop it from LongBench-E so it never reaches us.
CHAT_TEMPLATE_SKIP = {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}


def middle_truncate(tokenizer, prompt: str, max_length: int) -> str:
    """Keep the first half and last half of the tokenised prompt.

    Exact recipe from upstream pred.py:get_pred lines 60-66. We tokenise
    without truncation, slice the id list, then decode back. Cited rationale
    is the "Lost in the Middle" observation (https://arxiv.org/abs/2307.03172):
    information at either end is preserved better than information at the
    boundary, so cutting from the middle is the least-bad option.
    """
    ids = tokenizer(prompt, truncation=False, return_tensors=None)["input_ids"]
    if len(ids) <= max_length:
        return prompt
    half = max_length // 2
    head = tokenizer.decode(ids[:half], skip_special_tokens=True)
    tail = tokenizer.decode(ids[-half:], skip_special_tokens=True)
    return head + tail


def apply_chat_template_if_needed(tokenizer, task: str, prompt: str) -> str:
    if task in CHAT_TEMPLATE_SKIP:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def select_samples(
    samples,
    num_samples: int,
    seed: int,
) -> list:
    """Return up to num_samples samples, deterministically shuffled.

    num_samples < 0 means "use all" (paper preset). Otherwise we shuffle with
    a fixed seed so reruns with the same (task, num_samples, seed) get the
    same slice — important for cross-run comparability of K/V quant cells.
    """
    full = list(samples)
    if num_samples is None or num_samples < 0 or num_samples >= len(full):
        return full
    rng = random.Random(seed)
    idx = list(range(len(full)))
    rng.shuffle(idx)
    return [full[i] for i in idx[:num_samples]]


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def prepare(
    task: str,
    tokenizer_name: str,
    max_length: int,
    num_samples: int,
    output_file: str,
    upstream_dir: str,
    seed: int,
    data_dir: str,
    bench_variant: str = "longbench-e",
) -> int:
    # Defer the heavy import until after argparse so --help is fast.
    from transformers import AutoTokenizer

    cfg_path = os.path.join(upstream_dir, "config", "dataset2prompt.json")
    maxlen_path = os.path.join(upstream_dir, "config", "dataset2maxlen.json")
    if not os.path.isfile(cfg_path) or not os.path.isfile(maxlen_path):
        logger.error(
            "Upstream config files not found under %s. Did the clone succeed?",
            upstream_dir,
        )
        return 1
    with open(cfg_path) as f:
        dataset2prompt = json.load(f)
    with open(maxlen_path) as f:
        dataset2maxlen = json.load(f)

    if task not in dataset2prompt:
        logger.error("Task %r missing from dataset2prompt.json", task)
        return 1

    # Data layout matches upstream's data.zip: data/{task}.jsonl and
    # data/{task}_e.jsonl side-by-side. We pick the _e file when the caller
    # asks for LongBench-E (default, matches the paper's Table 1).
    fname = f"{task}_e.jsonl" if bench_variant == "longbench-e" else f"{task}.jsonl"
    jsonl_path = os.path.join(data_dir, fname)
    if not os.path.isfile(jsonl_path):
        logger.error(
            "Dataset file %s not found. Did the bash runner unzip data.zip?",
            jsonl_path,
        )
        return 1
    logger.info("Loading %s ...", jsonl_path)
    ds = _load_jsonl(jsonl_path)

    logger.info("Loading tokenizer %s...", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    selected = select_samples(ds, num_samples, seed)
    logger.info(
        "Task %s: %d / %d samples selected (seed=%d)",
        task,
        len(selected),
        len(ds),
        seed,
    )

    prompt_format = dataset2prompt[task]
    max_gen = dataset2maxlen[task]

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    n_written = 0
    n_truncated = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for sample in selected:
            raw_prompt = prompt_format.format(
                input=sample["input"],
                context=sample["context"],
            )
            truncated = middle_truncate(tokenizer, raw_prompt, max_length)
            if truncated is not raw_prompt:
                n_truncated += 1
            final_prompt = apply_chat_template_if_needed(tokenizer, task, truncated)

            rec = {
                "input": final_prompt,
                "answers": list(sample["answers"]),
                "all_classes": sample.get("all_classes"),
                "length": int(sample.get("length", 0)),
                "max_gen": max_gen,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    logger.info(
        "Task %s: wrote %d samples to %s (%d middle-truncated to %d tokens)",
        task,
        n_written,
        output_file,
        n_truncated,
        max_length,
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[longbench-prepare] %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--max-length", type=int, required=True)
    p.add_argument(
        "--num-samples", type=int, default=-1,
        help="Sample cap (-1 = use all; matches paper preset).",
    )
    p.add_argument("--output-file", required=True)
    p.add_argument(
        "--upstream-dir", required=True,
        help="Path to the cloned LongBench subdir (e.g. tests/longbench/LongBench).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--variant", choices=["longbench", "longbench-e"], default="longbench-e",
        help="LongBench-E used by the paper (default); LongBench-V1 for full 21-task set.",
    )
    p.add_argument(
        "--data-dir", required=True,
        help="Directory containing the unzipped data.zip JSONL files (e.g. tests/longbench/_raw/data).",
    )
    args = p.parse_args()

    return prepare(
        task=args.task,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        num_samples=args.num_samples,
        output_file=args.output_file,
        upstream_dir=args.upstream_dir,
        seed=args.seed,
        data_dir=args.data_dir,
        bench_variant=args.variant,
    )


if __name__ == "__main__":
    sys.exit(main())
