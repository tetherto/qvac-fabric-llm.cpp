#!/usr/bin/env python3
"""
ZeroSCROLLS validation-split data preparation for a llama-server-based runner.

Mirrors the prompt-construction half of upstream tau-nlp/zero_scrolls'
experiments/hf/run_hf_model.py:

    1. Read the upstream's per-task validation.jsonl (downloaded + unzipped
       by zeroscrolls-bench.sh, one file per task under tests/zeroscrolls/_raw/).
    2. Group rows by `id` — NarrativeQA and Qasper ship multiple references
       per input (separate rows, same `id`, different `pid` + `output`).
       The orchestrator scores against max-over-references per sample.
    3. If the tokenised input exceeds max_length, apply upstream's
       suffix-preserving truncation: keep the prefix as much as fits, then
       insert `truncation_seperator` and tack on the original query suffix
       (everything from `query_start_index` onward). Verbatim port of
       process_model_input() in run_hf_model.py.
    4. Wrap the resulting (possibly truncated) input with the model's HF
       chat template; ZeroSCROLLS inputs are zero-shot prompts written for
       T5-style decoder-only inference, so wrapping in Llama-3.1-Instruct's
       chat format is needed for the instruct model to recognise them as
       user messages.
    5. Write JSONL with one prepared sample per line:

        {"input":      <ready-to-send text>,
         "references": ["gt1", ...],
         "id":         "<unique sample id>",
         "task":       "<task name>",
         "max_gen":    1024}

We use the *validation* split because ZeroSCROLLS is a hidden-test-set
benchmark — `output` is None for every test-split row, and per the upstream
license / submission protocol all official scoring happens server-side.
Validation has ~20 labelled samples per task, which is enough for cache-type
comparison even if it's noisy for absolute headline numbers.

Per-task generation budget is 1024 tokens, matching run_hf_model.py's
default. This is generous for QA (which usually answers in 1–10 tokens)
but necessary for summarization tasks.
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict

logger = logging.getLogger(__name__)

# Upstream's run_hf_model.py uses max_new_tokens=1024 for every task. For
# llama-server we set this per-sample so the orchestrator's n_predict
# matches what the upstream baseline used.
MAX_GEN_TOKENS = 1024


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _group_by_id(rows: list[dict]) -> list[dict]:
    """Collapse multi-reference rows into one record per unique id.

    NarrativeQA / Qasper have the same `id` appearing multiple times with
    different `pid` and `output`. We take the first row's `input` (all rows
    with the same id have identical inputs) and collect all `output`s into
    a list. Other tasks already have one row per id; collapsing is a no-op.
    """
    by_id: dict[str, dict] = {}
    refs: dict[str, list] = defaultdict(list)
    for r in rows:
        rid = r["id"]
        if rid not in by_id:
            by_id[rid] = r
        if r.get("output") is not None:
            refs[rid].append(r["output"])
    out = []
    for rid, base in by_id.items():
        rec = dict(base)
        rec["references"] = refs[rid]
        out.append(rec)
    return out


def _suffix_preserving_truncate(
    tokenizer,
    sample: dict,
    max_length: int,
) -> str:
    """Truncate input to max_length tokens, keeping the query suffix intact.

    Exact port of upstream pred.py:process_model_input. The query lives at
    the END of `input` (at character `query_start_index`); we always keep
    the full `truncation_seperator + query` suffix, and fit as much of the
    preceding (header + document) prefix as the remaining budget allows.

    If `query_start_index` is missing (rare in summarization tasks), fall
    back to keeping the last 30% of the tokens — same heuristic upstream
    uses indirectly by setting query_start_index to the document end.
    """
    text = sample["input"]
    ids_full = tokenizer(text, truncation=False)["input_ids"]
    if len(ids_full) <= max_length:
        return text

    sep = sample.get("truncation_seperator", "") or ""
    q_idx = sample.get("query_start_index")
    if q_idx is None or q_idx <= 0 or q_idx >= len(text):
        # No explicit query index — keep tail 30% as a sensible default.
        q_idx = int(len(text) * 0.7)

    suffix_text = f"{sep.strip()}\n\n{text[q_idx:].strip()}\n"
    suffix_ids = tokenizer(suffix_text, truncation=False)["input_ids"]
    # Reserve room for the suffix; trim the prefix tokenisation to fit.
    prefix_budget = max_length - len(suffix_ids)
    if prefix_budget <= 0:
        # Pathological: even the suffix alone exceeds max_length. Drop the
        # separator and keep only the bare query.
        bare_q = text[q_idx:].strip()
        bare_q_ids = tokenizer(bare_q, truncation=False)["input_ids"][:max_length]
        return tokenizer.decode(bare_q_ids, skip_special_tokens=True)
    prefix_text = text[:q_idx]
    prefix_ids = tokenizer(prefix_text, truncation=False)["input_ids"][:prefix_budget]
    head = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    return head + suffix_text


def _apply_chat_template(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _select_samples(samples: list[dict], num_samples: int, seed: int) -> list[dict]:
    full = list(samples)
    if num_samples is None or num_samples < 0 or num_samples >= len(full):
        return full
    rng = random.Random(seed)
    idx = list(range(len(full)))
    rng.shuffle(idx)
    return [full[i] for i in idx[:num_samples]]


def prepare(
    task: str,
    tokenizer_name: str,
    max_length: int,
    num_samples: int,
    output_file: str,
    data_dir: str,
    seed: int,
) -> int:
    from transformers import AutoTokenizer

    jsonl_path = os.path.join(data_dir, task, "validation.jsonl")
    if not os.path.isfile(jsonl_path):
        logger.error(
            "Dataset file %s not found. Did the bash runner unzip %s.zip?",
            jsonl_path, task,
        )
        return 1

    logger.info("Loading %s ...", jsonl_path)
    rows = _load_jsonl(jsonl_path)
    grouped = _group_by_id(rows)
    logger.info(
        "Task %s: %d raw rows → %d unique samples (validation split)",
        task, len(rows), len(grouped),
    )

    logger.info("Loading tokenizer %s ...", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    selected = _select_samples(grouped, num_samples, seed)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    n_written = 0
    n_truncated = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for sample in selected:
            if not sample.get("references"):
                # No labelled output — skip (can happen if the validation
                # file gets accidentally swapped with test).
                continue
            truncated = _suffix_preserving_truncate(tokenizer, sample, max_length)
            if truncated is not sample["input"]:
                n_truncated += 1
            final_prompt = _apply_chat_template(tokenizer, truncated)
            rec = {
                "input": final_prompt,
                "references": list(sample["references"]),
                "id": sample["id"],
                "task": task,
                "max_gen": MAX_GEN_TOKENS,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    logger.info(
        "Task %s: wrote %d samples to %s (%d truncated to %d tokens)",
        task, n_written, output_file, n_truncated, max_length,
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[zeroscrolls-prepare] %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--max-length", type=int, required=True)
    p.add_argument("--num-samples", type=int, default=-1)
    p.add_argument("--output-file", required=True)
    p.add_argument("--data-dir", required=True,
                   help="Directory containing the per-task validation.jsonl files.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    return prepare(
        task=args.task,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        num_samples=args.num_samples,
        output_file=args.output_file,
        data_dir=args.data_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
