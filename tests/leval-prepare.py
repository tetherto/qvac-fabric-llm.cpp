#!/usr/bin/env python3
"""
L-Eval closed-ended data preparation for a llama-server-based runner.

L-Eval's per-row schema (from L4NLP/LEval and OpenLMLab/LEval/LEval-data):

    {
        "instructions": ["q1", "q2", ...],   # multiple queries per doc
        "outputs":      ["a1", "a2", ...],   # references aligned with instructions
        "input":        "<long document>",
        "source":       "<domain>",
        "evaluation":   "exam" | "ngram" | "f1" | "human" | "LLM"
    }

We expand each row into N inference samples — one per instruction — so the
runner can score them independently. The document is shared across the row's
N samples, which means a smarter implementation could amortise the long
prefix across them via the server's prompt cache; we deliberately set
cache_prompt=false in the runner to keep each sample independent (which
the upstream baseline also does, since each Q gets its own model.generate
call in Baselines/*).

Per-task generation budget (max_new_tokens upstream uses):
    multi-choice (tpo, quality, coursera) → 16 tokens   (answer is letters)
    gsm100                                → 64 tokens   ("The answer is N")
    codeU                                 → 64 tokens   (short output)
    sci_fi                                → 64 tokens   ("true [fact: false]")
    topic_retrieval_longchat              → 64 tokens

Prompt template — task-specific instruction prefix + the long doc + the
question. Matches the prompt shape used by upstream's Baselines/*.py
(they prepend the prefix in code; the dataset rows themselves are just
input + instructions + outputs).
"""

import argparse
import json
import logging
import os
import random
import sys

logger = logging.getLogger(__name__)

# How many tokens we let the model emit per task. Numbers picked to match
# the upstream Baselines/*.py defaults (most use max_new_tokens=512 but
# closed-ended answers are short; cap further for throughput).
TASK_MAX_GEN = {
    "tpo":                      16,
    "quality":                  16,
    "coursera":                 32,
    "gsm100":                   64,
    "codeU":                    64,
    "sci_fi":                   64,
    "topic_retrieval_longchat": 64,
}

# Per-task instruction prefix. Pulled from the L-Eval paper §3 prompt
# templates and the upstream Baselines/*.py snippets. The {document} and
# {question} placeholders are filled in below.
TASK_PROMPT = {
    "tpo": (
        "Read the following lecture and answer the question. "
        "Choose the single most likely answer.\n\n"
        "Lecture:\n{document}\n\nQuestion: {question}\n\nAnswer:"
    ),
    "quality": (
        "Read the following story and answer the multi-choice question. "
        "Choose the single most likely answer.\n\n"
        "Story:\n{document}\n\nQuestion: {question}\n\nAnswer:"
    ),
    "coursera": (
        "Read the following lecture transcript and answer the multi-response "
        "question. Some questions may have multiple correct answers; "
        "respond with the letters together, e.g., AB or BCD.\n\n"
        "Transcript:\n{document}\n\nQuestion: {question}\n\nAnswer:"
    ),
    "gsm100": (
        "The following are 16 examples of math problems with worked "
        "solutions. After the examples, solve the final question. "
        "Conclude your answer with 'The answer is N' where N is the number.\n\n"
        "{document}\n\nQuestion: {question}\n\nAnswer:"
    ),
    "codeU": (
        "Read the following Python code and deduce the output of running "
        "the final snippet. Answer with the printed output only.\n\n"
        "Code:\n{document}\n\nQuestion: {question}\n\nAnswer:"
    ),
    "sci_fi": (
        "Read the following science-fiction story. Then answer the "
        "true/false question, using the exact format: 'true [fact: true]' "
        "or 'false [fact: false]' etc.\n\n"
        "Story:\n{document}\n\nQuestion: {question}\n\nAnswer:"
    ),
    "topic_retrieval_longchat": (
        "Below is a multi-round conversation. Identify the topic of the "
        "requested message. Respond with the topic name only.\n\n"
        "Conversation:\n{document}\n\nQuestion: {question}\n\nAnswer:"
    ),
}


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _expand_row(row: dict) -> list[dict]:
    """Expand one (doc, [q1, q2, ...], [a1, a2, ...]) row into N samples."""
    instructions = row.get("instructions") or []
    outputs = row.get("outputs") or []
    doc = row.get("input", "")
    out = []
    for i, q in enumerate(instructions):
        if i >= len(outputs):
            break
        gt = outputs[i]
        if gt is None:
            continue
        out.append({
            "document": doc,
            "question": q,
            "reference": gt,
        })
    return out


def _middle_truncate(tokenizer, prompt: str, max_length: int) -> str:
    """Same recipe as LongBench's prepare script — keep first/last halves."""
    ids = tokenizer(prompt, truncation=False)["input_ids"]
    if len(ids) <= max_length:
        return prompt
    half = max_length // 2
    head = tokenizer.decode(ids[:half], skip_special_tokens=True)
    tail = tokenizer.decode(ids[-half:], skip_special_tokens=True)
    return head + tail


def _apply_chat_template(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _select_samples(samples, num_samples, seed):
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

    if task not in TASK_PROMPT:
        logger.error("Unknown closed-ended task: %s", task)
        return 1

    src = os.path.join(data_dir, f"{task}.jsonl")
    if not os.path.isfile(src):
        logger.error("Data file %s not found. Did the bash runner clone OpenLMLab/LEval?", src)
        return 1

    rows = _load_jsonl(src)
    logger.info("Task %s: %d rows from %s", task, len(rows), src)

    samples = []
    for r in rows:
        samples.extend(_expand_row(r))
    logger.info("Task %s: expanded to %d (doc, question) samples", task, len(samples))

    selected = _select_samples(samples, num_samples, seed)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    template = TASK_PROMPT[task]
    max_gen = TASK_MAX_GEN[task]

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    n_written = 0
    n_truncated = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for s in selected:
            raw = template.format(document=s["document"], question=s["question"])
            truncated = _middle_truncate(tokenizer, raw, max_length)
            if truncated is not raw:
                n_truncated += 1
            final_prompt = _apply_chat_template(tokenizer, truncated)
            rec = {
                "input": final_prompt,
                "references": [s["reference"]],
                "task": task,
                "max_gen": max_gen,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    logger.info(
        "Task %s: wrote %d samples to %s (%d middle-truncated to %d tokens)",
        task, n_written, output_file, n_truncated, max_length,
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[leval-prepare] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--max-length", type=int, required=True)
    p.add_argument("--num-samples", type=int, default=-1)
    p.add_argument("--output-file", required=True)
    p.add_argument("--data-dir", required=True,
                   help="Directory containing LEval-data/Closed-ended-tasks/*.jsonl files.")
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
