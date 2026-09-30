#!/usr/bin/env python3
"""Compare llama-laya against the reference laya Agent (PyTorch).

    pip install torch "transformers>=5" git+https://github.com/NandhaKishorM/laya
    python tools/laya/compare-laya.py --model-dir path/to/laya --gguf laya.gguf --bin build/bin/llama-laya

For every request both implementations build the token sequences, which must match exactly, and
the raw option / act logits and the decoded answers are compared.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np

REQUESTS = [
    {
        "states": [
            "My payment failed twice and I was charged both times. Please refund the duplicate.",
            "The app crashes on startup since the last update.  It shows   error 0x80070005.\n\nPlease help!",
        ],
        "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle this ticket?",
                           "criteria": {"billing": "payments, refunds, invoices", "technical": "bugs, outages, errors",
                                        "sales": "pricing, upgrades"}},
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["not urgent", "somewhat urgent", "very urgent"]},
            "refund": {"type": "noul", "instructions": "The customer asks for a refund."},
        },
    },
    {
        # structured state, list labels of mixed scalar types, structured criteria, custom noul labels
        "states": [
            {"subject": "Invoice #4411", "amount": 129.5, "items": [1, 2.0, 1e-05, 1e16, True, None],
             "body": "Hi team,\n\tthe invoice [MASK] is wrong — please fix it ASAP \U0001F600"},
        ],
        "questions": {
            "labels": {"type": "choice", "instructions": {"task": "pick one", "n": 3}, "criteria": ["a", 1, 2.5, False, "True"]},
            "tier": {"type": "choice", "instructions": "Which plan fits?",
                     "criteria": {"free": "", "pro": {"price": 20, "seats": [1, 5]}, "enterprise": None}},
            "rubric": {"type": "score", "instructions": "Rate the tone", "criteria": [{"desc": "rude"}, "neutral", 3]},
            "valid": {"type": "noul", "instructions": "The invoice total is plausible.",
                      "criteria": {"true": "the amount looks right"}, "labels": {"false": "no", "true": "yes"}},
        },
    },
    {
        # conversation list: truncated from the left when too long
        "states": [[{"role": "user", "content": "hello " * 300}, {"role": "assistant", "content": "How can I help?"},
                    {"role": "user", "content": "cancel my subscription please"}]],
        "questions": {
            "intent": {"type": "choice", "instructions": "What does the user want now?",
                       "criteria": ["cancel", "upgrade", "greeting", "other"]},
        },
    },
    {
        # many options: the head budget caps every option
        "states": ["I would like to book a flight from Berlin to Tokyo next Tuesday, window seat."],
        "questions": {
            "intent": {"type": "choice", "instructions": "Classify the intent.",
                       "criteria": {f"intent_{i}": f"a fairly long description of intent number {i} that uses many tokens"
                                    for i in range(40)}},
        },
    },
    {
        "states": [
            "मेरा भुगतान दो बार विफल हो गया और मुझसे दोनों बार शुल्क लिया गया। कृपया डुप्लिकेट राशि वापस करें।",
            "我的付款失败了两次，但两次都被扣款了。请退还重复的款项。",
            "Esqueci minha senha e não consigo entrar na conta.   Podem ajudar?",
            "  leading spaces, a <unk> token, [CLS] and <bos> text, and trailing spaces   ",
        ],
        "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle this ticket?",
                           "criteria": ["billing", "technical", "account"]},
            "refund": {"type": "noul", "instructions": "The customer asks for a refund."},
        },
    },
    {
        # whitespace: newlines and tabs next to words, runs of spaces, a literal U+2581
        "states": [
            "line one\nline two\n\n\nline three\n",
            "\tindented\ttext \n  next  line\r\nwindows line",
            "a\u2581b \u2581\u2581c   \n\n  d",
            "\n\nstarts with newlines",
        ],
        "questions": {
            "lines": {"type": "score", "instructions": "How many lines\nare there?", "criteria": ["one", "two", "three or more"]},
            "flag": {"type": "noul", "instructions": "The text\tcontains tabs."},
        },
    },
]


def run_reference(agent, req):
    states, questions = req["states"], req["questions"]
    ids = list(questions.keys())
    for qid in ids:
        agent._check_question(qid, questions[qid])
    internal = {qid: agent._to_internal(questions[qid]) for qid in ids}

    rows = []
    for s, state in enumerate(states):
        items = agent._encode_state(state, ids, internal)
        for qid, item in zip(ids, items):
            rows.append({"state": s, "question": qid, "ids": item["ids"], "markers": item["markers"], "item": item})

    import torch
    from laya.common import collate_items
    for row in rows:
        b = collate_items([[row["item"]]], agent.tok.pad_token_id)
        with torch.no_grad():
            logits, act = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        k = len(row["markers"])
        row["logits"] = logits[0, :k].float().numpy()
        row["act"] = act[0].float().numpy()

    # compare the answers as JSON, as llama-laya prints them
    results = json.loads(json.dumps(agent.predict_batch(states, questions), ensure_ascii=False))
    return rows, results


def run_llama(args, req):
    cmd = [args.bin, "-m", args.gguf, "-p", json.dumps(req), "--verbose-prompt", "-t", str(args.threads), "-b", "16384"]
    if args.ngl is not None:
        cmd += ["-ngl", str(args.ngl)]
    cmd += args.extra
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        raise RuntimeError("llama-laya failed")
    rows = []
    for line in proc.stderr.splitlines():
        pos = line.find("laya_row: ")
        if pos >= 0:
            rows.append(json.loads(line[pos + len("laya_row: "):]))
    return rows, json.loads(proc.stdout)


def compare_answers(ref, out, tol):
    worst = 0.0
    mismatches = []
    for qid, a in ref["answers"].items():
        b = out["answers"][qid]
        if a["type"] == "choice" and a["choice"] != b["choice"]:
            mismatches.append(f"{qid}: choice {a['choice']!r} != {b['choice']!r}")
        for key in ("score", "noul", "confidence", "answer_confidence"):
            if key in a:
                worst = max(worst, abs(a[key] - b[key]))
        worst = max(worst, abs(a["action"]["act_probability"] - b["action"]["act_probability"]))
        for k, v in a.get("probabilities", {}).items():
            worst = max(worst, abs(v - b["probabilities"][k]))
    if ref["usage"] != out["usage"]:
        mismatches.append(f"usage {ref['usage']} != {out['usage']}")
    if worst > tol:
        mismatches.append(f"answers differ by {worst:.4f}")
    return worst, mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="Laya checkpoint directory (model.safetensors, rl_agent_config.json, ...)")
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--bin", default="build/bin/llama-laya")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ngl", type=int, default=None)
    # the reference's own bf16 GPU path moves the probabilities by up to ~2e-2 from its fp32 path
    parser.add_argument("--tol", type=float, default=5e-3, help="tolerance on the decoded probabilities")
    parser.add_argument("extra", nargs="*", help="extra llama-laya arguments, after --")
    args = parser.parse_args()

    from laya import Agent
    agent = Agent(args.model_dir, device="cpu")

    failed = False
    for n, req in enumerate(REQUESTS):
        ref_rows, ref_results = run_reference(agent, req)
        out_rows, out_results = run_llama(args, req)
        if not isinstance(out_results, list):
            out_results = [out_results]

        assert len(ref_rows) == len(out_rows), (len(ref_rows), len(out_rows))
        max_logit = 0.0
        max_act = 0.0
        for a, b in zip(ref_rows, out_rows):
            where = f"request {n} state {a['state']} question {a['question']}"
            if a["ids"] != b["ids"]:
                failed = True
                i = next((i for i, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y), min(len(a["ids"]), len(b["ids"])))
                print(f"FAIL {where}: tokens differ at {i}: ref {a['ids'][max(0, i - 3):i + 5]} llama {b['ids'][max(0, i - 3):i + 5]} "
                      f"(lengths {len(a['ids'])} / {len(b['ids'])})")
                continue
            max_logit = max(max_logit, float(np.abs(a["logits"] - np.array(b["logits"])).max()))
            # the act head is saturated (logits in the thousands), so compare relative to its scale
            max_act = max(max_act, float(np.abs(a["act"] - np.array(b["act"])).max() / max(1.0, np.abs(a["act"]).max())))

        worst = 0.0
        for ref, out in zip(ref_results, out_results):
            w, mismatches = compare_answers(ref, out, args.tol)
            worst = max(worst, w)
            for m in mismatches:
                failed = True
                print(f"FAIL request {n}: {m}")

        print(f"request {n}: {len(ref_rows)} rows, max |logit diff| {max_logit:.2e}, max rel act logit diff {max_act:.2e}, "
              f"max answer diff {worst:.4f}")

    print("FAILED" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
