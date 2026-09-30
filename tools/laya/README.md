# llama-laya

Typed decisions (`choice`, `score`, `noul`) with the [Laya](https://github.com/NandhaKishorM/laya) checkpoints, in a single
forward pass per question. The output matches `laya.Agent.predict` / `predict_batch`.

| checkpoint | encoder |
|---|---|
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large |
| [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | mmBERT-base |
| convaiinnovations/laya, subfolder `typed-decisions` | ModernBERT-large |

## Convert

Point the converter at a checkpoint directory (`model.safetensors`, `rl_agent_config.json`, `encoder/`, `tokenizer/`):

```bash
hf download convaiinnovations/laya-multilingual --local-dir laya-multilingual
python convert_hf_to_gguf.py laya-multilingual --outtype f16 --outfile laya-multilingual-f16.gguf
```

Use f16 (or f32). Measured against the fp32 PyTorch reference, f16 moves the answer probabilities by at most 0.01. The
reference's own bf16 GPU path moves them by up to 0.02. `Q8_0` reaches 0.09 on out-of-distribution inputs and `Q4_K_M` 0.26,
which is enough to flip close decisions.

## Run

```bash
llama-laya -m laya-multilingual-f16.gguf -ngl 99 -f request.json
```

The request holds one state or a list of states, and the questions to answer over each of them:

```json
{
  "state": "My payment failed twice and I was charged both times. Please refund the duplicate.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this ticket?",
                   "criteria": {"billing": "payments, refunds, invoices", "technical": "bugs, outages, errors"}},
    "urgency":    {"type": "score", "instructions": "How urgent is this?",
                   "criteria": ["not urgent", "somewhat urgent", "very urgent"]},
    "refund":     {"type": "noul", "instructions": "The customer asks for a refund."}
  }
}
```

- `"states": [...]` instead of `"state"` returns a list with one result per state.
- `"max_len"` and `"head_max_len"` override the checkpoint's token budgets.
- A question's `"option_order"` (a permutation of its option indices) changes the order the options are shown in;
  the answer still lists them in their original order.
- A state can be a string, an object, or a list. Objects and lists are serialized as `json.dumps` would. Lists are
  conversations, so a list that does not fit keeps its newest turns.

The result has the same shape as laya's: `answers` (choice / score / noul with probabilities, `confidence`,
`answer_confidence` and `action.act_probability`) and `usage` (tokens, state truncation, collapsed options).

All sequences of a request are packed into forward passes of at most `-b` tokens, 2048 by default. Every sequence must fit
into one pass. Raise `-b` only for longer sequences (for example `"max_len": 8192` on the multilingual checkpoint), because
attention over a packed pass costs O(`-b`²). `--verbose-prompt` prints the token ids and raw logits of every sequence.

## Library

`llama-laya` is a thin wrapper around [common/laya.h](../../common/laya.h), which any program linking `llama-common` can use:

```cpp
#include "laya.h"

llama_model_params mparams = llama_model_default_params();
mparams.n_gpu_layers = 99;
llama_model * model = llama_model_load_from_file("laya-multilingual-f16.gguf", mparams);

llama_context_params cparams = llama_context_default_params();
common_laya_context_params(cparams); // embeddings, RANK pooling, one 2048-token ubatch per pass
llama_context * ctx = llama_init_from_model(model, cparams);

common_laya_warmup(ctx); // optional, keeps backend initialization out of the first request

auto request = nlohmann::ordered_json::parse(R"({
    "state": "My payment failed twice",
    "questions": {"department": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "technical"]}}
})");

try {
    common_laya_result res = common_laya_predict(ctx, request);
    std::string choice = res.response["answers"]["department"]["choice"];
} catch (const std::invalid_argument & e) {
    // malformed request: unknown question type, missing criteria, sequence longer than the batch, ...
} catch (const std::runtime_error & e) {
    // evaluation failed
}
```

- `res.response` is the laya response: one result for `"state"`, or a list for `"states"`.
- `res.sequences` holds the token ids, marker positions and raw act / option logits of every (state, question) pair.
- `res.n_tokens`, `res.n_passes` and `res.t_ms` describe the forward passes.

A context is not thread-safe: use one context per thread, or serialize the calls.

## How it maps to the model

The `laya` architecture is the ModernBert encoder followed by the decision head of `laya/common.py` `DecisionModel`:

- The embedding of the question type is added to every encoder output. The graph reads the question type from the token
  that follows `[CLS]`, since every sequence starts with `[CLS] <type> question: ...`.
- Two pre-norm transformer encoder layers run with bidirectional attention inside each sequence.
- The hidden state of every `[MASK]` option marker is scored into one logit per option.
- The act head reads the `[CLS]` state plus four features of the option distribution: the top probability, the margin
  to the second, the normalized entropy, and the option count / 255.

With `LLAMA_POOLING_TYPE_RANK` (the GGUF default), `llama_get_embeddings_seq` returns `n_act + 255` floats per sequence:
the act logits, then the option logits in marker order. The sequence layout, option budget, calibration temperatures and
answer decoding live in `common/laya.cpp`. The temperatures and budgets come from the `laya.decision.config` metadata.

For mmBERT, the Metaspace pre-tokenizer is applied in `common/laya.cpp`. That pre-tokenizer prepends a space to every segment
between added tokens and starts a new piece at every space. The GGUF stores the byte-fallback BPE as an SPM vocabulary
scored by merge rank.

## Checking against the reference

```bash
pip install torch "transformers>=5" git+https://github.com/NandhaKishorM/laya
python tools/laya/compare-laya.py --model-dir laya-multilingual --gguf laya-multilingual-f16.gguf --bin build/bin/llama-laya --ngl 99
```

The script requires the token sequences to be identical, then compares the raw logits and the decoded answers.

## Not supported

`Router` (checkpoint selection by language), `predict_long`, `predict_shortlist`, `decide` (structured schemas), hooks,
and per-language temperatures (`lang_temperatures`) belong to the Python library and are not part of this tool.
