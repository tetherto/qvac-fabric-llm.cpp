// Typed decisions with Laya checkpoints (https://github.com/NandhaKishorM/laya)
//
// A request holds one state or a list of states, and the questions to answer over each of them:
//
//   {
//     "state": "My payment failed twice",               // or "states": [...]; a state is a string, object or list
//     "questions": {
//       "department": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "technical"]},
//       "urgency":    {"type": "score",  "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
//       "refund":     {"type": "noul",   "instructions": "The customer asks for a refund."}
//     },
//     "max_len": 512, "head_max_len": 192                 // optional, default to the checkpoint's budgets
//
// A question may also set "option_order", a permutation of its option indices: slot s of the sequence
// shows option option_order[s], and the answer is reported in the canonical option order.
//   }
//
// The response follows laya's Agent.predict (for "state") and Agent.predict_batch (for "states"):
// {"model", "answers": {id: {...}}, "usage": {...}}, or a list of them.

#pragma once

#include "llama.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

// one question over one state
struct common_laya_sequence {
    size_t                   state;    // index of the state in the request
    std::string              question; // question id

    std::vector<llama_token> tokens;
    std::vector<int32_t>     markers;  // position of each option marker
    std::vector<float>       act;      // act head logits
    std::vector<float>       logits;   // option logits, uncalibrated, in slot order (see "option_order")
};

struct common_laya_result {
    nlohmann::ordered_json response;

    std::vector<common_laya_sequence> sequences;

    int32_t n_tokens = 0; // tokens evaluated
    int32_t n_passes = 0; // forward passes
    double  t_ms     = 0.0;
};

// context parameters for common_laya_predict: every sequence must fit into one batch of n_batch tokens
void common_laya_context_params(llama_context_params & cparams, uint32_t n_batch = 2048);

// Answers a request. The sequences of all states and questions are packed into as few forward passes as the
// context batch allows. A malformed request throws std::invalid_argument, a failed evaluation std::runtime_error.
common_laya_result common_laya_predict(llama_context * ctx, const nlohmann::ordered_json & request);

// Evaluates one short decision, to initialize the backends before timing or serving requests
// (the generic warmup batch is not a laya sequence).
void common_laya_warmup(llama_context * ctx);
