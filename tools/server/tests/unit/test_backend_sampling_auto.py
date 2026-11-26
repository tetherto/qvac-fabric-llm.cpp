import os
import pytest
from utils import *

# --backend-sampling (here from the environment) samples every request on the backend, so the automatic choice never runs
pytestmark = pytest.mark.skipif(
    os.environ.get("LLAMA_ARG_BACKEND_SAMPLING", "") in ("on", "enabled", "true", "1"),
    reason="automatic backend sampling applies only without --backend-sampling",
)

server = ServerPreset.tinyllama2()

PROMPTS = [
    "I believe the meaning of life is",
    "Once upon a time, in a land far away,",
]


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def complete(prompt: str, **params):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": 32,
        "return_tokens": True,
        **params,
    })
    assert res.status_code == 200
    return res.body


def greedy_outputs():
    outputs = []
    for prompt in PROMPTS:
        body = complete(prompt, temperature=0.0)
        outputs.append((body["content"], body["tokens"], body["generation_settings"]["backend_sampling"]))
    return outputs


def test_greedy_auto_matches_cpu_sampling():
    global server
    server.start()
    auto = greedy_outputs()
    server.stop()

    server = ServerPreset.tinyllama2()
    server.no_backend_sampling_auto = True
    server.start()
    cpu = greedy_outputs()

    for (content_a, tokens_a, backend_a), (content_c, tokens_c, backend_c) in zip(auto, cpu):
        assert backend_a is True
        assert backend_c is False
        assert tokens_a == tokens_c
        assert content_a == content_c


@pytest.mark.parametrize("params", [
    {"temperature": 0.8, "seed": 42},
    {"temperature": 0.0, "n_probs": 3},
    {"temperature": 0.0, "repeat_penalty": 1.2},
    {"temperature": 0.0, "logit_bias": [[" the", -2.0]]},
    {"temperature": 0.0, "grammar": "root ::= [a-z ]+"},
])
def test_auto_falls_back_to_cpu_sampling(params):
    global server
    server.start()
    body = complete(PROMPTS[0], **params)
    assert body["generation_settings"]["backend_sampling"] is False


def test_auto_and_cpu_requests_alternate():
    global server
    server.start()
    first = complete(PROMPTS[0], temperature=0.0)
    complete(PROMPTS[1], temperature=0.8, seed=42)
    second = complete(PROMPTS[0], temperature=0.0)
    assert first["generation_settings"]["backend_sampling"] is True
    assert second["generation_settings"]["backend_sampling"] is True
    assert first["tokens"] == second["tokens"]


def test_request_backend_sampling_false_opts_out():
    global server
    server.start()
    auto = complete(PROMPTS[0], temperature=0.0)
    cpu = complete(PROMPTS[0], temperature=0.0, backend_sampling=False)
    assert auto["generation_settings"]["backend_sampling"] is True
    assert cpu["generation_settings"]["backend_sampling"] is False
    assert auto["tokens"] == cpu["tokens"]


def test_request_backend_sampling_true_greedy_matches_cpu_sampling():
    global server
    server.start()
    cpu = complete(PROMPTS[0], temperature=0.0, backend_sampling=False)
    backend = complete(PROMPTS[0], temperature=0.0, backend_sampling=True)
    assert backend["generation_settings"]["backend_sampling"] is True
    assert backend["tokens"] == cpu["tokens"]


def test_n_probs_with_auto_reports_cpu_probabilities():
    global server
    server.start()
    auto = complete(PROMPTS[0], temperature=0.0, n_probs=3)
    server.stop()

    server = ServerPreset.tinyllama2()
    server.no_backend_sampling_auto = True
    server.start()
    cpu = complete(PROMPTS[0], temperature=0.0, n_probs=3)

    assert all(len(p["top_logprobs"]) == 3 for p in auto["completion_probabilities"])
    assert auto["completion_probabilities"] == cpu["completion_probabilities"]
