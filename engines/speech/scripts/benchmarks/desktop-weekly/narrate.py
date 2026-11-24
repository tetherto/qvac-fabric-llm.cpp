#!/usr/bin/env python3
"""Ask Claude for a short engineer-facing narrative of a benchmark report.

usage: narrate.py --report report.md --out narrative.md [--model claude-opus-5-5]

Reads ANTHROPIC_API_KEY from the environment and calls the Messages API over plain HTTPS (the bench machines run
the stdlib only, so no SDK is installed). Without a key, or on any API error, it writes nothing and exits 0 so the
weekly run and its e-mail go out without the narrative.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 8192
TIMEOUT_SECONDS = 300
STOP_REFUSAL = "refusal"
SYSTEM_PROMPT = (
    "You summarize a weekly performance report of the qvac-fabric-speech.cpp engines for the engineers who maintain them. "
    "Write at most twelve short lines of plain text: the overall state, every failed or missing cell, every metric that "
    "moved by more than the report's threshold with the most likely cause given the pins that changed, and anything a "
    "maintainer should look at this week. Quote numbers from the report. Do not restate tables, do not add headings."
)


def build_request(report_text, model):
    return {"model": model, "max_tokens": MAX_TOKENS, "system": SYSTEM_PROMPT, "fallbacks": "default",
            "messages": [{"role": "user", "content": f"Here is this week's report:\n\n{report_text}"}]}


def request_headers(api_key):
    return {"content-type": "application/json", "x-api-key": api_key, "anthropic-version": API_VERSION,
            "anthropic-beta": FALLBACK_BETA}


def call_api(payload, api_key):
    request = urllib.request.Request(API_URL, data=json.dumps(payload).encode(), headers=request_headers(api_key))
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.load(response)


def narrative_text(response):
    if response.get("stop_reason") == STOP_REFUSAL:
        return ""
    return "\n".join(block["text"] for block in response.get("content", []) if block.get("type") == "text").strip()


def narrate(report_text, model, api_key, transport=None):
    send = transport or call_api
    return narrative_text(send(build_request(report_text, model), api_key))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=os.environ.get("NARRATIVE_MODEL") or DEFAULT_MODEL)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("narrate.py: no ANTHROPIC_API_KEY, skipping the narrative", file=sys.stderr)
        return
    with open(args.report) as handle:
        report_text = handle.read()
    try:
        text = narrate(report_text, args.model, api_key)
    except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
        print(f"narrate.py: {error}", file=sys.stderr)
        return
    if text:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
