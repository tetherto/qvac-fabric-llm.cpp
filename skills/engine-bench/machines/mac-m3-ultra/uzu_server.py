#!/usr/bin/env python3
"""OpenAI-style chat completions server for Uzu, built on Mirai's `uzu` Python package (the same engine as the `mirai` CLI).

usage: uzu_server.py --model <local model dir> [--port 8098] [--host 127.0.0.1]

Why not `mirai server`: the CLI server always opens Mirai's settings store, which lives in the macOS keychain, and a cron
or SSH session on the shared Mac has no keychain, so it exits with "A default keychain could not be found". The Python
package's engine has no application identifier by default and skips that store (it only holds cloud API keys and a
download path). Building the CLI from source instead needs Xcode 26's Metal Toolchain, which is not installed there.

This mirrors what the CLI server (crates/legacy/cli/src/server in trymirai/uzu) does for the requests the bench sends:
- one chat session behind a lock, so concurrent requests queue;
- the session is reset before every request (the CLI's --no-prefix-cache);
- max_completion_tokens / max_tokens is the token limit; temperature <= 0 is greedy, otherwise stochastic with the
  request's temperature, top_p and top_k;
- enable_thinking (top level or in chat_template_kwargs) and reasoning_effort become a reasoning effort on a leading
  system message, which is where the engine reads it;
- streaming sends each step's new text (and reasoning) as deltas and the usage in the last chunk, with the CLI's usage
  fields (prompt_tokens, completion_tokens, spec_verify_ct = decode forward passes when a drafter runs).
Tool calls, response_format and the CLI's prefix cache are not implemented.
"""
import argparse
import asyncio
import json
import os
import sys
import time
import uuid

from aiohttp import web  # ty: ignore[unresolved-import]
from uzu import (ChatConfig, ChatMessage, ChatReplyConfig, ChatSessionStreamChunk, Engine,  # ty: ignore[unresolved-import]
                 EngineConfig, ReasoningEffort, SamplingMethod)

# the package's enums are not hashable, so they are matched by name
FINISH = {"Stop": "stop", "Cancelled": "stop", "Length": "length", "ContextLimitReached": "length", "ToolCalls": "tool_calls",
          "Rejected": "content_filter"}
EFFORTS = {"disabled": ReasoningEffort.Disabled, "default": ReasoningEffort.Default, "low": ReasoningEffort.Low,
           "medium": ReasoningEffort.Medium, "high": ReasoningEffort.High, "xhigh": ReasoningEffort.XHigh}
ROLES = {"user": ChatMessage.user, "system": ChatMessage.system, "assistant": ChatMessage.assistant,
         "developer": ChatMessage.developer}


class BadRequest(Exception):
    pass


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def build_messages(body):
    out = []
    for m in body.get("messages") or []:
        make = ROLES.get(m.get("role"))
        if make is None:
            raise BadRequest(f"unsupported role {m.get('role')!r}")
        out.append(make().with_text(text_of(m.get("content"))))
    if not out:
        raise BadRequest("messages is empty")
    enable = body.get("enable_thinking")
    kw = body.get("chat_template_kwargs") or {}
    if enable is None and isinstance(kw, dict):
        enable = kw.get("enable_thinking")
    effort = None
    if body.get("reasoning_effort") is not None:
        effort = EFFORTS.get(str(body["reasoning_effort"]).lower())
        if effort is None:
            raise BadRequest(f"unknown reasoning_effort {body['reasoning_effort']!r}")
    elif enable is not None:
        effort = ReasoningEffort.Default if enable else ReasoningEffort.Disabled
    if effort is not None:
        # merge into a leading system message: templates reject two system messages in a row
        if body["messages"][0].get("role") == "system":
            out[0] = out[0].with_reasoning_effort(effort)
        else:
            out.insert(0, ChatMessage.system().with_reasoning_effort(effort))
    return out


def build_config(body):
    cfg = ChatReplyConfig.create().with_token_limit(body.get("max_completion_tokens") or body.get("max_tokens"))
    t, p, k = body.get("temperature"), body.get("top_p"), body.get("top_k")
    if t is not None and t <= 0:
        cfg = cfg.with_sampling_method(SamplingMethod.Greedy())
    elif t is not None or p is not None or k is not None:
        cfg = cfg.with_sampling_method(SamplingMethod.Stochastic(t, k, p, None, None, None))
    return cfg


def usage_of(stats):
    if stats is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    p, c = stats.tokens_count_input or 0, stats.tokens_count_output or 0
    u = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
    if stats.speculator_stats is not None:
        u["spec_verify_ct"] = stats.speculator_stats.num_decode_forward_passes
    return u


class Server:
    def __init__(self, session, name):
        self.session, self.name, self.lock = session, name, asyncio.Lock()

    async def models(self, request):
        return web.json_response({"object": "list", "data": [{"id": self.name, "object": "model", "owned_by": "mirai"}]})

    async def chat(self, request):
        try:
            body = await request.json()
            messages, config = build_messages(body), build_config(body)
        except (BadRequest, ValueError, json.JSONDecodeError) as e:
            return web.json_response({"error": {"message": str(e), "type": "invalid_request_error"}}, status=400)
        rid, created, stream = f"chatcmpl-{uuid.uuid4().hex}", int(time.time()), bool(body.get("stream"))

        def chunk(delta, finish=None, usage=None):
            ev = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": self.name,
                  "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage is not None:
                ev["usage"] = usage
            return f"data: {json.dumps(ev)}\n\n".encode()

        resp = None  # the stream response, set iff the request streams
        async with self.lock:
            await self.session.reset()
            s = await self.session.reply_with_stream(messages, config)
            if stream:
                resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
                await resp.prepare(request)
                await resp.write(chunk({"role": "assistant"}))
            sent_text = sent_reason = 0
            final, error = None, None
            try:
                async for ch in s:
                    if isinstance(ch, ChatSessionStreamChunk.Replies) and ch.replies:
                        final = ch.replies[-1]
                        if resp is not None:
                            reason, text = final.message.reasoning or "", final.message.text or ""
                            if len(reason) > sent_reason:
                                await resp.write(chunk({"reasoning_content": reason[sent_reason:]}))
                                sent_reason = len(reason)
                            if len(text) > sent_text:
                                await resp.write(chunk({"content": text[sent_text:]}))
                                sent_text = len(text)
                    elif isinstance(ch, ChatSessionStreamChunk.Error):
                        error = ch.error
            except (ConnectionResetError, asyncio.CancelledError):
                # the client went away: stop at the next token and let the session clean up before the next request
                s.cancel_token.cancel()
                async for _ in s:
                    pass
                raise
        if error is not None and final is None:
            if resp is not None:
                await resp.write(f"data: {json.dumps({'error': {'message': error}})}\n\n".encode())
                await resp.write(b"data: [DONE]\n\n")
                return resp
            return web.json_response({"error": {"message": error, "type": "server_error"}}, status=500)
        reason = final.finish_reason if final is not None else None
        finish = FINISH.get(str(reason).rsplit(".", 1)[-1], "stop") if reason is not None else "stop"
        usage = usage_of(final.stats if final is not None else None)
        if resp is not None:
            await resp.write(chunk({}, finish, usage))
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        msg = {"role": "assistant", "content": final.message.text if final else ""}
        if final is not None and final.message.reasoning:
            msg["reasoning_content"] = final.message.reasoning
        return web.json_response({"id": rid, "object": "chat.completion", "created": created, "model": self.name,
                                  "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local model directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8098)
    a = ap.parse_args()
    engine = await Engine.create(EngineConfig.create())
    model = await engine.model(os.path.abspath(a.model))
    if model is None:
        sys.exit(f"model not found: {a.model}")
    async for _ in (await engine.download(model)).iterator():
        pass
    session = await engine.chat(model, ChatConfig.create())
    srv = Server(session, os.path.basename(os.path.abspath(a.model)))
    app = web.Application(client_max_size=64 * 1024 * 1024)
    for prefix in ("", "/v1"):
        app.router.add_get(prefix + "/models", srv.models)
        app.router.add_post(prefix + "/chat/completions", srv.chat)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, a.host, a.port).start()
    print(f"Loaded: {getattr(model, 'identifier', a.model)}; listening on {a.host}:{a.port}", flush=True)  # noqa: NP100
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
