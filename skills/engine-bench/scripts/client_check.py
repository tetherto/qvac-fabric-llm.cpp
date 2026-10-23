#!/usr/bin/env python3
"""Check one bench_client.py call and append a row to <run dir>/errors.jsonl for each problem it had.

usage: client_check.py <errors.jsonl> <client log> <offset> <exit code> <label> <step> [key=value ...] [-- <client args>]

run_engine.sh calls it after every client call; the client log from byte <offset> on is that call's output.
- The client exited non-zero: one row with the step it stopped at and the exception that stopped it. bench_client.py
  saves its rows only at the end, so a conc or spec job that fails part way loses the levels it had measured
  ("not_saved").
- Requests failed inside a concurrency level whose row still came out (a worker thread died): one row for that level,
  whose numbers count only the requests that finished.
- A spec request came back with no text (one token or none): one row for that prompt and rep. Some servers reject a
  request inside an HTTP 200 stream (SGLang: "Input length ... exceeds the maximum allowed length"), so the client
  sees an empty answer, not an error.
<step> names the step ("depth 57344", "warm-up"); "-" derives it from the client arguments after "--" and the rows
the call printed (conc: the first level without a row; spec: the prompt and rep). With <client log> "-" (no client,
e.g. the warm-up curl) a non-zero exit code writes one row with the reason given as error=... key=value pairs
(engine=up, mode=ar) go into every row; empty values are left out. srvlog=<server log> is not a column: the last
error-like line of that log is added to each reason, since the client alone often sees only a dropped connection.
Row: {"label", "step", "error" (the reason), "at" (local time, ISO 8601), ...}.
"""
import json
import re
import sys
import time

PROMPTS = ["code", "rewrite", "story", "explain"]  # bench_client.py's spec prompts, in the order it runs them
SERVER_ERR = re.compile(r"exceeds|out of memory|OutOfMemory|CUDA error|GGML_ASSERT|\bERROR\b|Error: |[Aa]bort|Killed|"
                        r"[Ss]egmentation fault")


def server_hint(path):
    """The last error-like line of the server log, or ""."""
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()[-500:]
    except (OSError, TypeError):
        return ""
    hits = [line.strip() for line in lines if SERVER_ERR.search(line)]
    return f"; server log: {hits[-1][:200]}" if hits else ""


def opt(args, name, default):
    for i, a in enumerate(args[:-1]):
        if a == name:
            return args[i + 1]
    return default


def main():
    argv = sys.argv[1:]
    client = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    out, log, offset, rc, label, step = argv[:6]
    rc = int(rc)
    fields = dict(kv.split("=", 1) for kv in argv[6:])
    srvlog = fields.pop("srvlog", None)

    rows, exc_at, pending, in_tb, last = [], [], [], False, ""
    if log != "-":
        try:
            with open(log, "rb") as f:
                f.seek(int(offset.strip() or 0))
                text = f.read().decode("utf-8", "replace")
        except OSError:
            text = ""
        for line in text.splitlines():
            if line.strip():
                last = line.strip()
            if line.startswith("{"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("label") == label:
                    exc_at.append(pending)
                    rows.append(r)
                    pending = []
                continue
            if line.startswith("Traceback (most recent call last):"):
                in_tb = True
            elif in_tb and line and not line[0].isspace():
                # the first unindented line after the frames is the exception and its message
                pending.append(line.strip()[:300])
                in_tb = False

    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    new = []
    for r, excs in zip(rows, exc_at):
        if r.get("mode") == "spec" and (r.get("n_tok") or 0) <= 1:
            new.append({"label": label, "step": f"spec {r.get('prompt')} rep {r.get('rep')}",
                        "error": f"the request came back with {r.get('n_tok') or 0} token(s) and no text"})
        if excs and r.get("mode") == "conc":
            s = step if step != "-" else f"conc C={r['conc']}"
            new.append({"label": label, "step": s,
                        "error": f"{len(excs)} of {r['n_req']} requests failed, the row is measured on the rest: {excs[0]}"})
    if rc != 0:
        mode = opt(client, "--mode", "")
        not_saved = ""
        if step == "-" and mode == "conc":
            levels = [int(c) for c in opt(client, "--concurrency", "1,2,4,8,16,32").split(",")]
            done = [r["conc"] for r in rows if r.get("mode") == "conc"]
            nxt = levels[len(done)] if len(done) < len(levels) else None
            pt = int(opt(client, "--prompt-tokens", "512"))
            step = "conc" + (f" C={nxt}" if nxt else "") + (f" ({pt}-token prompts)" if pt != 128 else "")
            not_saved = ",".join(f"C={c}" for c in done)
        elif step == "-" and mode == "spec":
            reps = int(opt(client, "--reps", "2"))
            k = len([r for r in rows if r.get("mode") == "spec"])
            step = f"spec {PROMPTS[k // reps]} rep {k % reps}" if k < len(PROMPTS) * reps else "spec"
            not_saved = f"{k} requests" if k else ""
        elif step == "-":
            step = "client"
        # a worker thread's exception is the cause; the main thread's StatisticsError (no request finished) only follows
        cause = [e for e in pending if not e.startswith("statistics.StatisticsError")] or pending
        why = fields.pop("error", None) or (cause[0] if cause else None) or (last[:300] if last else None) \
            or f"exit code {rc}" + (" (killed)" if rc in (137, 143) else "")
        new.append({"label": label, "step": step, "error": why, "not_saved": not_saved})
    fields.pop("error", None)
    hint = server_hint(srvlog) if new and srvlog else ""
    with open(out, "a") as f:
        for row in new:
            row = {**row, "error": row["error"] + hint, "at": now, **fields}
            f.write(json.dumps({k: v for k, v in row.items() if v not in ("", None)}) + "\n")
    for row in new:
        print(f"error row: {row['label']} {row['step']}: {row['error'][:160]}")  # noqa: NP100


if __name__ == "__main__":
    main()
