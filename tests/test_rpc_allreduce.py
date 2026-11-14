#!/usr/bin/env python3
"""Run RPC collective tests against isolated local servers (CPU by default)."""

import argparse
import contextlib
import logging
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time


def reserve_port(port=0):
    sock = socket.socket()
    try:
        if os.name == "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind(("127.0.0.1", port))
    except OSError:
        sock.close()
        raise
    return sock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--client", required=True)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--device", default="CPU", help="device exposed by each local server, e.g. CPU or CUDA0")
    parser.add_argument("--wire", choices=("f32", "bf16"), default="f32")
    parser.add_argument("--port-base", action="store_true")
    parser.add_argument("--init-failure", action="store_true")
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    if args.ranks < 2 or args.ranks & (args.ranks - 1):
        parser.error("--ranks must be a power of two >= 2")
    if args.init_failure and args.ranks < 4:
        parser.error("--init-failure needs at least four ranks")

    env = os.environ.copy()
    for key in ("GGML_RPC_COMM_PORT", "GGML_RPC_NO_COMM", "GGML_RPC_NO_WIRE_BF16",
                "GGML_RPC_TEST_INIT_FAILURE", "GGML_RPC_TEST_UNREACHABLE_ENDPOINTS"):
        env.pop(key, None)
    if args.wire == "f32":
        env["GGML_RPC_NO_WIRE_BF16"] = "1"
    if args.disabled:
        env["GGML_RPC_NO_COMM"] = "1"
    if args.init_failure:
        env["GGML_RPC_TEST_INIT_FAILURE"] = "1"

    with contextlib.ExitStack() as stack:
        ports = []
        listeners = []
        comm_listeners = []
        for _ in range(args.ranks):
            for attempt in range(100):
                rpc = reserve_port()
                port = rpc.getsockname()[1]
                try:
                    if port > 64535:
                        raise OSError("RPC port leaves no room for comm port")
                    comm = reserve_port(port + 1000)
                except OSError:
                    rpc.close()
                    continue
                listeners.append(stack.enter_context(rpc))
                comm_listeners.append(stack.enter_context(comm))
                ports.append(port)
                break
            else:
                raise RuntimeError("could not reserve RPC ports")

        if args.port_base or args.init_failure:
            for attempt in range(100):
                reserved = []
                try:
                    reserved.append(reserve_port())
                    base = reserved[0].getsockname()[1]
                    if base + args.ranks > 65535:
                        raise OSError("port range overflow")
                    for rank in range(1, args.ranks):
                        reserved.append(reserve_port(base + rank))
                except OSError:
                    for sock in reserved:
                        sock.close()
                    continue
                for sock in comm_listeners:
                    sock.close()
                comm_listeners = [stack.enter_context(sock) for sock in reserved]
                env["GGML_RPC_COMM_PORT"] = str(base)
                break
            else:
                raise RuntimeError("could not reserve communication port range")

        for rank, sock in enumerate(comm_listeners):
            if not (args.init_failure and rank == 1):
                sock.close()

        with tempfile.TemporaryDirectory(prefix="rpc-allreduce-") as directory, contextlib.ExitStack() as logs:
            servers = []
            try:
                for rank, port in enumerate(ports):
                    log = logs.enter_context(open(Path(directory) / f"rank-{rank}.log", "w+"))
                    listeners[rank].close()
                    process = subprocess.Popen(
                        [args.server, "--device", args.device, "--host", "127.0.0.1", "--port", str(port),
                         "--threads", "1"], stdout=log, stderr=subprocess.STDOUT, env=env)
                    servers.append((process, log))
                deadline = time.monotonic() + 30
                for port, (process, _) in zip(ports, servers):
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError("RPC server exited during startup")
                        try:
                            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                                break
                        except OSError:
                            if time.monotonic() >= deadline:
                                raise RuntimeError("RPC server startup timed out")
                            time.sleep(0.05)
                endpoints = [f"127.0.0.1:{port}" for port in ports]
                command = [args.client, *endpoints]
                if args.legacy:
                    env["GGML_RPC_TEST_ENDPOINTS"] = ",".join(endpoints)
                    command = [args.client]
                subprocess.run(command, env=env, check=True, timeout=120)
            except BaseException:
                for rank, (_, log) in enumerate(servers):
                    log.flush()
                    log.seek(0)
                    logging.error("--- RPC rank %d ---\n%s", rank, log.read())
                raise
            finally:
                for process, _ in servers:
                    process.terminate()
                for process, _ in servers:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


if __name__ == "__main__":
    main()
