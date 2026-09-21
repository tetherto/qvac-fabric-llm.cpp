#!/usr/bin/env bash
#
# Print a TCP port for the server integration tests.
#
# QVAC-24501: tools/server/tests/utils.py binds 127.0.0.1:8080 unless PORT is
# set. That was fine on upstream's dedicated llama-server pool where one job
# owned the box; the qvac fleet runs several runner accounts per host, so two
# server jobs can collide. Either llama-server fails to bind, or it binds and
# one job's tests drive the other job's server and pass against a binary built
# with the wrong flags - the second outcome is the dangerous one.
#
# A runner account runs one job at a time, so RUNNER_NAME identifies a
# concurrent job on a host; deriving from it separates co-located jobs by
# construction. The scan covers what the derivation cannot: two names hashing
# alike, and ports held by something else.
#
# Usage: PORT=$(.github/scripts/pick-server-port.sh)

set -euo pipefail

readonly RANGE_START=20000
readonly RANGE_SIZE=20000  # 20000-39999, clear of the ephemeral range
readonly MAX_PROBES=200

# The fallback keeps this usable outside CI.
seed="${RUNNER_NAME:-${HOSTNAME:-local}}"
base=$(( RANGE_START + $(printf '%s' "$seed" | cksum | cut -d' ' -f1) % RANGE_SIZE ))

for (( probe = 0; probe < MAX_PROBES; probe++ )); do
    port=$(( RANGE_START + (base - RANGE_START + probe) % RANGE_SIZE ))

    # A successful connect means something is already listening.
    if ! (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
        echo "${port}"
        exit 0
    fi
    exec 3>&- 2>/dev/null || true
done

echo "pick-server-port: no free port in ${MAX_PROBES} probes from ${base} (seed: ${seed})" >&2
exit 1
