#!/usr/bin/env bash
#
# Print the first port of a free 8-port block for the server integration tests.
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
# A block, not a single port: some tests need a second server alongside the
# first (see test_compat_anthropic.py) and offset from PORT to get it. Blocks
# never overlap, so those offsets stay inside the job's own reservation.
#
# Usage: PORT=$(.github/scripts/pick-server-port.sh)   # tests may use PORT+1..+7

set -euo pipefail

readonly RANGE_START=20000
readonly BLOCK_SIZE=8
# 20000-32767. The ceiling is ip_local_port_range's default floor of 32768: a
# block above it can already hold an outbound connection's source port, which a
# connect probe cannot see (it only detects listeners), so the block would be
# reported free and llama-server would fail to bind later.
readonly BLOCK_COUNT=1596
readonly MAX_PROBES=200

# The fallback keeps this usable outside CI.
seed="${RUNNER_NAME:-${HOSTNAME:-local}}"
first_block=$(( $(printf '%s' "$seed" | cksum | cut -d' ' -f1) % BLOCK_COUNT ))

for (( probe = 0; probe < MAX_PROBES; probe++ )); do
    block=$(( (first_block + probe) % BLOCK_COUNT ))
    base=$(( RANGE_START + block * BLOCK_SIZE ))

    free=1
    for (( offset = 0; offset < BLOCK_SIZE; offset++ )); do
        # A successful connect means something is already listening.
        if (exec 3<>"/dev/tcp/127.0.0.1/$(( base + offset ))") 2>/dev/null; then
            exec 3>&- 2>/dev/null || true
            free=0
            break
        fi
    done

    if (( free )); then
        echo "${base}"
        exit 0
    fi
done

echo "pick-server-port: no free ${BLOCK_SIZE}-port block in ${MAX_PROBES} probes (seed: ${seed})" >&2
exit 1
