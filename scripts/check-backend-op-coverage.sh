#!/usr/bin/env bash
# check-backend-op-coverage.sh — guard against silent backend op-coverage shrink.
#
# Why this exists: when a backend's supports_op() stops accepting a case,
# test-backend-ops does not fail — the case falls back to CPU and is reported
# as "not supported [<backend>]", i.e. skipped. Commit e09ae0b71 removed
# Q4_1/Q4_K from the OpenCL MUL_MAT support set and produced zero test
# failures; nothing in CI noticed the de-offload. This script turns the
# support set itself into a tested artifact:
#
#   1. Run `test-backend-ops support -b <backend> -o <op>` for a curated op
#      list.
#   2. Reduce the output to a support set keyed by (op, relevant type/variant
#      params).
#   3. Diff against a checked-in manifest: coverage SHRINK is a hard FAIL;
#      growth is an advisory to regenerate the manifest.
#
# Manifests are per device class (ci/op-coverage/<backend>-<class>.txt)
# because intentional per-GPU gates (e.g. Adreno A7x FA rejection, MoE
# exclusions) differ between devices and must not false-alarm.
#
# Usage:
#   scripts/check-backend-op-coverage.sh check    <manifest> [options]
#   scripts/check-backend-op-coverage.sh generate <manifest> [options]
#
# Options:
#   -t <path>     test-backend-ops binary
#                 (default: build-opencl/bin/test-backend-ops under repo root)
#   -b <backend>  backend name passed to test-backend-ops -b
#                 (default: GPUOpenCL — this tree's OpenCL backend name)
#
# Notes for pocl (CPU-only boxes): the stock backend rejects pocl by device
# name. Generation/checking under pocl needs a build carrying the test-only
# enablement patch (see opencl-pocl-validation-b10069.md) and
# GGML_OPENCL_ALLOW_UNSUPPORTED_DEVICE=1 in the environment. The patch does
# not touch supports_op, so the support set it reports is the stock one.
#
# Exit codes: 0 ok (advisories possible), 1 coverage shrank or the SVE
# tripwire fired, 2 usage/environment error.

set -u

# Ops audited and the param keys that define a support-set entry for each.
# Keys are matched against the `key=value` tokens of the printed case params;
# every selected key must be present. Keep keys type-shaped (no ne/shape
# fields) so the manifest stays stable across sweep-shape churn.
OPS_KEYS='
MUL_MAT:type_a
MUL_MAT_ID:type_a
FLASH_ATTN_EXT:type_K,type_V
UPSCALE:type,mode
SET_ROWS:type_src,type_dst,type_idx
GET_ROWS:type
CPY:type_src,type_dst
'

usage() {
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

MODE="${1:-}"; MANIFEST="${2:-}"
case "$MODE" in check|generate) ;; *) usage ;; esac
[ -n "$MANIFEST" ] || usage
shift 2

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO_ROOT/build-opencl/bin/test-backend-ops"
BACKEND="GPUOpenCL"

while getopts "t:b:" opt; do
    case "$opt" in
        t) BIN="$OPTARG" ;;
        b) BACKEND="$OPTARG" ;;
        *) usage ;;
    esac
done

# ---------------------------------------------------------------------------
# T5 tripwire: linux-arm64 CPU variant list must stay SVE/SVE2-free.
#
# SVE/SVE2 variant kernels are numerically broken on Graviton3/Neoverse-V1
# (vla/pi05 integration cos ~0.77 vs ~0.999 reference); they were stripped
# from the linux-arm64 GGML_CPU_ALL_VARIANTS list (commits 29578a222 and
# 678fc8986 re-ports). This is a *static* tripwire against the list quietly
# regrowing SVE in a future rebase. Honest limitation: the real functional
# guard would be running the model battery on Graviton/Neoverse hardware,
# which this repo's CI does not have — a source grep is the best available
# proxy. NOSVE is allowed (it is the opt-out flag, not the feature).
# ---------------------------------------------------------------------------
check_sve_tripwire() {
    local cmakelists="$REPO_ROOT/ggml/src/CMakeLists.txt"
    [ -f "$cmakelists" ] || { echo "ERROR: $cmakelists not found" >&2; return 1; }
    local offenders
    offenders=$(awk '
        /GGML_SYSTEM_ARCH STREQUAL "ARM"/                { arm = 1 }
        arm && /CMAKE_SYSTEM_NAME MATCHES "Linux"/       { lin = 1; next }
        arm && lin && /elseif|else\(\)|endif/            { exit }
        arm && lin && /ggml_add_cpu_backend_variant\(/ &&
            /[ (]SVE2?[ )]/                              { print }
    ' "$cmakelists")
    if [ -n "$offenders" ]; then
        echo "FAIL: SVE/SVE2 variants reappeared in the linux-arm64 GGML_CPU_ALL_VARIANTS list:" >&2
        echo "$offenders" >&2
        echo "These variant kernels are numerically wrong on Graviton3/Neoverse-V1; see 29578a222." >&2
        return 1
    fi
    echo "ok: linux-arm64 variant list is SVE/SVE2-free"
    return 0
}

# ---------------------------------------------------------------------------
# Support-set collection
# ---------------------------------------------------------------------------
collect_support_set() {
    [ -x "$BIN" ] || { echo "ERROR: test-backend-ops binary not executable: $BIN" >&2; return 2; }
    local entry op keys out rc raw
    raw=$(mktemp)
    for entry in $OPS_KEYS; do
        op=${entry%%:*}; keys=${entry#*:}
        # One process per op: a crash while probing one op must not wipe out
        # the whole sweep's output.
        out=$("$BIN" support -b "$BACKEND" -o "$op" 2>/dev/null)
        rc=$?
        if [ $rc -ne 0 ] && [ -z "$out" ]; then
            echo "ERROR: '$BIN support -b $BACKEND -o $op' produced no output (rc=$rc)" >&2
            rm -f "$raw"
            return 2
        fi
        printf '%s\n' "$out" \
        | sed -e 's/\x1b\[[0-9;]*m//g' \
        | awk -v op="$op" -v keys="$keys" '
            BEGIN { nk = split(keys, K, ",") }
            $0 ~ "^  " op "\\(.*\\): SUPPORTED$" {
                params = $0
                sub("^  " op "\\(", "", params)
                sub("\\): SUPPORTED$", "", params)
                # tokenize on commas outside brackets ([..] holds ne arrays)
                depth = 0; tok = ""; ntok = 0
                for (i = 1; i <= length(params); i++) {
                    c = substr(params, i, 1)
                    if (c == "[") depth++
                    else if (c == "]") depth--
                    if (c == "," && depth == 0) { T[++ntok] = tok; tok = "" }
                    else tok = tok c
                }
                if (tok != "") T[++ntok] = tok
                delete V
                for (i = 1; i <= ntok; i++) {
                    eq = index(T[i], "=")
                    if (eq > 0) V[substr(T[i], 1, eq - 1)] = substr(T[i], eq + 1)
                }
                line = op; ok = 1
                for (k = 1; k <= nk; k++) {
                    if (!(K[k] in V)) { ok = 0; break }
                    line = line " " K[k] "=" V[K[k]]
                }
                if (ok) print line
            }' >> "$raw"
    done
    sort -u "$raw"
    rm -f "$raw"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
sve_rc=0
check_sve_tripwire || sve_rc=1

CURRENT="$(mktemp)"
trap 'rm -f "$CURRENT"' EXIT
collect_support_set > "$CURRENT" || exit 2

if [ ! -s "$CURRENT" ]; then
    echo "ERROR: collected support set is empty — backend '$BACKEND' missing or filters wrong" >&2
    exit 2
fi

if [ "$MODE" = generate ]; then
    mkdir -p "$(dirname "$MANIFEST")"
    {
        echo "# Backend op-coverage manifest — generated by scripts/check-backend-op-coverage.sh"
        echo "# backend: $BACKEND"
        echo "# Each line is a (op, type/variant) combination the backend claims support for."
        echo "# Regenerate with: scripts/check-backend-op-coverage.sh generate <this file> [-t bin] [-b backend]"
        cat "$CURRENT"
    } > "$MANIFEST"
    echo "wrote $(grep -vc '^#' "$MANIFEST") entries to $MANIFEST"
    exit $sve_rc
fi

[ -f "$MANIFEST" ] || { echo "ERROR: manifest not found: $MANIFEST" >&2; exit 2; }

SHRUNK=$(grep -v '^#' "$MANIFEST" | sort -u | comm -23 - "$CURRENT")
GROWN=$(grep -v '^#' "$MANIFEST" | sort -u | comm -13 - "$CURRENT")

rc=$sve_rc
if [ -n "$SHRUNK" ]; then
    echo "FAIL: backend '$BACKEND' lost op coverage present in $MANIFEST:" >&2
    printf '%s\n' "$SHRUNK" | sed 's/^/  - /' >&2
    echo "A supports_op regression silently de-offloads these to CPU (no test fails)." >&2
    echo "If the removal is intentional, regenerate the manifest and say so in the commit." >&2
    rc=1
fi
if [ -n "$GROWN" ]; then
    echo "advisory: backend '$BACKEND' gained coverage not in $MANIFEST (regenerate to record it):"
    printf '%s\n' "$GROWN" | sed 's/^/  + /'
fi
if [ -z "$SHRUNK" ] && [ -z "$GROWN" ]; then
    echo "ok: op coverage matches $MANIFEST ($(grep -vc '^#' "$MANIFEST") entries)"
fi
exit $rc
