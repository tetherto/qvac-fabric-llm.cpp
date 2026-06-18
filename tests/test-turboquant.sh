#!/usr/bin/env bash
#
# Run quantization tests for TurboQuant/PolarQuant types only.
#
# Usage:
#   tests/test-turboquant.sh [--short|--full] [build_dir]
#
# Modes (only affects the subgroup-coverage legs; the fns / perf / backend-ops
# legs always run their full matrix because they're quick):
#   --short   (default) The SG-coverage legs (native Vulkan + the 3 lavapipe
#             stitch legs) only test tbq3_0 + pq3_0 + their _64 siblings, not
#             every type. Cuts runtime of the slow legs by ~50% while still
#             exercising both QUANT_K = 128 (tbq3_0 / pq3_0) and QUANT_K = 64
#             (tbq3_0_64 / pq3_0_64) code paths, which is the dimension the
#             stitch logic actually varies over. Picked for CI / quick local
#             iteration.
#   --full    All 8 TBQ/PQ types (tbq3/tbq4/pq3/pq4 + each _64) on every leg.
#             Use before landing shader changes to the cooperative path.
#
# Examples:
#   tests/test-turboquant.sh                # short mode (default), build dir "build"
#   tests/test-turboquant.sh --full         # full mode, build dir "build"
#   tests/test-turboquant.sh --short build-debug
#   tests/test-turboquant.sh --full  build-release
#
set -euo pipefail

mode="short"
B=""
while [ $# -gt 0 ]; do
    case "$1" in
        --short) mode="short"; shift;;
        --full|--long) mode="full"; shift;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | sed 's/^#\s\{0,1\}//;/^set -euo/d'
            exit 0;;
        -*) echo "unknown flag: $1" >&2; exit 2;;
        *)
            if [ -n "$B" ]; then
                echo "unexpected positional arg: $1" >&2
                exit 2
            fi
            B="$1"; shift;;
    esac
done
B="${B:-build}"

# Which types the SG-coverage legs iterate. Restricted in --short mode for
# speed; see header comment for rationale.
if [ "$mode" = "full" ]; then
    SG_TYPES_ARGS=()
else
    SG_TYPES_ARGS=(--types tbq3_0,pq3_0,tbq3_0_64,pq3_0_64)
fi

TQ_TYPES=(tbq3_0 tbq4_0 pq3_0 pq4_0 tbq3_0_64 tbq4_0_64 pq3_0_64 pq4_0_64)

echo "=========================================="
echo " TurboQuant/PolarQuant Test Suite"
echo " Build dir: $B"
echo " Mode:      $mode$([ "$mode" = "short" ] && echo ' (SG-coverage legs restricted to tbq3_0/pq3_0/_64)')"
echo "=========================================="
echo ""

num_failed=0

# Per-leg summaries for the coverage table rendered at the end. Each entry is
# a TSV row "leg<TAB>sg<TAB>nsg<TAB>result". The fields are kept as raw
# strings (not integers) so we can mix e.g. "32, 64" into the sg column for
# the native leg without second-guessing what the device exposed. Populated
# by run_sg_leg() below; rendered by render_coverage_table() at the end.
coverage_rows=()

# Run a single leg of test-copy-tbq-subgroups, tee its output so the user
# still sees per-case tables live, capture the final "PASSED/FAILED: ran=..."
# line, and record a summary row for the final coverage table.
#
# Args:
#   $1  human-readable leg label (e.g. "native", "lavapipe W=128")
#   $2  subgroup-size column text as it should appear in the table (e.g. "32, 64", "4")
#   $3  NSG column text        (e.g. "1 (fast path)", "8 (stitch)")
#   $4+ command to execute (already env-wrapped as needed)
run_sg_leg() {
    local leg="$1" sg="$2" nsg="$3"
    shift 3
    local log
    log=$(mktemp)
    # Run the leg. The test binary exits 0 on pass, 1 on fail -- we deliberately
    # swallow the exit with || so we keep going and can render the table.
    "$@" 2>&1 | tee "$log" || true
    local status_line
    status_line=$(grep -E '^(PASSED|FAILED|SKIPPED):' "$log" | tail -1 || true)
    local result="MISSING"
    if [ -n "$status_line" ]; then
        result="$status_line"
        if echo "$status_line" | grep -q '^FAILED:'; then
            num_failed=$((num_failed + 1))
        fi
    else
        num_failed=$((num_failed + 1))
    fi
    coverage_rows+=("${leg}	${sg}	${nsg}	${result}")
    rm -f "$log"
}

# Run a single test-backend-ops QJL leg and record it in the same final
# subgroup table. test-backend-ops reports "N/N tests passed" rather than the
# "PASSED:" line emitted by test-copy-tbq-subgroups.
run_qjl_sg_leg() {
    local leg="$1" sg="$2" nsg="$3"
    shift 3
    local log
    log=$(mktemp)
    set +e
    "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    local result_line
    result_line=$(grep -E '[0-9]+/[0-9]+ tests passed' "$log" | tail -1 || true)
    local result="MISSING"
    if [ "$rc" -eq 0 ] && [ -n "$result_line" ]; then
        result="OK: $result_line"
    else
        result="FAILED"
        num_failed=$((num_failed + 1))
    fi
    coverage_rows+=("${leg}	${sg}	${nsg}	${result}")
    rm -f "$log"
}

# Run a FLASH_ATTN_EXT leg of test-backend-ops and fail if the -p filter
# selected no cases at all.
#
# test-backend-ops 'test' mode exits 0 when every generated case reports
# "not supported", printing "0/0 tests passed" and a green OK. For the
# fp16-disabled PQ leg below that is a legitimate outcome -- supports_op
# rejecting the op so the scheduler falls back to CPU is one of the two
# accepted fixes named in its comment -- so we deliberately do NOT require a
# non-zero pass count here (that would fail the leg for behaving correctly).
#
# What we do require is that the leg still *generated* cases. A -p filter that
# silently stops matching (a renamed PQ type, a reworked FA case generator)
# also yields "0/0 tests passed" and a green OK, and is indistinguishable from
# the intended outcome by exit code alone. Counting the emitted case lines
# separates "the gate rejected N cases" from "there was nothing to reject".
run_fa_leg() {
    local leg="$1"
    shift
    local log
    log=$(mktemp)
    # set +e around the pipeline for the same reason run_qjl_sg_leg does it:
    # this script runs under `set -euo pipefail`, so a failing leg would abort
    # the run outright instead of being counted and reported in the summary.
    set +e
    "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    local cases
    cases=$(grep -c 'FLASH_ATTN_EXT(' "$log" || true)
    if [ "$rc" -ne 0 ]; then
        num_failed=$((num_failed + 1))
    elif [ "${cases:-0}" -eq 0 ]; then
        echo "FAILED: $leg selected 0 cases -- the -p filter matched nothing, so this leg tested nothing"
        num_failed=$((num_failed + 1))
    else
        echo "($leg: $cases case(s) selected)"
    fi
    rm -f "$log"
}

# Render a Unicode box-drawing table summarising the subgroup-coverage legs.
# Columns: Leg | Subgroup size | NSG | Result. Widths are computed dynamically
# from the longest cell in each column so the table stays aligned no matter
# which legs actually ran (e.g. lavapipe ICD missing -> only native row).
render_coverage_table() {
    if [ "${#coverage_rows[@]}" -eq 0 ]; then
        return
    fi
    local headers=("Leg" "Subgroup size" "NSG" "Result")
    local -a w=(0 0 0 0)
    local i
    for i in 0 1 2 3; do
        w[$i]=${#headers[$i]}
    done
    # Column widths: scan each TSV row and grow the max. We scope the IFS
    # change to a subshell so it doesn't leak into later code that relies on
    # default whitespace splitting (notably `seq 1 N` expansion below, which
    # silently breaks if IFS is set to tab).
    local row
    local widths_line
    widths_line=$(
        local IFS=$'\t'
        local -a cur=("${w[@]}")
        for row in "${coverage_rows[@]}"; do
            local -a cells
            # shellcheck disable=SC2206
            cells=($row)
            for i in 0 1 2 3; do
                local len=${#cells[$i]}
                if [ "$len" -gt "${cur[$i]}" ]; then
                    cur[$i]=$len
                fi
            done
        done
        printf '%s %s %s %s' "${cur[0]}" "${cur[1]}" "${cur[2]}" "${cur[3]}"
    )
    read -r w[0] w[1] w[2] w[3] <<<"$widths_line"
    # Box-drawing helpers. We pad each cell with one space on either side, so
    # the segment width between joints is w[i]+2. Build the dash run by simple
    # concatenation rather than `printf '─%.0s' $(seq 1 N)`, which would rely
    # on default IFS word-splitting and is fragile.
    local line_top="┌" line_mid="├" line_bot="└"
    for i in 0 1 2 3; do
        local n=$((w[i] + 2))
        local dash=""
        local k=0
        while [ $k -lt $n ]; do
            dash+="─"
            k=$((k + 1))
        done
        line_top+="${dash}"
        line_mid+="${dash}"
        line_bot+="${dash}"
        if [ $i -lt 3 ]; then
            line_top+="┬"; line_mid+="┼"; line_bot+="┴"
        else
            line_top+="┐"; line_mid+="┤"; line_bot+="┘"
        fi
    done
    # Render a single data row. Takes a TSV-encoded string, splits on TAB in a
    # subshell (again to keep IFS from leaking), pads each cell, and emits the
    # "│ a │ b │ c │ d │" line.
    print_row() {
        local line="$1"
        (
            local IFS=$'\t'
            local -a cells
            # shellcheck disable=SC2206
            cells=($line)
            local row_str="│"
            local j
            for j in 0 1 2 3; do
                local cell="${cells[$j]:-}"
                row_str+=$(printf ' %-*s ' "${w[$j]}" "$cell")
                row_str+="│"
            done
            echo "$row_str"
        )
    }
    echo "$line_top"
    print_row "$(printf '%s\t%s\t%s\t%s' "${headers[0]}" "${headers[1]}" "${headers[2]}" "${headers[3]}")"
    echo "$line_mid"
    for row in "${coverage_rows[@]}"; do
        print_row "$row"
    done
    echo "$line_bot"
}

echo "=== test-quantize-fns (all types, includes TBQ/PQ correctness assertions) ==="
if [ -f "$B/bin/test-quantize-fns" ]; then
    "$B/bin/test-quantize-fns" || num_failed=$((num_failed + 1))
else
    echo "SKIP: $B/bin/test-quantize-fns not found"
fi
echo ""

echo "=== test-quantize-perf CPU (TBQ/PQ types, with perf sanity checks) ==="
if [ -f "$B/bin/test-quantize-perf" ]; then
    "$B/bin/test-quantize-perf" --type tbq3_0 --type tbq4_0 --type pq3_0 --type pq4_0 --type q4_0 || true
else
    echo "SKIP: $B/bin/test-quantize-perf not found"
fi
echo ""

echo "=== test-quantize-perf Vulkan (TBQ/PQ types, with perf sanity checks) ==="
if [ -f "$B/bin/test-quantize-perf" ]; then
    "$B/bin/test-quantize-perf" -b vulkan --type tbq3_0 --type tbq4_0 --type pq3_0 --type pq4_0 --type q4_0 || true
else
    echo "SKIP: $B/bin/test-quantize-perf not found"
fi
echo ""

echo "=== test-backend-ops (TBQ/PQ ops only) ==="
if [ -f "$B/bin/test-backend-ops" ]; then
    "$B/bin/test-backend-ops" test -p "tbq|pq" || num_failed=$((num_failed + 1))
else
    echo "SKIP: $B/bin/test-backend-ops not found"
fi
echo ""

# Cross-subgroup-size correctness for the Vulkan copy_to_quant shader. This
# probes each SG in {0, 4, 8, 16, 32, 64} once to ask the backend which
# overrides it honors, then self-spawns workload children only for the SGs
# that were accepted. SGs the device does not expose are labelled
# SKIP-<reason> with empty metric columns -- no workload child is spawned
# for them (they would otherwise just repeat the sg=0 run and produce
# duplicate numbers). Requires a Vulkan-capable backend; harmlessly skips
# otherwise.
echo "=== test-copy-tbq-subgroups (Vulkan copy_to_quant, SG sweep 0/4/8/16/32/64) ==="
if [ -f "$B/bin/test-copy-tbq-subgroups" ]; then
    # "native" leg: whatever real Vulkan GPU ggml-vulkan picks. Subgroup size
    # and stitch factor depend on the device: desktop GPUs end up at NSG=1
    # (fast path), while a hypothetical small-SG device would exercise stitch
    # here too. We label them generically so the table is device-independent.
    run_sg_leg "native GPU" "device default (>=32 on typical GPUs)" "1 (fast path on typical GPUs)" \
        "$B/bin/test-copy-tbq-subgroups" "${SG_TYPES_ARGS[@]}"
else
    echo "SKIP: $B/bin/test-copy-tbq-subgroups not found"
fi
echo ""

# Stitch-path coverage via lavapipe. The generalized copy_to_quant shader
# dispatches on NSG = TQ_WG (=32) / gl_SubgroupSize:
#   * NSG == 1 -> fast path, subgroupAdd() is a full workgroup reduction.
#   * NSG >  1 -> shared-memory stitch path reduces across NSG subgroups.
#
# On the vast majority of real GPUs the native subgroup size is >= 32, which
# always hits the fast path:
#   * NVIDIA: subgroupSize is fixed at 32 on every generation. Pass.
#   * AMD RDNA (desktop / APU, e.g. any gfx10xx/11xx): native 32 or 64,
#     minSubgroupSize >= 32. Pass.
#   * AMD GCN / CDNA (older / datacenter): native 64, no smaller sizes. Pass.
#   * Apple / most mobile ARM Mali, Qualcomm Adreno in desktop-ish modes:
#     also >= 32. Pass.
# VK_EXT_subgroup_size_control on these devices cannot request SG < 32 because
# minSubgroupSize is >= 32, so the first leg above necessarily exercises NSG=1
# only. That's fine for production (the real hardware never needs the stitch)
# but means NSG > 1 is unreachable from real GPUs in CI.
#
# lavapipe (Mesa llvmpipe's Vulkan driver) closes this gap. It is a CPU-side
# software implementation whose native subgroup size equals
# LP_NATIVE_VECTOR_WIDTH / 32 (one lane per 32-bit slot in the JIT'd SIMD
# vector, SG range is clamped to that single value). Sweeping
# LP_NATIVE_VECTOR_WIDTH over its three supported values yields every
# non-trivial NSG the shader supports:
#
#   LP_NATIVE_VECTOR_WIDTH | lavapipe SG | NSG (=TQ_WG/SG) | stitch factor
#   -----------------------+-------------+-----------------+--------------
#        128                |      4      |       8         | 8-way shared-mem reduction
#        256                |      8      |       4         | 4-way shared-mem reduction
#        512                |     16      |       2         | 2-way shared-mem reduction
#
# Combined with the native-GPU leg above (NSG=1), this gives full coverage of
# the helper's {1, 2, 4, 8} NSG branches on any host, regardless of the real
# GPU installed. GGML_VK_ALLOW_CPU_DEVICES=1 opts into picking CPU-type
# Vulkan ICDs (ggml-vulkan normally rejects them to avoid accidentally
# running production workloads on a software renderer); VK_ICD_FILENAMES
# forces the loader to enumerate only lavapipe so the test cannot pick the
# real GPU. We only run this leg when the ICD file is present -- on
# Debian/Ubuntu it's bundled as `mesa-vulkan-drivers`; on Fedora/Arch the
# path may differ and the leg silently skips.
LVP_ICD="/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
if [ -f "$B/bin/test-copy-tbq-subgroups" ] && [ -f "$LVP_ICD" ]; then
    for W in 128 256 512; do
        SG=$((W / 32))
        NSG=$((32 / SG))
        echo "=== test-copy-tbq-subgroups (lavapipe LP_NATIVE_VECTOR_WIDTH=$W, SG=$SG, NSG=$NSG stitch) ==="
        run_sg_leg "lavapipe W=${W}" "${SG}" "${NSG} (stitch)" \
            env GGML_VK_ALLOW_CPU_DEVICES=1 \
                VK_ICD_FILENAMES="$LVP_ICD" \
                LP_NATIVE_VECTOR_WIDTH="$W" \
                "$B/bin/test-copy-tbq-subgroups" "${SG_TYPES_ARGS[@]}"
        echo ""
    done
fi

# Standalone non-FA TBQ QJL correction coverage. Unlike copy_to_quant.comp,
# mul_mm_tbq_qjl_correction.comp uses one workgroup per output element with
# local_size_x == QUANT_K (64 or 128), so even SG=32 still spans multiple
# subgroups. `test-backend-ops test` compares Vulkan against CPU reference;
# restricting to TBQ + f32 + n=16 keeps this focused on the mat-mat path that
# runs mul_mm followed by the QJL correction pass.
echo "=== test-backend-ops (lavapipe TBQ MUL_MAT QJL SG sweep) ==="
if [ -f "$B/bin/test-backend-ops" ] && [ -f "$LVP_ICD" ]; then
    for W in 128 256 512; do
        SG=$((W / 32))
        NSG64=$((64 / SG))
        NSG128=$((128 / SG))
        echo "=== test-backend-ops TBQ QJL (lavapipe LP_NATIVE_VECTOR_WIDTH=$W, SG=$SG, NSG64=$NSG64, NSG128=$NSG128 stitch) ==="
        run_qjl_sg_leg "QJL lavapipe W=${W}" "${SG}" "64:${NSG64}, 128:${NSG128} (stitch)" \
            env GGML_VK_ALLOW_CPU_DEVICES=1 \
            VK_ICD_FILENAMES="$LVP_ICD" \
            LP_NATIVE_VECTOR_WIDTH="$W" \
            GGML_VK_TBQ_COPY_SG_SIZE="$SG" \
            "$B/bin/test-backend-ops" test \
                -o MUL_MAT \
                -p 'type_a=tbq.*type_b=f32.*n=16'
        echo ""
    done
else
    echo "SKIP: test-backend-ops or lavapipe ICD not found"
fi
echo ""

# Guard against regressions in the shared MUL_MAT dispatcher. Our widening of
# `supports_op` (pipeline_cpy_quant_f16) relaxed the non-dim01-contiguous rule
# for every quantized src0 type, not just TBQ/PQ. So the same dispatcher bug
# that breaks TBQ/PQ also breaks upstream q8_0 with a permuted src0. Picking
# q8_0 here means the check runs on any Vulkan box (no TBQ model needed) and
# keeps the regression catchable under the "tbq|pq" narrow filter above.
echo "=== test-backend-ops (q8_0 MUL_MAT with permuted src0, shared dispatcher sanity) ==="
if [ -f "$B/bin/test-backend-ops" ]; then
    "$B/bin/test-backend-ops" test \
        -p 'type_a=q8_0.*per=\[0,2,1,3\]' \
        || num_failed=$((num_failed + 1))
else
    echo "SKIP: $B/bin/test-backend-ops not found"
fi
echo ""

# Guard against regressions in the shared FLASH_ATTN_EXT shaders. The TBQ
# patches edit flash_attn_base.glsl / flash_attn.comp / flash_attn_cm1.comp,
# which are also the upstream f16 KV path. The "tbq|pq" filter on the
# test-backend-ops leg above does NOT match type_K=f16,type_V=f16 cases, and
# the in-tree mixed-type FA loop skips same-type pairs, so a regression on
# the shared f16 path is invisible to test-turboquant.sh.
#
# This leg picks a tiny f16/f16 cluster (~8 cases, ~1s) that exercises the
# cooperative-matrix v1 path on devices that pick it (nb >= 2 routes through
# FA_COOPMAT1 on KHR_cooperative_matrix devices; the scalar path is hit for
# nb=1 via the n_rows==1 fallback in get_fa_tuning_params). Specifically:
#
#   * kv=113 + nb=32 -> small KV, padded batch, exercises KV_bounds_check
#   * kv=512 + nb=3  -> exact-Bc-aligned KV, partial batch
#   * kv=512 + nb=32 -> aligned KV and batch, hot path
#   * prec=f32 / prec=def -> f32acc and f16acc variants
#
# These collectively trigger the bug where flash_attn_cm1.comp guards its
# f16 P*V section by `#if V_BLOCK_SIZE == 1`, which evaluates as false when
# V_BLOCK_SIZE is undefined (i.e. the f16 same-type variant emitted without
# any DATA_V_* macro), dropping the entire P*V accumulation from the SPIR-V
# and pushing PPL to ~1600 on the f16/f16 row of the perplexity sweep.
echo "=== test-backend-ops (f16 FLASH_ATTN_EXT, shared FA shader sanity) ==="
if [ -f "$B/bin/test-backend-ops" ]; then
    run_fa_leg "f16 FLASH_ATTN_EXT" \
        "$B/bin/test-backend-ops" test -o FLASH_ATTN_EXT \
        -p 'hsk=128,hsv=128,nh=4,nr23=\[1,1\],kv=(113|512),nb=(3|32),mask=1,sinks=0,max_bias=0\..*,logit_softcap=0\..*,prec=(f32|def),type_K=f16,type_V=f16,permute=\[0,1,2,3\]'
else
    echo "SKIP: $B/bin/test-backend-ops not found"
fi
echo ""

# Homogeneous PQ K/V FLASH_ATTN_EXT with fp16 disabled. supports_op currently
# advertises PQ FA regardless of device->fp16, but the scalar FA pipelines for
# tk == tv (PQ3/PQ3, PQ4/PQ4) are registered only inside the device->fp16
# branch. On a real fp16-less GPU (or with GGML_VK_DISABLE_F16=1 forcing the
# else-branch on any device), dispatch hits an uninitialized lazy pipeline and
# asserts `Br == pipeline->wg_denoms[0]`. Either the gate must reject the op so
# the scheduler falls back to CPU, or fp32 PQ FA pipelines must be added.
echo "=== test-backend-ops (homogeneous PQ FLASH_ATTN_EXT, GGML_VK_DISABLE_F16=1) ==="
if [ -f "$B/bin/test-backend-ops" ]; then
    run_fa_leg "homogeneous PQ FLASH_ATTN_EXT (fp16 disabled)" \
        env GGML_VK_DISABLE_F16=1 "$B/bin/test-backend-ops" test -o FLASH_ATTN_EXT \
        -p 'type_K=(pq3_0|pq4_0|pq3_0_64|pq4_0_64),type_V=\1'
else
    echo "SKIP: $B/bin/test-backend-ops not found"
fi
echo ""

if [ "${#coverage_rows[@]}" -gt 0 ]; then
    echo ""
    echo "=== Subgroup coverage summary ==="
    render_coverage_table
    echo ""
fi

echo "=========================================="
if [ "$num_failed" -eq 0 ]; then
    echo " All checks passed."
else
    echo " $num_failed check(s) failed."
fi
echo "=========================================="

exit "$num_failed"
