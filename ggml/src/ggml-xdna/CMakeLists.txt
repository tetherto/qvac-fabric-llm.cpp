list(APPEND CMAKE_PREFIX_PATH /opt/xilinx/xrt)
find_package(AIEBU)
find_package(xrt REQUIRED)

ggml_add_backend_library(ggml-xdna
                         ggml-xdna.cpp
                         xdna-runtime.cpp
                         xdna-seq.cpp
                         xdna-quant.cpp
                         xdna-gemv.cpp
                         xdna-seq-attn.cpp
                         xdna-ops.cpp
                         xdna-pgemm.cpp
                         xdna-attn-mm.cpp
                         xdna-gdn-mm.cpp
                         xdna-fa-prefill.cpp
                         xdna-gdn-prefill.cpp
                         xdna-conv-prefill.cpp
                         xdna-rec.cpp
                         xdna-rec-gemv.cpp
                         xdna-att.cpp
                         xdna-att-layer.cpp
                         xdna-head.cpp
                        )

target_link_libraries(ggml-xdna PRIVATE XRT::xrt_coreutil)

# The one place the backend wants threads: re-quantising a weight tensor to
# int8 the first time a GEMM asks for it. It is once per tensor and still
# 1638 ms of a 6.8 s prompt single-threaded.
if (GGML_OPENMP)
    find_package(OpenMP)
    if (OpenMP_FOUND)
        target_compile_definitions(ggml-xdna PRIVATE GGML_USE_OPENMP)
        target_link_libraries(ggml-xdna PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# Two GEMM geometries baked into the IRON designs (see kernels/gemm.py): the
# decode M-block (M32, native bf16 r=4, exact) and the larger prefill M-block
# (bfp16-emulated r=8, 2x mmul). The backend target receives the same values
# as compile definitions, so xdna_gemm_tiles (xdna-seq.h) and the kernel
# compiles stay in sync.
#
# The prefill tile_m is the reuse factor for the weights: one B sweep feeds
# tile_m * n_compute_rows rows, so the 64-row int8 tile re-reads B from DDR
# four times less often than the 16-row tile the first prefill kernels used.
# bf16 cannot go that far - its double-width A/B banks plus the f32 C tile
# exceed the 64 KB core L1 past 32 rows - so the two routes bake different
# tiles. The prefill M blocks must stay multiples of tile_m * n_compute_rows.
set(XDNA_GEMM_M 32)         # decode M block (tile_m * n_compute_rows)
set(XDNA_TILE_M 8)          # decode per-core M tile (r=4 native bf16)
set(XDNA_GEMM_M_BIG 256)    # base prefill M block
set(XDNA_TILE_M_BIG 32)     # bf16 prefill per-core M tile
set(XDNA_TILE_M_BIG_I8 64)  # int8 prefill per-core M tile
set(XDNA_GEMM_M_BIG_EXTRA 512 2048)   # extra prefill M blocks (baked variants)
# The option was called XDNA_GATED_FMT while the README and the runtime
# diagnostic told users to set GGML_XDNA_GATED_FMT. Carry a value set under the
# old name over rather than silently reconfiguring an existing build directory
# to the 8-bit layout.
if (DEFINED CACHE{XDNA_GATED_FMT} AND NOT DEFINED CACHE{GGML_XDNA_GATED_FMT})
    message(WARNING "ggml-xdna: XDNA_GATED_FMT is now GGML_XDNA_GATED_FMT; using ${XDNA_GATED_FMT}")
    set(GGML_XDNA_GATED_FMT "${XDNA_GATED_FMT}" CACHE STRING
        "ggml-xdna: gated activation layout baked into the fused layer (0 = 4-bit, for a Q4_K ssm_out; 1 = 8-bit, for Q5_K/Q6_K)")
endif()
set(GGML_XDNA_GATED_FMT 1 CACHE STRING
    "ggml-xdna: gated activation layout baked into the fused layer (0 = 4-bit, for a Q4_K ssm_out; 1 = 8-bit, for Q5_K/Q6_K)")
set(XDNA_TILE_K 64)         # per-core K tile (larger = fewer C reloads per call)
set(XDNA_TILE_N 64)         # per-core N tile (wider = longer B DDR bursts)
set(XDNA_N_COLS 8)          # AIE columns
set(XDNA_N_COMPUTE_ROWS 4)  # compute tile rows

target_compile_definitions(ggml-xdna PRIVATE
    GGML_XDNA_GEMM_M=${XDNA_GEMM_M}
    GGML_XDNA_TILE_M=${XDNA_TILE_M}
    GGML_XDNA_TILE_M_BIG=${XDNA_TILE_M_BIG}
    GGML_XDNA_TILE_M_BIG_I8=${XDNA_TILE_M_BIG_I8}
    GGML_XDNA_TILE_K=${XDNA_TILE_K}
    GGML_XDNA_TILE_N=${XDNA_TILE_N}
    GGML_XDNA_N_COLS=${XDNA_N_COLS}
    GGML_XDNA_N_COMPUTE_ROWS=${XDNA_N_COMPUTE_ROWS}
    GGML_XDNA_GATED_FMT=${GGML_XDNA_GATED_FMT}
)

# NPU kernels are compiled out-of-line by kernels/gemm.py (IRON/MLIR-AIE) at
# build time. Each variant is a (K N) tuple; M and the tiles are fixed by the
# geometry above. Artifacts land in the runtime output directory and are loaded
# from there (or the executable directory / cwd) at runtime. The .insts.bin is
# a reference copy of the compiled DMA sequence; the runtime builds streams.
option(GGML_XDNA_BUILD_KERNELS "ggml-xdna: build NPU GEMM kernels (requires IRON)" ON)
set(GGML_XDNA_GEMM_PYTHON "" CACHE FILEPATH
    "ggml-xdna: python interpreter with IRON (mlir_aie + llvm-aie) installed")

# GEMM variants to build: "K N". K must be a multiple of 8, N of 256.
set(GGML_XDNA_GEMM_VARIANTS
    "1024 2048"
)

if (GGML_XDNA_BUILD_KERNELS)
    if (GGML_XDNA_GEMM_PYTHON)
        set(GEMM_PYTHON ${GGML_XDNA_GEMM_PYTHON})
    else()
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        set(GEMM_PYTHON ${Python3_EXECUTABLE})
    endif()

    # The build only needs IRON when kernels are actually compiled; disable
    # silently if the interpreter lacks it (the backend still builds).
    execute_process(
        COMMAND ${GEMM_PYTHON} -c "import aie.iron, aie.helpers.taplib, aie.iron.kernels"
        RESULT_VARIABLE GEMM_IRON_RESULT
        OUTPUT_QUIET ERROR_QUIET
    )
    if (NOT GEMM_IRON_RESULT EQUAL 0)
        message(WARNING "ggml-xdna: IRON not found in ${GEMM_PYTHON}; skipping kernel build")
        set(GGML_XDNA_BUILD_KERNELS OFF)
    endif()
endif()

if (GGML_XDNA_BUILD_KERNELS)
    set(GEMM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
    set(XDNA_ARTIFACTS "")
    # The shared kernel headers are pasted into each source by kernelsrc.py, so
    # a design must rebuild when one of them changes.
    set(XDNA_KERNEL_HEADERS
        "${GEMM_DIR}/xdna-vec.h"
        "${GEMM_DIR}/xdna-math.h"
    )

    # Geometry: the decode M block (native bf16 r=4) plus every prefill M block
    # (bfp16 r=8 / int8 r=8; the two routes bake different tile_m). Stems must
    # match the runtime scan in xdna_ops_init (gemm_bf16_f32_M%d_K%d_N%d_c%d,
    # gemm_int8_int32_...).
    set(XDNA_GEMM_PREFILL_BLOCKS ${XDNA_GEMM_M_BIG} ${XDNA_GEMM_M_BIG_EXTRA})

    set(XDNA_GEOMETRIES "${XDNA_GEMM_M} ${XDNA_TILE_M} 0")
    foreach(M_VAL IN LISTS XDNA_GEMM_PREFILL_BLOCKS)
        list(APPEND XDNA_GEOMETRIES "${M_VAL} ${XDNA_TILE_M_BIG} 1")
    endforeach()

    foreach(g IN LISTS XDNA_GEOMETRIES)
        string(REPLACE " " ";" FIELDS "${g}")
        list(GET FIELDS 0 M_VAL)
        list(GET FIELDS 1 TM_VAL)
        list(GET FIELDS 2 BFP16_VAL)

        # Artifact stem matches the runtime cache key exactly.
        set(KERNEL_STEM "gemm_bf16_f32_M${M_VAL}_K1024_N2048_c${XDNA_N_COLS}")
        set(XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${KERNEL_STEM}.xclbin")
        set(INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${KERNEL_STEM}.insts.bin")

        if (BFP16_VAL EQUAL 0)
            set(BFP16_ARGS "--no-bfp16")
        else()
            set(BFP16_ARGS "")
        endif()

        add_custom_command(
            OUTPUT ${XCLBIN} ${INSTS}
            COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/gemm.py"
                    -M ${M_VAL} -K 1024 -N 2048
                    --tile-m ${TM_VAL}
                    --tile-k ${XDNA_TILE_K}
                    --tile-n ${XDNA_TILE_N}
                    --n-aie-cols ${XDNA_N_COLS}
                    --n-aie-rows ${XDNA_N_COMPUTE_ROWS}
                    -d npu2
                    ${BFP16_ARGS}
                    --xclbin-path ${XCLBIN}
                    --insts-path ${INSTS}
            DEPENDS "${GEMM_DIR}/gemm.py" ${XDNA_KERNEL_HEADERS}
            WORKING_DIRECTORY "${GEMM_DIR}"
            COMMENT "ggml-xdna: compiling GEMM kernel M${M_VAL} K1024 N2048 bfp16=${BFP16_VAL} c${XDNA_N_COLS}"
            VERBATIM
        )

        list(APPEND XDNA_ARTIFACTS ${XCLBIN} ${INSTS})
    endforeach()

    # Native int8 x int8 -> int32 GEMM for quantized (Q4_K/Q5_K/Q6_K) weights
    # (gemm.py --dtype int8_int32). The int8 mmul is r=8 on AIE2P, so tile_m is a
    # multiple of 16; its A/B banks are half the bf16 ones, which buys the wider
    # XDNA_TILE_M_BIG_I8 tile. Decode (M < 64) keeps the bf16 M32 kernel. The
    # stream builder is shared with the bf16 route; only the RTP base and the
    # loop counts move with the tile.
    foreach(M_VAL IN LISTS XDNA_GEMM_PREFILL_BLOCKS)
        set(INT8_STEM "gemm_int8_int32_M${M_VAL}_K1024_N2048_c${XDNA_N_COLS}")
        set(INT8_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${INT8_STEM}.xclbin")
        set(INT8_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${INT8_STEM}.insts.bin")

        add_custom_command(
            OUTPUT ${INT8_XCLBIN} ${INT8_INSTS}
            COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/gemm.py"
                    -M ${M_VAL} -K 1024 -N 2048
                    --tile-m ${XDNA_TILE_M_BIG_I8}
                    --tile-k ${XDNA_TILE_K}
                    --tile-n ${XDNA_TILE_N}
                    --n-aie-cols ${XDNA_N_COLS}
                    --n-aie-rows ${XDNA_N_COMPUTE_ROWS}
                    --dtype int8_int32
                    -d npu2
                    --xclbin-path ${INT8_XCLBIN}
                    --insts-path ${INT8_INSTS}
            DEPENDS "${GEMM_DIR}/gemm.py" ${XDNA_KERNEL_HEADERS}
            WORKING_DIRECTORY "${GEMM_DIR}"
            COMMENT "ggml-xdna: compiling int8 GEMM kernel M${M_VAL} K1024 N2048 c${XDNA_N_COLS}"
            VERBATIM
        )
        list(APPEND XDNA_ARTIFACTS ${INT8_XCLBIN} ${INT8_INSTS})
    endforeach()

    # GDN prefill kernel (gdn_prefill.py + gdn-prefill.cc): the bf16 CS=64-token
    # GGML_OP_GATED_DELTA_NET recurrence, run by xdna-gdn-prefill.cpp. The DMA
    # schedule is a host-built TXN stream at runtime; the .insts.bin here is
    # only a build-time reference and is never read by the runner. Stem follows
    # the gemm_<dtype>_<geom>_c<cols> convention.
    set(GDN_PREFILL_STEM "gdn_prefill_bf16_S128_H16_CS64_c8")
    set(GDN_PREFILL_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${GDN_PREFILL_STEM}.xclbin")
    set(GDN_PREFILL_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${GDN_PREFILL_STEM}.insts.bin")

    add_custom_command(
        OUTPUT ${GDN_PREFILL_XCLBIN} ${GDN_PREFILL_INSTS}
        COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/gdn_prefill.py"
                -d npu2
                --xclbin-path ${GDN_PREFILL_XCLBIN}
                --insts-path ${GDN_PREFILL_INSTS}
        DEPENDS "${GEMM_DIR}/gdn_prefill.py" "${GEMM_DIR}/gdn-prefill.cc" ${XDNA_KERNEL_HEADERS}
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling GDN prefill kernel S128 H16 CS64"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${GDN_PREFILL_XCLBIN} ${GDN_PREFILL_INSTS})

    # SSM_CONV kernel (conv.py + conv-ssm.cc, ported from the qvac-fabric max2
    # branch): causal depthwise conv1d, token-major layout (no host transpose),
    # C=16 channels x 8 cols x 4 rows = 512 channels per tile, T=256 token
    # tiles, MB=8 tiles streamed per run. Host loops the 6144/512 channel
    # groups and the token tiles. Stem: conv_prefill_bf16_c16_t256_kw4_mb8.
    # bf16 halves the four BO streams of the op; the accumulators stay f32.
    set(CONV_STEM "conv_prefill_bf16_c16_t256_kw4_mb8")
    set(CONV_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONV_STEM}.xclbin")
    set(CONV_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONV_STEM}.insts.bin")

    add_custom_command(
        OUTPUT ${CONV_XCLBIN} ${CONV_INSTS}
        COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/conv.py"
                -d npu2
                --token-major
                --bf16
                --xclbin-path ${CONV_XCLBIN}
                --insts-path ${CONV_INSTS}
        DEPENDS "${GEMM_DIR}/conv.py" "${GEMM_DIR}/conv-ssm.cc" ${XDNA_KERNEL_HEADERS}
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling conv kernel C16 T256 KW4 MB8 (token-major, bf16)"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${CONV_XCLBIN} ${CONV_INSTS})

    # Flash attention kernel (fa.py + attn-fa.cc) for the six full-attention prefill
    # layers: 8 columns = 8 query heads, 4 cores a column each owning 32
    # queries, 64 key tiles of 8 keys a dispatch. The host chains chunks of the
    # key range and rounds of the batch, so this one artifact covers every n_kv.
    set(FA_STEM "fa_prefill_bf16_D256_MT32_JT8_NJ64_c8")
    set(FA_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${FA_STEM}.xclbin")
    set(FA_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${FA_STEM}.insts.bin")

    add_custom_command(
        OUTPUT ${FA_XCLBIN} ${FA_INSTS}
        COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/fa.py"
                -d npu2
                --cols 8 --KVH 2 --MT 32 --JT 8 --NMB 1 --NJ 64
                --xclbin-path ${FA_XCLBIN}
                --insts-path ${FA_INSTS}
        DEPENDS "${GEMM_DIR}/fa.py" "${GEMM_DIR}/attn-fa.cc" ${XDNA_KERNEL_HEADERS}
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling flash attention kernel D256 MT32 JT8 NJ64"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${FA_XCLBIN} ${FA_INSTS})

    # Prefill GEMM (pgemm.py + gemm-expand.cc):
    # every core expands the decode GEMV's own weight tiles into bf16 and runs
    # the bfp16 mmul. The cores read their loop counts from the first A object,
    # so one artifact covers every M, K, N and both tile formats; the runner
    # (xdna-pgemm.cpp) builds each call's TXN stream, and the .insts.bin here
    # is only the sample shape's.
    set(PG_STEM "pgemm_c8")
    set(PG_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PG_STEM}.xclbin")
    set(PG_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PG_STEM}.insts.bin")
    add_custom_command(
        OUTPUT ${PG_XCLBIN} ${PG_INSTS}
        COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/pgemm.py"
                -d npu2
                --xclbin-path ${PG_XCLBIN}
                --insts-path ${PG_INSTS}
        DEPENDS "${GEMM_DIR}/pgemm.py" "${GEMM_DIR}/gemm-expand.cc" ${XDNA_KERNEL_HEADERS}
                "${GEMM_DIR}/gemv_q4.py" "${GEMM_DIR}/wfmt.py"
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling the prefill GEMM"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${PG_XCLBIN} ${PG_INSTS})

    # Prefill attention (attn_mm.py + attn-mm.cc):
    # the six full-attention layers on the mmul, the cores driven by a header
    # (passes, the ubatch's first position), so one artifact covers every
    # length; the runner (xdna-attn-mm.cpp) builds each call's TXN stream.
    set(AM_STEM "attn_mm_c8")
    set(AM_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${AM_STEM}.xclbin")
    set(AM_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${AM_STEM}.insts.bin")
    add_custom_command(
        OUTPUT ${AM_XCLBIN} ${AM_INSTS}
        COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/attn_mm.py"
                -d npu2
                --xclbin-path ${AM_XCLBIN}
                --insts-path ${AM_INSTS}
        DEPENDS "${GEMM_DIR}/attn_mm.py" "${GEMM_DIR}/attn-mm.cc" ${XDNA_KERNEL_HEADERS}
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling the prefill attention"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${AM_XCLBIN} ${AM_INSTS})

    # Prefill gated delta rule (gdn_mm.py + gdn-mm.cc): the recurrent layers'
    # state on the array for the whole ubatch, the cores driven by a header, one
    # artifact for every length; the runner
    # (xdna-gdn-mm.cpp) builds each call's TXN stream.
    set(GM_STEM "gdn_mm_c8")
    set(GM_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${GM_STEM}.xclbin")
    set(GM_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${GM_STEM}.insts.bin")
    add_custom_command(
        OUTPUT ${GM_XCLBIN} ${GM_INSTS}
        COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/gdn_mm.py"
                -d npu2
                --xclbin-path ${GM_XCLBIN}
                --insts-path ${GM_INSTS}
        DEPENDS "${GEMM_DIR}/gdn_mm.py" "${GEMM_DIR}/gdn-mm.cc" ${XDNA_KERNEL_HEADERS}
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling the prefill gated delta rule"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${GM_XCLBIN} ${GM_INSTS})

    # The recurrent layers' conv input on the array (gdn_conv.py +
    # gdn-conv.cc): the in-projection's rows convolved, SiLU'd and K, Q
    # normalized straight into the prefill GDN's input; the runner
    # (xdna-gdn-mm.cpp) builds each call's TXN stream.
    set(GC_STEM "gdn_conv_c8")
    set(GC_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${GC_STEM}.xclbin")
    set(GC_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${GC_STEM}.insts.bin")
    add_custom_command(
        OUTPUT ${GC_XCLBIN} ${GC_INSTS}
        COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/gdn_conv.py"
                -d npu2
                --xclbin-path ${GC_XCLBIN}
                --insts-path ${GC_INSTS}
        DEPENDS "${GEMM_DIR}/gdn_conv.py" "${GEMM_DIR}/gdn-conv.cc" ${XDNA_KERNEL_HEADERS}
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling the GDN conv input"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${GC_XCLBIN} ${GC_INSTS})

    # Fused decode (fused_layer.py): the whole recurrent layer in one xclbin -
    # the conv+norm+gdn+gated core, the decode GEMV its projections run on, and
    # the FFN transition tile, side by side so a layer's dispatches share one
    # hardware context. The .insts.bin is only a build-time reference; the
    # runner binds a host-built phased TXN stream.
    #
    # The design itself puts the FFN transition on a tile of this array (the
    # residual add, the norm and the quantization), so the host has nothing to
    # do between a layer's dispatches; it sets ACT_RAW in fused_layer.py rather
    # than reading it here, which keeps that part of its identity in its
    # source.
    set(FL_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/fused_layer.xclbin")
    set(FL_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/fused_layer.insts.bin")
    add_custom_command(
        OUTPUT ${FL_XCLBIN} ${FL_INSTS}
        COMMAND ${CMAKE_COMMAND} -E env GATED_FMT=${GGML_XDNA_GATED_FMT}
                ${GEMM_PYTHON} "${GEMM_DIR}/fused_layer.py"
                -d npu2
                --xclbin-path ${FL_XCLBIN}
                --insts-path ${FL_INSTS}
        DEPENDS "${GEMM_DIR}/fused_layer.py" ${XDNA_KERNEL_HEADERS}
                "${GEMM_DIR}/gemv_q4.py" "${GEMM_DIR}/gemv-q4.cc"
                "${GEMM_DIR}/gemv-zero.cc" "${GEMM_DIR}/gemv-merge.cc"
                "${GEMM_DIR}/attn-conv.cc" "${GEMM_DIR}/attn-norm.cc"
                "${GEMM_DIR}/gdn-v.cc" "${GEMM_DIR}/rec-gated.cc"
                "${GEMM_DIR}/gated-h2.cc" "${GEMM_DIR}/attn-dec.cc"
                "${GEMM_DIR}/act-att.cc"
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling the merged recurrent layer"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${FL_XCLBIN} ${FL_INSTS})


    # Decode GEMV (gemv_q4.py + gemv-q4.cc): one artifact per array split. Both
    # code widths pack into the same tile size, so one design streams either and
    # Q4_K keeps its 4-bit codes; a second artifact would be a second hardware
    # context, and the NPU reconfigures the array between them.
    # The core program depends only on (format, n_core, k_tile) - K, N and the
    # active column count are runtime parameters written per core - so every
    # decode shape shares one xclbin and therefore one hardware context. The
    # shape it is compiled at only fixes IRON's own instruction stream, which
    # the backend replaces with xdna_gemv_seq_build.
    set(XDNA_GEMV_COLS 8)     # AIE columns instantiated

    # The per-op decode split: four rows of 32 output columns per core covers
    # 1024 columns per pass. The fused layer's projections use the split that
    # shares the array with the GDN core instead, which is built inside
    # fused_layer.xclbin and loaded under that artifact's name.
    set(GV_N_CORE 32)
    set(GV_ROWS 4)
    set(GV_ROW_MAP "2,3,4,5")
    set(GV_STEM "gemv_n${GV_N_CORE}_r${GV_ROWS}_c${XDNA_GEMV_COLS}")
    set(GV_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${GV_STEM}.xclbin")
    set(GV_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${GV_STEM}.insts.bin")

    add_custom_command(
        OUTPUT ${GV_XCLBIN} ${GV_INSTS}
        COMMAND ${CMAKE_COMMAND} -E env GATED_FMT=${GGML_XDNA_GATED_FMT}
                ${GEMM_PYTHON} "${GEMM_DIR}/gemv_q4.py"
                -d npu2 --fmt q8g16
                -K 1024 -N 1024
                --n-core ${GV_N_CORE}
                --rows ${GV_ROW_MAP}
                --cols ${XDNA_GEMV_COLS}
                --xclbin-path ${GV_XCLBIN}
                --insts-path ${GV_INSTS}
        DEPENDS "${GEMM_DIR}/gemv_q4.py" "${GEMM_DIR}/gemv-q4.cc" ${XDNA_KERNEL_HEADERS}
                "${GEMM_DIR}/gemv-zero.cc" "${GEMM_DIR}/wfmt.py"
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling decode GEMV ${GV_STEM}"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${GV_XCLBIN} ${GV_INSTS})

    # Tagged copies of every artifact the backend loads under a design-tagged
    # name (design_tag.py). One run covers them all, after any of them built:
    # the tag hashes all the design sources, so a change anywhere moves the tag
    # for the designs ninja did not rebuild too.
    #
    # The tag is computed here because the tagged names are declared as the
    # rule's outputs and are installed with the rest: without it they come out
    # as "fused_layer_.xclbin", files the rule never creates, and ninja reruns
    # the whole kernel target on every build. Every source the tag hashes is a
    # configure dependency, so editing one moves the declared names with it.
    file(GLOB XDNA_KERNEL_SOURCES CONFIGURE_DEPENDS
         "${GEMM_DIR}/*.cc" "${GEMM_DIR}/*.h" "${GEMM_DIR}/*.py")
    list(REMOVE_ITEM XDNA_KERNEL_SOURCES "${GEMM_DIR}/design_tag.py")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${XDNA_KERNEL_SOURCES})
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env GATED_FMT=${GGML_XDNA_GATED_FMT}
                ${GEMM_PYTHON} "${GEMM_DIR}/design_tag.py" --extra npu2
        OUTPUT_VARIABLE XDNA_DESIGN_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE XDNA_TAG_RESULT
    )
    if (NOT XDNA_TAG_RESULT EQUAL 0 OR XDNA_DESIGN_TAG STREQUAL "")
        message(FATAL_ERROR "ggml-xdna: design_tag.py did not produce a tag (${XDNA_TAG_RESULT})")
    endif()
    # The backend includes xdna-design-tag.h, which otherwise only the stamping
    # step writes - after ninja has already decided the backend is up to date,
    # so a build after a kernel edit, or a clean one next to the committed
    # header, compiles the previous tag in. Write the tag computed here, as
    # design_tag.py does, before anything compiles; only when it changed, so an
    # unchanged design does not recompile the backend.
    set(XDNA_TAG_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/xdna-design-tag.h")
    set(XDNA_TAG_HEADER_TEXT "#pragma once\n// Generated by kernels/design_tag.py.\n#define XDNA_DESIGN_TAG \"${XDNA_DESIGN_TAG}\"\n")
    set(XDNA_TAG_HEADER_NOW "")
    if (EXISTS "${XDNA_TAG_HEADER}")
        file(READ "${XDNA_TAG_HEADER}" XDNA_TAG_HEADER_NOW)
    endif()
    if (NOT "${XDNA_TAG_HEADER_NOW}" STREQUAL "${XDNA_TAG_HEADER_TEXT}")
        file(WRITE "${XDNA_TAG_HEADER}" "${XDNA_TAG_HEADER_TEXT}")
    endif()
    set(XDNA_TAGGED_STEMS fused_layer gemv_n32_r4_c8)
    set(XDNA_TAGGED_ARTIFACTS)
    foreach(TS IN LISTS XDNA_TAGGED_STEMS)
        foreach(TE IN ITEMS .xclbin .insts.bin)
            list(APPEND XDNA_TAGGED_ARTIFACTS
                 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TS}_${XDNA_DESIGN_TAG}${TE}")
        endforeach()
    endforeach()
    add_custom_command(
        OUTPUT ${XDNA_TAGGED_ARTIFACTS}
        COMMAND ${CMAKE_COMMAND} -E env GATED_FMT=${GGML_XDNA_GATED_FMT}
                ${GEMM_PYTHON} "${GEMM_DIR}/design_tag.py"
                --stamp-dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                --extra npu2
        DEPENDS ${XDNA_ARTIFACTS} "${GEMM_DIR}/design_tag.py" ${XDNA_KERNEL_HEADERS}
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: stamping kernel artifacts with the design tag"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${XDNA_TAGGED_ARTIFACTS})

    add_custom_target(ggml-xdna-kernels ALL DEPENDS ${XDNA_ARTIFACTS})

    # design_tag.py also writes xdna-design-tag.h, which the backend includes.
    # Without this the library compiles before the header is written and the
    # binary looks for artifacts under the previous tag, so a clean build has to
    # be run twice before it will load anything.
    add_dependencies(ggml-xdna ggml-xdna-kernels)

    install(FILES ${XDNA_ARTIFACTS} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()


