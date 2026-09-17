list(APPEND CMAKE_PREFIX_PATH /opt/xilinx/xrt)
find_package(AIEBU)
find_package(xrt REQUIRED)

ggml_add_backend_library(ggml-xdna
                         ggml-xdna.cpp
                         xdna-runtime.cpp
                         xdna-kernel-pool.cpp
                         xdna-seq.cpp
                         xdna-verify.cpp
                         xdna-quant.cpp
                         xdna-gemv.cpp
                         xdna-seq-attn.cpp
                         xdna-ops.cpp
                         xdna-fa-prefill.cpp
                         xdna-gdn-prefill.cpp
                         xdna-conv-prefill.cpp
                         xdna-rec.cpp
                         xdna-rec-gemv.cpp
                         xdna-rec-post.cpp
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
set(XDNA_TILE_K 64)         # per-core K tile (larger = fewer C reloads per call)
set(XDNA_TILE_N 64)         # per-core N tile (wider = longer B DDR bursts)
set(XDNA_N_COLS 8)          # AIE columns
set(XDNA_N_COMPUTE_ROWS 4)  # compute tile rows

target_compile_definitions(ggml-xdna PRIVATE
    GGML_XDNA_GEMM_M=${XDNA_GEMM_M}
    GGML_XDNA_TILE_M=${XDNA_TILE_M}
    GGML_XDNA_GEMM_M_BIG=${XDNA_GEMM_M_BIG}
    GGML_XDNA_TILE_M_BIG=${XDNA_TILE_M_BIG}
    GGML_XDNA_TILE_M_BIG_I8=${XDNA_TILE_M_BIG_I8}
    GGML_XDNA_TILE_K=${XDNA_TILE_K}
    GGML_XDNA_TILE_N=${XDNA_TILE_N}
    GGML_XDNA_N_COLS=${XDNA_N_COLS}
    GGML_XDNA_N_COMPUTE_ROWS=${XDNA_N_COMPUTE_ROWS}
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
            DEPENDS "${GEMM_DIR}/gemm.py"
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
            DEPENDS "${GEMM_DIR}/gemm.py"
            WORKING_DIRECTORY "${GEMM_DIR}"
            COMMENT "ggml-xdna: compiling int8 GEMM kernel M${M_VAL} K1024 N2048 c${XDNA_N_COLS}"
            VERBATIM
        )
        list(APPEND XDNA_ARTIFACTS ${INT8_XCLBIN} ${INT8_INSTS})
    endforeach()

    # GDN prefill kernel (gdn-prefill.py + gdn-prefill.cc): the bf16 CS=64-token
    # GGML_OP_GATED_DELTA_NET recurrence, run by xdna-gdn-prefill.cpp. The DMA
    # schedule is a host-built TXN stream at runtime; the .insts.bin here is
    # only a build-time reference and is never read by the runner. Stem follows
    # the gemm_<dtype>_<geom>_c<cols> convention.
    set(GDN_PREFILL_STEM "gdn_prefill_bf16_S128_H16_CS64_c8")
    set(GDN_PREFILL_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${GDN_PREFILL_STEM}.xclbin")
    set(GDN_PREFILL_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${GDN_PREFILL_STEM}.insts.bin")

    add_custom_command(
        OUTPUT ${GDN_PREFILL_XCLBIN} ${GDN_PREFILL_INSTS}
        COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/gdn-prefill.py"
                -d npu2
                --xclbin-path ${GDN_PREFILL_XCLBIN}
                --insts-path ${GDN_PREFILL_INSTS}
        DEPENDS "${GEMM_DIR}/gdn-prefill.py" "${GEMM_DIR}/gdn-prefill.cc"
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling GDN prefill kernel S128 H16 CS64"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${GDN_PREFILL_XCLBIN} ${GDN_PREFILL_INSTS})

    # SSM_CONV kernel (conv.py + conv.cc, ported from the qvac-fabric max2
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
        DEPENDS "${GEMM_DIR}/conv.py" "${GEMM_DIR}/conv.cc"
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling conv kernel C16 T256 KW4 MB8 (token-major, bf16)"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${CONV_XCLBIN} ${CONV_INSTS})

    # Flash attention kernel (fa.py + fa.cc) for the six full-attention prefill
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
        DEPENDS "${GEMM_DIR}/fa.py" "${GEMM_DIR}/fa.cc"
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling flash attention kernel D256 MT32 JT8 NJ64"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${FA_XCLBIN} ${FA_INSTS})

    # Fused decode A-half kernel (attn_gdn_gated.py + attn_cn.py + gdn_v.py +
    # rec_gated.py): the merged recurrent-core decode kernel with the gated
    # epilogue on-chip of the fused layer path (xdna-rec.cpp). The .insts.bin is
    # only a build-time reference; the runner binds a host-built phased TXN
    # stream (xdna-seq-attn.cpp). Stem: attn_gdn_gated.
    set(ATTN_GDN_STEM "attn_gdn_gated")
    set(ATTN_GDN_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${ATTN_GDN_STEM}.xclbin")
    set(ATTN_GDN_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${ATTN_GDN_STEM}.insts.bin")

    add_custom_command(
        OUTPUT ${ATTN_GDN_XCLBIN} ${ATTN_GDN_INSTS}
        COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/attn_gdn_gated.py"
                -d npu2
                --workdir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                --tag ""
        DEPENDS "${GEMM_DIR}/attn_gdn_gated.py" "${GEMM_DIR}/attn_cn.py"
                "${GEMM_DIR}/gdn_v.py" "${GEMM_DIR}/rec_gated.py"
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling fused decode kernel attn_gdn_gated"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${ATTN_GDN_XCLBIN} ${ATTN_GDN_INSTS})

    # Fused decode ssm_out kernel (attn_so_mmul.py): the int4 (Q5_K ssm_out)
    # matmul of the fused layer path (xdna-rec.cpp). The .insts.bin is only a
    # build-time reference; the runner binds a host-built TXN stream. Stem:
    # attn_so_mmul. Geometry is fixed by the Qwen3.5-0.8B constants in the
    # design (K_SO/N_SO/NCOL), which the C++ rec path also assumes.
    set(SO_STEM "attn_so_mmul")
    set(SO_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${SO_STEM}.xclbin")
    set(SO_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${SO_STEM}.insts.bin")

    add_custom_command(
        OUTPUT ${SO_XCLBIN} ${SO_INSTS}
        COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/attn_so_mmul.py"
                -d npu2
                --workdir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                --compile
        DEPENDS "${GEMM_DIR}/attn_so_mmul.py" "${GEMM_DIR}/ffn_layer_mmul.py"
                "${GEMM_DIR}/proj_qK.py" "${GEMM_DIR}/ffn_mid.py"
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling fused decode ssm_out kernel attn_so_mmul"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${SO_XCLBIN} ${SO_INSTS})

    # Fused decode FFN kernel (ffn_layer_mmul.py --stage ab): the whole FFN of
    # the fused layer path (xdna-rec.cpp), stage-A gate/up int4 + stage-B down
    # int4 with the silu in-core. The .insts.bin is only a build-time
    # reference. Stem: ffn_mmul_ab. Geometry is fixed by the Qwen3.5-0.8B
    # constants in the design (K/N_MID/K_DOWN/N_OUT, n_a=n_b=4).
    set(FFN_STEM "ffn_mmul_ab")
    set(FFN_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${FFN_STEM}.xclbin")
    set(FFN_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${FFN_STEM}.insts.bin")

    add_custom_command(
        OUTPUT ${FFN_XCLBIN} ${FFN_INSTS}
        COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/ffn_layer_mmul.py"
                -d npu2
                --workdir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                --compile
        DEPENDS "${GEMM_DIR}/ffn_layer_mmul.py" "${GEMM_DIR}/proj_qK.py"
                "${GEMM_DIR}/ffn_mid.py"
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling fused decode FFN kernel ffn_mmul_ab"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${FFN_XCLBIN} ${FFN_INSTS})

    # Decode GEMV (gemv_q4.py + gemv_q4.cc): a single artifact. Both code
    # widths pack into the same tile size, so one design streams either and
    # Q4_K keeps its 4-bit codes; a second artifact would be a second hardware
    # context, and the NPU reconfigures the array between them.
    # The core program depends only on (format, n_core, k_tile) - K, N and the
    # active column count are runtime parameters written per core - so every
    # decode shape shares one xclbin and therefore one hardware context. The
    # shape it is compiled at only fixes IRON's own instruction stream, which
    # the backend replaces with xdna_gemv_seq_build.
    set(XDNA_GEMV_COLS   8)     # AIE columns instantiated

    # Two ways of splitting the same array, both covering 1024 output columns
    # per pass: "n_core rows row-list". Four rows of 32 is the standalone
    # decode geometry; two rows of 64 leaves every column a row for the fused
    # layer's GDN core, and is also how much less of the array a design can
    # configure while doing the same work, which the backend can A/B with
    # GGML_XDNA_GEMV_ROWS=2.
    set(XDNA_GEMV_SPLITS
        "32 4 2,3,4,5"
        "64 2 3,5|3,5|2,5|2,5|2,3|2,3|2,3|2,3")

    foreach(GV_SPLIT IN LISTS XDNA_GEMV_SPLITS)
        string(REGEX MATCHALL "[^ ]+" GV_PARTS "${GV_SPLIT}")
        list(GET GV_PARTS 0 GV_N_CORE)
        list(GET GV_PARTS 1 GV_ROWS)
        list(GET GV_PARTS 2 GV_ROW_MAP)
        set(GV_STEM "gemv_n${GV_N_CORE}_r${GV_ROWS}_c${XDNA_GEMV_COLS}")
        set(GV_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${GV_STEM}.xclbin")
        set(GV_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${GV_STEM}.insts.bin")

        add_custom_command(
            OUTPUT ${GV_XCLBIN} ${GV_INSTS}
            COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/gemv_q4.py"
                    -d npu2 --fmt q8g16
                    -K 1024 -N 1024
                    --n-core ${GV_N_CORE}
                    --rows ${GV_ROW_MAP}
                    --cols ${XDNA_GEMV_COLS}
                    --xclbin-path ${GV_XCLBIN}
                    --insts-path ${GV_INSTS}
            DEPENDS "${GEMM_DIR}/gemv_q4.py" "${GEMM_DIR}/gemv_q4.cc"
                    "${GEMM_DIR}/gemv_zero.cc" "${GEMM_DIR}/wfmt.py"
            WORKING_DIRECTORY "${GEMM_DIR}"
            COMMENT "ggml-xdna: compiling decode GEMV ${GV_STEM}"
            VERBATIM
        )
        list(APPEND XDNA_ARTIFACTS ${GV_XCLBIN} ${GV_INSTS})
    endforeach()

    # The whole recurrent layer in one xclbin (fused_layer.py): the fused
    # conv+norm+gdn+gated core and the decode GEMV configured side by side, so
    # a layer's dispatches share one hardware context instead of reconfiguring
    # the array between them. The fused decode path uses it when it is present
    # and falls back to the separate artifacts when it is not.
    option(GGML_XDNA_BUILD_FUSED_LAYER "build the merged recurrent-layer xclbin" ON)
    if (GGML_XDNA_BUILD_FUSED_LAYER)
        set(FL_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/fused_layer.xclbin")
        set(FL_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/fused_layer.insts.bin")
        # ACT_RAW=1 puts the FFN transition on a tile of this design (the
        # residual add, the norm and the quantization), so the host has nothing
        # to do between a layer's two dispatches. Only this artifact gets it:
        # the standalone GEMV ones have no prologue tile and would be handed
        # numbers where they expect codes.
        add_custom_command(
            OUTPUT ${FL_XCLBIN} ${FL_INSTS}
            COMMAND ${CMAKE_COMMAND} -E env ACT_RAW=1
                    ${GEMM_PYTHON} "${GEMM_DIR}/fused_layer.py"
                    -d npu2
                    --xclbin-path ${FL_XCLBIN}
                    --insts-path ${FL_INSTS}
            DEPENDS "${GEMM_DIR}/fused_layer.py" "${GEMM_DIR}/attn_gdn_gated.py"
                    "${GEMM_DIR}/gemv_q4.py" "${GEMM_DIR}/gemv_q4.cc"
                    "${GEMM_DIR}/gemv_zero.cc" "${GEMM_DIR}/attn_cn.py"
                    "${GEMM_DIR}/gdn_v.py" "${GEMM_DIR}/rec_gated.py"
            WORKING_DIRECTORY "${GEMM_DIR}"
            COMMENT "ggml-xdna: compiling the merged recurrent layer"
            VERBATIM
        )
        list(APPEND XDNA_ARTIFACTS ${FL_XCLBIN} ${FL_INSTS})

        # The same design as a self-contained full ELF. The sequence becomes
        # the ELF's control code instead of an instruction stream the host
        # builds and patches, so the layer's buffers are plain kernel
        # arguments - the six a patched stream is limited to is what keeps a
        # layer at more than one dispatch.
        set(FL_ELF "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/fused_layer_full.elf")
        add_custom_command(
            OUTPUT ${FL_ELF}
            COMMAND ${CMAKE_COMMAND} -E env ACT_RAW=1
                    ${GEMM_PYTHON} "${GEMM_DIR}/fused_layer.py"
                    -d npu2
                    --elf-path ${FL_ELF}
            DEPENDS "${GEMM_DIR}/fused_layer.py" "${GEMM_DIR}/attn_gdn_gated.py"
                    "${GEMM_DIR}/gemv_q4.py" "${GEMM_DIR}/gemv_q4.cc"
                    "${GEMM_DIR}/gemv_zero.cc" "${GEMM_DIR}/attn_cn.py"
                    "${GEMM_DIR}/gdn_v.py" "${GEMM_DIR}/rec_gated.py"
            WORKING_DIRECTORY "${GEMM_DIR}"
            COMMENT "ggml-xdna: compiling the merged recurrent layer as a full ELF"
            VERBATIM
        )
        list(APPEND XDNA_ARTIFACTS ${FL_ELF})
    endif()

    # Tagged copies of every artifact the backend loads under a design-tagged
    # name (design_tag.py). One run covers them all, after any of them built:
    # the tag hashes all the design sources, so a change anywhere moves the tag
    # for the designs ninja did not rebuild too.
    set(XDNA_TAGGED_STEMS
        attn_gdn_gated attn_so_mmul ffn_mmul_ab fused_layer
        gemv_n32_r4_c8 gemv_n64_r2_c8)
    set(XDNA_TAGGED_ARTIFACTS)
    foreach(TS IN LISTS XDNA_TAGGED_STEMS)
        foreach(TE IN ITEMS .xclbin .insts.bin)
            list(APPEND XDNA_TAGGED_ARTIFACTS
                 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TS}_${XDNA_DESIGN_TAG}${TE}")
        endforeach()
    endforeach()
    add_custom_command(
        OUTPUT ${XDNA_TAGGED_ARTIFACTS}
        COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/design_tag.py"
                --stamp-dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                --extra npu2
        DEPENDS ${XDNA_ARTIFACTS} "${GEMM_DIR}/design_tag.py"
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: stamping kernel artifacts with the design tag"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${XDNA_TAGGED_ARTIFACTS})

    # The FFN transition as its own xclbin (post_norm.py): residual add, rms
    # norm and the FFN's activation tiles on one core, dispatched between the
    # fused core and the FFN pair.
    set(POST_STEM "post_norm")
    set(POST_XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${POST_STEM}.xclbin")
    set(POST_INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${POST_STEM}.insts.bin")
    add_custom_command(
        OUTPUT ${POST_XCLBIN} ${POST_INSTS}
        COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/post_norm.py"
                -d npu2
                --xclbin-path ${POST_XCLBIN}
                --insts-path ${POST_INSTS}
        DEPENDS "${GEMM_DIR}/post_norm.py" "${GEMM_DIR}/design_tag.py"
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling the FFN transition kernel post_norm"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${POST_XCLBIN} ${POST_INSTS})

    # The same design as a self-contained full ELF (PDIs + control code), the
    # platform's mode for a dispatch between other dispatches: no xclbin
    # context to cross.
    set(POST_ELF "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${POST_STEM}_full.elf")
    add_custom_command(
        OUTPUT ${POST_ELF}
        COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/post_norm.py"
                -d npu2
                --elf-path ${POST_ELF}
        DEPENDS "${GEMM_DIR}/post_norm.py"
        WORKING_DIRECTORY "${GEMM_DIR}"
        COMMENT "ggml-xdna: compiling the FFN transition as a full ELF"
        VERBATIM
    )
    list(APPEND XDNA_ARTIFACTS ${POST_ELF})

    add_custom_target(ggml-xdna-kernels ALL DEPENDS ${XDNA_ARTIFACTS})

    install(FILES ${XDNA_ARTIFACTS} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
