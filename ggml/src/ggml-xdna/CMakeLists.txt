list(APPEND CMAKE_PREFIX_PATH /opt/xilinx/xrt)
find_package(AIEBU)
find_package(xrt REQUIRED)

ggml_add_backend_library(ggml-xdna
                         ggml-xdna.cpp
                         xdna-runtime.cpp
                         xdna-kernel-pool.cpp
                         xdna-seq.cpp
                         xdna-ops.cpp
                        )

target_link_libraries(ggml-xdna PRIVATE XRT::xrt_coreutil)

# Two GEMM geometries baked into the IRON designs (see kernels/gemm.py): the
# decode M-block (M32, native bf16 r=4, exact) and a larger prefill M-block
# (M64, bfp16-emulated r=8, 2x mmul). The backend target receives the same
# values as compile definitions, so xdna_gemm_tiles (xdna-seq.h) and the
# kernel compiles stay in sync.
set(XDNA_GEMM_M 32)         # decode M block (tile_m * n_compute_rows)
set(XDNA_TILE_M 8)          # decode per-core M tile (r=4 native bf16)
set(XDNA_GEMM_M_BIG 64)     # prefill M block
set(XDNA_TILE_M_BIG 16)     # prefill per-core M tile (2*r for the bfp16 r=8 mmul)
set(XDNA_TILE_K 64)         # per-core K tile (larger = fewer C reloads per call)
set(XDNA_TILE_N 64)         # per-core N tile (wider = longer B DDR bursts)
set(XDNA_N_COLS 8)          # AIE columns
set(XDNA_N_COMPUTE_ROWS 4)  # compute tile rows

target_compile_definitions(ggml-xdna PRIVATE
    GGML_XDNA_GEMM_M=${XDNA_GEMM_M}
    GGML_XDNA_TILE_M=${XDNA_TILE_M}
    GGML_XDNA_GEMM_M_BIG=${XDNA_GEMM_M_BIG}
    GGML_XDNA_TILE_M_BIG=${XDNA_TILE_M_BIG}
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

    # Geometry: "M tile_m bfp16(0/1)". Stems must match the runtime scan in
    # xdna_ops_init (gemm_bf16_f32_M%d_K%d_N%d_c%d).
    set(XDNA_GEOMETRIES
        "${XDNA_GEMM_M} ${XDNA_TILE_M} 0"
        "${XDNA_GEMM_M_BIG} ${XDNA_TILE_M_BIG} 1"
    )

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

    add_custom_target(ggml-xdna-kernels ALL DEPENDS ${XDNA_ARTIFACTS})

    install(FILES ${XDNA_ARTIFACTS} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
