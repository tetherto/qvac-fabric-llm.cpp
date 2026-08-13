list(APPEND CMAKE_PREFIX_PATH /opt/xilinx/xrt)
find_package(AIEBU)
find_package(xrt REQUIRED)

ggml_add_backend_library(ggml-xdna
                         ggml-xdna.cpp
                         xdna-runtime.cpp
                         xdna-kernel-pool.cpp
                         xdna-ops.cpp
                        )

target_link_libraries(ggml-xdna PRIVATE XRT::xrt_coreutil)

# NPU kernels are compiled out-of-line by kernels/gemm.py (IRON/MLIR-AIE) at
# build time. Each variant is a (K N n_cols) tuple; M is fixed by the bf16
# geometry. Artifacts land in the runtime output directory and are loaded from
# there (or from GGML_XDNA_KERNELS_DIR) at runtime.
option(GGML_XDNA_BUILD_KERNELS "ggml-xdna: build NPU GEMM kernels (requires IRON)" ON)
set(GGML_XDNA_GEMM_PYTHON "" CACHE FILEPATH
    "ggml-xdna: python interpreter with IRON (mlir_aie + llvm-aie) installed")

# Native bf16 uses the r=4 mmul, so tile-m = 8 and the M block is 32.
set(XDNA_GEMM_M 32)
set(XDNA_TILE_M 8)
set(XDNA_TILE_K 16)

# GEMM variants to build: "K N n_cols". K must be a multiple of 8, N of 128.
set(GGML_XDNA_GEMM_VARIANTS
    "1024 2048 8"
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

    foreach(v IN LISTS GGML_XDNA_GEMM_VARIANTS)
        string(REPLACE " " ";" FIELDS "${v}")
        list(GET FIELDS 0 K_VAL)
        list(GET FIELDS 1 N_VAL)
        list(GET FIELDS 2 COLS)

        # Artifact stem matches the runtime cache key exactly.
        set(KERNEL_STEM "gemm_bf16_f32_M${XDNA_GEMM_M}_K${K_VAL}_N${N_VAL}_c${COLS}")
        set(XCLBIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${KERNEL_STEM}.xclbin")
        set(INSTS  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${KERNEL_STEM}.insts.bin")

        # tile_n=32 needs N to be a multiple of 32 * n_aie_cols (256); N<256
        # uses the k=16/n=16 fallback.
        if (N_VAL GREATER_EQUAL 256)
            set(XDNA_TILE_N_VAL 32)
        else()
            set(XDNA_TILE_N_VAL 16)
        endif()

        add_custom_command(
            OUTPUT ${XCLBIN} ${INSTS}
            COMMAND ${GEMM_PYTHON} "${GEMM_DIR}/gemm.py"
                    -M ${XDNA_GEMM_M} -K ${K_VAL} -N ${N_VAL}
                    --tile-m ${XDNA_TILE_M}
                    --tile-k ${XDNA_TILE_K}
                    --tile-n ${XDNA_TILE_N_VAL}
                    --n-aie-cols ${COLS}
                    -d npu2
                    --xclbin-path ${XCLBIN}
                    --insts-path ${INSTS}
            DEPENDS "${GEMM_DIR}/gemm.py"
            WORKING_DIRECTORY "${GEMM_DIR}"
            COMMENT "ggml-xdna: compiling GEMM kernel ${XDNA_GEMM_M}x${K_VAL}x${N_VAL} c${COLS}"
            VERBATIM
        )

        list(APPEND XDNA_ARTIFACTS ${XCLBIN} ${INSTS})
    endforeach()

    add_custom_target(ggml-xdna-kernels ALL DEPENDS ${XDNA_ARTIFACTS})

    install(FILES ${XDNA_ARTIFACTS} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
