option(GGML_XDNA_BUILD_KERNELS "ggml-xdna: build NPU kernels with the IRON toolchain" ON)
option(GGML_XDNA_EXPERIMENTAL "ggml-xdna: enable experimental kernels and graph paths" OFF)
set(GGML_XDNA_TILE      "16384" CACHE STRING "ggml-xdna: elements per NPU submission")
set(GGML_XDNA_DMA_TILE  "1024"  CACHE STRING "ggml-xdna: DMA tile size inside the designs")
set(GGML_XDNA_RMS_ROW   "1024"  CACHE STRING "ggml-xdna: row length (ne0) baked into the RMS_NORM design")
set(GGML_XDNA_GEMM_MB "16" CACHE STRING "ggml-xdna: row blocks per MUL_MAT submit (M = MB*128)")
set(GGML_XDNA_IRON_PYTHON "$ENV{HOME}/ironenv/bin/python3" CACHE FILEPATH
    "ggml-xdna: python interpreter with the IRON toolchain (mlir_aie + llvm-aie)")

ggml_add_backend_library(ggml-xdna
                         ggml-xdna.cpp
                         xdna-npu.cpp
                        )

target_compile_definitions(ggml-xdna PRIVATE
    GGML_XDNA_TILE=${GGML_XDNA_TILE}
)
if (GGML_XDNA_EXPERIMENTAL)
    target_compile_definitions(ggml-xdna PRIVATE GGML_XDNA_EXPERIMENTAL=1)
endif()

# The host still packs GEMM operands into the swizzled layout the design reads,
# which is pure memory traffic and the largest single cost of a prefill step.
if (GGML_OPENMP)
    find_package(OpenMP)
    if (OpenMP_FOUND)
        message(STATUS "ggml-xdna: parallelising host operand rearrangement with OpenMP")
        target_compile_definitions(ggml-xdna PRIVATE GGML_XDNA_OPENMP=1)
        target_link_libraries(ggml-xdna PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
    endif()
endif()

# XRT is the NPU runtime. It ships as a system package; without it the backend
# still builds but reports no device.
set(_XRT_ROOT "$ENV{XILINX_XRT}")
if (NOT _XRT_ROOT)
    set(_XRT_ROOT "/opt/xilinx/xrt")
endif()

if (EXISTS "${_XRT_ROOT}/include/xrt/xrt_device.h" AND EXISTS "${_XRT_ROOT}/lib/libxrt_coreutil.so")
    message(STATUS "ggml-xdna: linking XRT from ${_XRT_ROOT}")
    target_compile_definitions(ggml-xdna PRIVATE GGML_XDNA_HAS_XRT=1)
    target_include_directories(ggml-xdna PRIVATE "${_XRT_ROOT}/include")
    target_link_directories(ggml-xdna PRIVATE "${_XRT_ROOT}/lib")
    target_link_libraries(ggml-xdna PRIVATE xrt_coreutil)
    set_target_properties(ggml-xdna PROPERTIES
        BUILD_RPATH   "${_XRT_ROOT}/lib"
        INSTALL_RPATH "${_XRT_ROOT}/lib"
    )

    # Standalone residency diagnostic. Links XRT only - no ggml - so its answer
    # about xclbin load/switch cost does not depend on the backend.
    add_executable(xdna-probe xdna-probe.cpp)
    target_include_directories(xdna-probe PRIVATE "${_XRT_ROOT}/include")
    target_link_directories(xdna-probe PRIVATE "${_XRT_ROOT}/lib")
    target_link_libraries(xdna-probe PRIVATE xrt_coreutil)

    # xrt::uuid::to_string, used for the CSV, resolves against libuuid.
    find_library(_XDNA_PROBE_UUID uuid)
    if (_XDNA_PROBE_UUID)
        target_link_libraries(xdna-probe PRIVATE "${_XDNA_PROBE_UUID}")
    endif()

    # aiebu ships with XRT and turns a transaction blob into an ELF, which is
    # what lets the same design be driven by a module instead of a per-submit
    # instruction buffer. Optional: without it the probe just skips that section.
    if (EXISTS "${_XRT_ROOT}/lib/libaiebu.a" AND EXISTS "${_XRT_ROOT}/include/aiebu/aiebu_assembler.h")
        message(STATUS "ggml-xdna: xdna-probe gets the module delivery path from ${_XRT_ROOT}/lib/libaiebu.a")
        target_compile_definitions(xdna-probe PRIVATE XDNA_PROBE_HAS_AIEBU=1)
        target_link_libraries(xdna-probe PRIVATE "${_XRT_ROOT}/lib/libaiebu.a")
    endif()

    set_target_properties(xdna-probe PROPERTIES
        BUILD_RPATH   "${_XRT_ROOT}/lib"
        INSTALL_RPATH "${_XRT_ROOT}/lib"
    )
else()
    message(STATUS "ggml-xdna: XRT not found at ${_XRT_ROOT} - backend will report no NPU device")
endif()

# NPU kernel artifacts. Built out-of-line by the IRON/MLIR-AIE toolchain so the
# host build never depends on it; the backend loads them at runtime from the
# directory of the executable (see ggml_xdna_find_*_kernel).
if (GGML_XDNA_BUILD_KERNELS)
    if (NOT EXISTS "${GGML_XDNA_IRON_PYTHON}")
        message(FATAL_ERROR
            "ggml-xdna: GGML_XDNA_BUILD_KERNELS=ON but no IRON python at "
            "${GGML_XDNA_IRON_PYTHON} - set -DGGML_XDNA_IRON_PYTHON=<venv>/bin/python3")
    endif()

    set(XDNA_KERNEL_ARTIFACTS "")
    set(XDNA_KERNEL_DESIGNS   "")

    # MUL_MAT: KT=16 (K=1024). Always build MB=16 and MB=32; runtime picks by M.
    set(_stem16 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-xdna-gemm-npu2-bf16-16-16")
    set(_stem32 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-xdna-gemm-npu2-bf16-16-32")
    list(APPEND XDNA_KERNEL_ARTIFACTS
         "${_stem16}.xclbin" "${_stem16}.insts.bin"
         "${_stem32}.xclbin" "${_stem32}.insts.bin")
    list(APPEND XDNA_KERNEL_DESIGNS "${CMAKE_CURRENT_SOURCE_DIR}/iron/ggml-xdna-gemm.py")

    set(XDNA_KERNEL_ENV "${CMAKE_COMMAND}" -E env)
    if (GGML_XDNA_EXPERIMENTAL)
        list(APPEND XDNA_KERNEL_ENV "GGML_XDNA_EXPERIMENTAL=1")

        foreach(op add mul)
            set(_stem "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-xdna-${op}-npu2-f32-${GGML_XDNA_TILE}")
            list(APPEND XDNA_KERNEL_ARTIFACTS "${_stem}.xclbin" "${_stem}.insts.bin")
            list(APPEND XDNA_KERNEL_DESIGNS   "${CMAKE_CURRENT_SOURCE_DIR}/iron/ggml-xdna-${op}.py")
        endforeach()

        # RMS_NORM is sized by the model's row length, not by the elementwise tile
        set(_rms "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-xdna-rms-norm-npu2-f32-${GGML_XDNA_RMS_ROW}")
        list(APPEND XDNA_KERNEL_ARTIFACTS "${_rms}.xclbin" "${_rms}.insts.bin")
        list(APPEND XDNA_KERNEL_DESIGNS   "${CMAKE_CURRENT_SOURCE_DIR}/iron/ggml-xdna-rms-norm.py")

        set(_fuse "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-xdna-add-rms-mul-npu2-f32-${GGML_XDNA_RMS_ROW}")
        list(APPEND XDNA_KERNEL_ARTIFACTS "${_fuse}.xclbin" "${_fuse}.insts.bin")
        list(APPEND XDNA_KERNEL_DESIGNS   "${CMAKE_CURRENT_SOURCE_DIR}/iron/ggml-xdna-add-rms-mul.py")

        # GATED_DELTA_NET: single-strip (legacy) + 4-worker full-state (w4).
        set(_gdn "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-xdna-gdn-npu2-s128-r32-cs64")
        set(_gdn4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-xdna-gdn-npu2-s128-r32-cs64-w4")
        list(APPEND XDNA_KERNEL_ARTIFACTS
             "${_gdn}.xclbin" "${_gdn}.insts.bin"
             "${_gdn4}.xclbin" "${_gdn4}.insts.bin")
        list(APPEND XDNA_KERNEL_DESIGNS "${CMAKE_CURRENT_SOURCE_DIR}/iron/ggml-xdna-gdn.py")
    endif()

    add_custom_command(
        OUTPUT  ${XDNA_KERNEL_ARTIFACTS}
        COMMAND ${XDNA_KERNEL_ENV} bash
                "${CMAKE_CURRENT_SOURCE_DIR}/iron/build-kernels.sh"
                "${GGML_XDNA_IRON_PYTHON}"
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
                "${GGML_XDNA_TILE}"
                "${GGML_XDNA_DMA_TILE}"
                "${CMAKE_CURRENT_BINARY_DIR}/kernels"
                "${GGML_XDNA_RMS_ROW}"
                "${GGML_XDNA_GEMM_MB}"
        DEPENDS ${XDNA_KERNEL_DESIGNS}
                "${CMAKE_CURRENT_SOURCE_DIR}/iron/build-kernels.sh"
        COMMENT "ggml-xdna: building NPU kernel artifacts"
        VERBATIM
    )

    add_custom_target(ggml-xdna-kernels ALL
        DEPENDS ${XDNA_KERNEL_ARTIFACTS}
    )
    add_dependencies(ggml-xdna ggml-xdna-kernels)

    install(FILES ${XDNA_KERNEL_ARTIFACTS}
            DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
