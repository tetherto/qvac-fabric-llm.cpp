cmake_minimum_required(VERSION 3.20)

function(run)
    execute_process(COMMAND ${ARGV} RESULT_VARIABLE result)
    if (NOT result EQUAL 0)
        message(FATAL_ERROR "Hybrid package test failed: ${ARGV}")
    endif()
endfunction()

if (NOT DEFINED TEST_BINARY_DIR)
    message(FATAL_ERROR "TEST_BINARY_DIR is required")
endif()
set(source "${CMAKE_CURRENT_LIST_DIR}/hybrid-package")
set(prefix "${TEST_BINARY_DIR}/install")
run("${CMAKE_COMMAND}" -S "${source}" -B "${TEST_BINARY_DIR}/producer"
    -DHYBRID_PRODUCER=ON -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=${prefix}" ${HYBRID_CMAKE_ARGS})
run("${CMAKE_COMMAND}" --build "${TEST_BINARY_DIR}/producer" --config Release --parallel 4)
run("${CMAKE_COMMAND}" --install "${TEST_BINARY_DIR}/producer" --config Release)
file(READ "${TEST_BINARY_DIR}/producer/module-name-Release.txt" module_name)
run("${CMAKE_COMMAND}" -S "${source}" -B "${TEST_BINARY_DIR}/consumer"
    -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=${prefix}"
    "-DHYBRID_MODULE=${prefix}/test-modules/${module_name}"
    -DCMAKE_DISABLE_FIND_PACKAGE_OpenCL=ON -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON
    -DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=ON)
run("${CMAKE_COMMAND}" --build "${TEST_BINARY_DIR}/consumer" --config Release --parallel 4)
run("${CMAKE_CTEST_COMMAND}" --test-dir "${TEST_BINARY_DIR}/consumer" -C Release --output-on-failure)

function(reject_options name expected)
    execute_process(COMMAND "${CMAKE_COMMAND}" -S "${source}" -B "${TEST_BINARY_DIR}/${name}"
        -DHYBRID_PRODUCER=ON ${ARGN} RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if (result EQUAL 0 OR NOT "${output}${error}" MATCHES "${expected}")
        message(FATAL_ERROR "Expected rejection of ${name}: ${output}${error}")
    endif()
endfunction()

reject_options(cpu-disabled "GGML_CPU_STATIC requires GGML_CPU" -DGGML_CPU=OFF)
reject_options(cpu-variants "GGML_CPU_STATIC is incompatible with GGML_CPU_ALL_VARIANTS" -DGGML_CPU_ALL_VARIANTS=ON)
