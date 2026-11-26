# GNU backends import the shared core through its C API.
function(qvac_configure_cxx_runtimes)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        return()
    endif()
    set(cuda_module OFF)
    if(TARGET ggml-cuda)
        get_target_property(cuda_type ggml-cuda TYPE)
        get_target_property(cuda_dir ggml-cuda SOURCE_DIR)
        get_directory_property(cuda_compiler_id DIRECTORY "${cuda_dir}"
            DEFINITION CMAKE_CUDA_COMPILER_ID)
        if(cuda_type STREQUAL "MODULE_LIBRARY" AND cuda_compiler_id STREQUAL "NVIDIA")
            set(cuda_module ON)
        endif()
    endif()
    set(xdna_module OFF)
    if(TARGET ggml-xdna)
        get_target_property(xdna_type ggml-xdna TYPE)
        if(xdna_type STREQUAL "MODULE_LIBRARY")
            set(xdna_module ON)
        endif()
    endif()
    if(NOT cuda_module AND NOT xdna_module)
        return()
    endif()

    set(stdcxx_targets)
    if(cuda_module)
        list(APPEND stdcxx_targets ggml-cuda)
        if(TARGET ggml-cuda-cutlass)
            list(APPEND stdcxx_targets ggml-cuda-cutlass)
        endif()
    endif()
    if(xdna_module)
        list(APPEND stdcxx_targets ggml-xdna ggml-cpu-xdna)
    endif()
    foreach(target IN LISTS stdcxx_targets)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-stdlib=libstdc++>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-stdlib=libstdc++>)
        endif()
    endforeach()
    set(modules)
    if(cuda_module)
        list(APPEND modules ggml-cuda)
    endif()
    if(xdna_module)
        list(APPEND modules ggml-xdna)
    endif()
    foreach(target IN LISTS modules)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_link_options(${target} PRIVATE -stdlib=libstdc++)
        endif()
        # Only backend entry points cross the module boundary.
        set(exports "${CMAKE_CURRENT_BINARY_DIR}/${target}-exports.map")
        file(WRITE "${exports}" "{ global: ggml_backend_init; ggml_backend_score; local: *; };\n")
        set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${exports}")
        target_link_options(${target} PRIVATE
            "LINKER:--version-script=${exports}" LINKER:-z,defs)
    endforeach()
    if(cuda_module)
        # The weak pure-virtual reference does not pull this from the archive.
        target_link_options(ggml-cuda PRIVATE
            -static-libstdc++ LINKER:--undefined=__cxa_pure_virtual
            LINKER:--as-needed)
        # CMake may append an implicit shared -lstdc++ for CUDA. Resolve the
        # runtime from its archive first, then discard the unused shared copy.
        target_link_libraries(ggml-cuda PRIVATE
            "-Wl,-Bstatic,--whole-archive" stdc++ "-Wl,--no-whole-archive,-Bdynamic")
    endif()
    if(xdna_module)
        # XRT exchanges C++ objects with this module and needs shared libstdc++.
        # Suppress the driver's implicit runtime, including the triplet's static
        # runtime option, and link the shared library explicitly.
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_link_options(ggml-xdna PRIVATE -nostdlib++)
        endif()
        # Resolve the GNU C++ surface before considering the core's libc++.
        get_target_property(libraries ggml-xdna LINK_LIBRARIES)
        set_property(TARGET ggml-xdna PROPERTY LINK_LIBRARIES stdc++ ${libraries})
        set_target_properties(ggml-xdna PROPERTIES INSTALL_RPATH "$ORIGIN;${QVAC_XRT_ROOT}/lib")
    endif()
endfunction()
