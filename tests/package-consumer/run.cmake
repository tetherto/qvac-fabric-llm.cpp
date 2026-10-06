# Installs the ggml build into a scratch prefix, builds and runs the consumer
# project against that install, then checks the pkg-config metadata names
# libraries that exist in the installed libdir.
#
# Inputs (-D): GGML_BUILD_DIR, GGML_CONSUMER_SOURCE_DIR, GGML_LIB_OUTPUT_PREFIX,
#              GGML_CONSUMER_EXPECT_CPU, GGML_CONSUMER_CONFIG (multi-config generators)

set(prefix "${GGML_BUILD_DIR}/package-consumer/prefix")
set(consumer_build "${GGML_BUILD_DIR}/package-consumer/build")

set(config_args "")
if (GGML_CONSUMER_CONFIG)
    set(config_args --config "${GGML_CONSUMER_CONFIG}")
endif()

function(run_checked)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE result)
    if (NOT result EQUAL 0)
        message(FATAL_ERROR "command failed (${result}): ${ARGN}")
    endif()
endfunction()

function(install_package)
    file(REMOVE_RECURSE "${prefix}" "${consumer_build}")
    run_checked(${CMAKE_COMMAND} --install "${GGML_BUILD_DIR}" --prefix "${prefix}" ${config_args})
endfunction()

function(build_and_run_consumer)
    set(build_type_args "")
    if (GGML_CONSUMER_CONFIG)
        set(build_type_args "-DCMAKE_BUILD_TYPE=${GGML_CONSUMER_CONFIG}")
    endif()
    run_checked(${CMAKE_COMMAND} -S "${GGML_CONSUMER_SOURCE_DIR}" -B "${consumer_build}"
        "-DCMAKE_PREFIX_PATH=${prefix}"
        "-DGGML_CONSUMER_EXPECT_CPU=${GGML_CONSUMER_EXPECT_CPU}"
        ${build_type_args})
    run_checked(${CMAKE_COMMAND} --build "${consumer_build}" --target run-consumer ${config_args})
endfunction()

function(pkg_config_query out_var pkgconfig_dir)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env "PKG_CONFIG_PATH=${pkgconfig_dir}" "PKG_CONFIG_LIBDIR=${pkgconfig_dir}"
                "${PKG_CONFIG_EXECUTABLE}" ${ARGN}
        OUTPUT_VARIABLE output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE result)
    if (NOT result EQUAL 0)
        message(FATAL_ERROR "pkg-config ${ARGN} failed (${result})")
    endif()
    set(${out_var} "${output}" PARENT_SCOPE)
endfunction()

# Every ggml library the metadata names must exist under the installed libdir;
# system libraries in Libs.private (m, c++, frameworks) live elsewhere.
function(check_lib_names_exist libs libdir)
    string(REPLACE " " ";" tokens "${libs}")
    foreach(token IN LISTS tokens)
        if (NOT token MATCHES "^-l(${GGML_LIB_OUTPUT_PREFIX}ggml.*)$")
            continue()
        endif()
        set(name "${CMAKE_MATCH_1}")
        file(GLOB found "${libdir}/lib${name}.*" "${libdir}/${name}.*")
        if (NOT found)
            message(FATAL_ERROR "pkg-config names ${name} but nothing matches it in ${libdir}")
        endif()
    endforeach()
endfunction()

function(check_pkg_config)
    find_program(PKG_CONFIG_EXECUTABLE NAMES pkg-config pkgconf)
    if (NOT PKG_CONFIG_EXECUTABLE)
        message(STATUS "pkg-config not found, skipping the metadata check")
        return()
    endif()
    file(GLOB pc_files "${prefix}/*/pkgconfig/ggml*.pc")
    if (NOT pc_files)
        message(STATUS "no pkg-config files installed, skipping the metadata check")
        return()
    endif()
    list(GET pc_files 0 first_pc)
    get_filename_component(pkgconfig_dir "${first_pc}" DIRECTORY)
    foreach(pc_file IN LISTS pc_files)
        get_filename_component(module "${pc_file}" NAME_WE)
        pkg_config_query(libs "${pkgconfig_dir}" --libs --static "${module}")
        pkg_config_query(libdir "${pkgconfig_dir}" --variable=libdir "${module}")
        set(expected_name "${GGML_LIB_OUTPUT_PREFIX}${module}")
        if (NOT libs MATCHES "-l${expected_name}( |$)")
            message(FATAL_ERROR "${module}.pc does not link -l${expected_name}: ${libs}")
        endif()
        check_lib_names_exist("${libs}" "${libdir}")
    endforeach()
endfunction()

install_package()
build_and_run_consumer()
check_pkg_config()
