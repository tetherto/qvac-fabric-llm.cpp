# Shared Core ML runtime. Folded into each engine library under its
# own namespace; no new library is installed and the headers under include/
# are BUILD_INTERFACE only.
#
#   qvac_coreml_add_to(<engine-lib> NAMESPACE <ns> PREFIX <P> ENABLE <bool> [TEMPLATES <dirs>])
#       Compiles the runtime into <engine-lib> with QVAC_COREML_NS=<ns> when
#       ENABLE is true (the engine's Apple-only <P>_COREML option). <P> is the
#       engine's CMake option prefix (TTS_CPP, PARAKEET, AUDIOGEN).
#       With <P>_COREML_GENERATE (default ON on macOS, OFF on iOS) the
#       generator is compiled in too (recipe interpreter, MILBlob writer,
#       GgufSource, cache, on-device compile; needs the `ggml` target) and
#       every template directory in TEMPLATES is embedded uncompressed: one
#       .incbin assembler stub per file in __TEXT,__const plus a generated
#       {id, file, begin, end, sha256} table (src/templates.h). Configure fails
#       when the embedded files exceed QVAC_COREML_TEMPLATE_BUDGET_MB (default
#       32) per library. The template_dir / QVAC_COREML_TEMPLATE_DIR override is
#       compiled in only with <P>_BUILD_TESTS or QVAC_COREML_DEV.
#   qvac_coreml_add_tests([REQUIRE_GGML])
#       Portable tests (Linux and macOS): test-coreml-scheduler,
#       test-coreml-recipe, test-coreml-milblob, test-coreml-cache, plus the
#       host CLI qvac-coreml-tool when a `ggml` target exists (the generator's
#       GgufSource and the tool need ggml-base; the tests then also cover real
#       GGUFs). Each engine calls it from its *_BUILD_TESTS block with
#       REQUIRE_GGML, so a missing `ggml` target fails configure instead of
#       silently dropping the GGUF cases and the tool; only the standalone
#       coreml build leaves them optional. The first call wins in the
#       umbrella build.
#   qvac_coreml_add_link_test(TARGETS <libs...>)
#       macOS umbrella only: test-coreml-link links every listed engine library
#       and all of their runtime copies into one binary.
include_guard(GLOBAL)

set(QVAC_COREML_TEMPLATE_BUDGET_MB 32 CACHE STRING
    "Most Core ML template bytes (uncompressed, MiB) one library may embed; configure fails above it")
option(QVAC_COREML_DEV "Compile the Core ML template_dir / QVAC_COREML_TEMPLATE_DIR override into release builds" OFF)

function(_qvac_coreml_runtime_sources out_var with_objc)
    set(dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    set(srcs
        "${dir}/src/generation.cpp"
        "${dir}/src/hints.cpp"
        "${dir}/src/manager.cpp"
        "${dir}/src/options.cpp"
        "${dir}/src/pin.cpp"
        "${dir}/src/scheduler.cpp"
        "${dir}/src/sha256.cpp"
        "${dir}/src/status.cpp")
    if (with_objc)
        list(APPEND srcs "${dir}/src/apple/device.mm")
    else()
        list(APPEND srcs "${dir}/src/device_posix.cpp")
    endif()
    set(${out_var} ${srcs} PARENT_SCOPE)
endfunction()

# The portable generator: recipe parser and interpreter,
# MILBlob writer, keys, cache and failure records. No ggml. gguf_source.cpp
# (ggml-base) is separate. Compiled with -ffp-contract=off: the interpreter's
# arithmetic must be one IEEE operation per op on every host.
function(_qvac_coreml_generator_sources out_var)
    set(dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    set(${out_var}
        "${dir}/src/cache.cpp"
        "${dir}/src/digest.cpp"
        "${dir}/src/failed.cpp"
        "${dir}/src/interp.cpp"
        "${dir}/src/milblob.cpp"
        "${dir}/src/recipe.cpp"
        "${dir}/src/source_stat.cpp"
        "${dir}/src/templates.cpp"
        PARENT_SCOPE)
endfunction()

# Frameworks the Apple device facts need (src/apple/device.mm), plus Core ML
# for the on-device compile (src/apple/compile.mm) when `with_coreml`.
function(_qvac_coreml_apple_frameworks out_var)
    set(with_coreml OFF)
    if (ARGC GREATER 1)
        set(with_coreml ${ARGV1})
    endif()
    find_library(QVAC_COREML_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    set(fws "${QVAC_COREML_FOUNDATION_FRAMEWORK}")
    if (with_coreml)
        find_library(QVAC_COREML_COREML_FRAMEWORK CoreML REQUIRED)
        list(APPEND fws "${QVAC_COREML_COREML_FRAMEWORK}")
    endif()
    if (NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
        # SecCodeCopySelf (host identity for hints) is macOS-only.
        find_library(QVAC_COREML_SECURITY_FRAMEWORK Security REQUIRED)
        list(APPEND fws "${QVAC_COREML_SECURITY_FRAMEWORK}")
    endif()
    set(${out_var} ${fws} PARENT_SCOPE)
endfunction()

function(_qvac_coreml_configure_runtime target ns with_objc)
    set(dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    target_compile_definitions(${target} PRIVATE QVAC_COREML_NS=${ns})
    target_include_directories(${target} PRIVATE "${dir}/include" "${dir}/src" "${dir}/third_party")
    target_compile_features(${target} PRIVATE cxx_std_17)
    set_target_properties(${target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET     hidden
        OBJCXX_VISIBILITY_PRESET  hidden
        VISIBILITY_INLINES_HIDDEN ON)
    if (with_objc)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
    endif()
endfunction()

# Writes one .incbin stub per template file and the {id, file, begin, end,
# sha256} table, and fails configure over QVAC_COREML_TEMPLATE_BUDGET_MB.
# Sets <out_var> to the generated sources. `dirs` may be empty (empty table).
function(_qvac_coreml_embed_templates target ns gen out_var)
    set(dirs ${ARGN})
    file(MAKE_DIRECTORY "${gen}")
    set(srcs "")
    set(decls "")
    set(rows "")
    set(total 0)
    set(ids "")
    set(n 0)
    foreach (tdir IN LISTS dirs)
        get_filename_component(tdir "${tdir}" ABSOLUTE)
        get_filename_component(id "${tdir}" NAME)
        if (NOT EXISTS "${tdir}/template.json")
            message(FATAL_ERROR "qvac_coreml_add_to(${target}): ${tdir} is not a template directory (no template.json)")
        endif()
        file(READ "${tdir}/template.json" tjson)
        string(JSON json_id ERROR_VARIABLE json_err GET "${tjson}" id)
        if (json_err OR NOT json_id STREQUAL id)
            message(FATAL_ERROR "qvac_coreml_add_to(${target}): ${tdir}/template.json id must be the directory name '${id}'")
        endif()
        if (id IN_LIST ids)
            message(FATAL_ERROR "qvac_coreml_add_to(${target}): template id '${id}' is listed twice")
        endif()
        list(APPEND ids "${id}")
        # The embedded files: what recipe::Template::from_dir reads. The shape
        # manifest (reference-gguf.json) is test-only and never embedded.
        set(files Manifest.json model.mlmodel recipe.txt template.json)
        if (EXISTS "${tdir}/literals.bin")
            list(APPEND files literals.bin)
        endif()
        foreach (f IN LISTS files)
            set(path "${tdir}/${f}")
            if (NOT EXISTS "${path}")
                message(FATAL_ERROR "qvac_coreml_add_to(${target}): template ${id} lacks ${f}")
            endif()
            if (path MATCHES "[\"\\\\]")
                message(FATAL_ERROR "qvac_coreml_add_to(${target}): cannot embed ${path} (quote or backslash in the path)")
            endif()
            file(SIZE "${path}" size)
            math(EXPR total "${total} + ${size}")
            file(SHA256 "${path}" sha)
            # A changed template file reconfigures, so the table's sha256 and
            # the stub (which carries it) are rewritten.
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${path}")
            string(MAKE_C_IDENTIFIER "qvac_tpl_${ns}_${n}_${id}_${f}" sym)
            set(stub "${gen}/${sym}.cpp")
            file(CONFIGURE OUTPUT "${stub}" CONTENT
"// Generated by qvac_coreml.cmake: ${f} of Core ML template ${id}, embedded
// uncompressed in __TEXT,__const for ${target}.
static_assert(sizeof(\"sha256:${sha}\") > 1, \"the stub changes with its file\");
__asm__(
    \".section __TEXT,__const\\n\"
    \".globl _${sym}\\n\"
    \".private_extern _${sym}\\n\"
    \".p2align 4\\n\"
    \"_${sym}:\\n\"
    \".incbin \\\"${path}\\\"\\n\"
    \".globl _${sym}_end\\n\"
    \".private_extern _${sym}_end\\n\"
    \"_${sym}_end:\\n\");
" @ONLY)
            set_source_files_properties("${stub}" PROPERTIES OBJECT_DEPENDS "${path}")
            list(APPEND srcs "${stub}")
            string(APPEND decls "extern \"C\" const uint8_t ${sym}[];\nextern \"C\" const uint8_t ${sym}_end[];\n")
            string(APPEND rows "    {\"${id}\", \"${f}\", ${sym}, ${sym}_end, \"${sha}\"},\n")
            math(EXPR n "${n} + 1")
        endforeach()
    endforeach()

    set(budget_mb "${QVAC_COREML_TEMPLATE_BUDGET_MB}")
    if (NOT budget_mb MATCHES "^[0-9]+$")
        message(FATAL_ERROR "QVAC_COREML_TEMPLATE_BUDGET_MB must be a whole number of MiB (got '${budget_mb}')")
    endif()
    math(EXPR budget "${budget_mb} * 1048576")
    if (total GREATER budget)
        message(FATAL_ERROR "qvac_coreml_add_to(${target}): the embedded Core ML templates take ${total} bytes, over "
                            "QVAC_COREML_TEMPLATE_BUDGET_MB=${budget_mb} (${budget} bytes) for one library. "
                            "Drop a template or compress it offline (LZFSE).")
    endif()

    if (n EQUAL 0)
        set(table_body "const EmbeddedFile * embedded_files(size_t & count) {\n    count = 0;\n    return nullptr;\n}\n")
    else()
        set(table_body "namespace {\nconst EmbeddedFile FILES[] = {\n${rows}};\n}  // namespace\n\nconst EmbeddedFile * embedded_files(size_t & count) {\n    count = sizeof(FILES) / sizeof(FILES[0]);\n    return FILES;\n}\n")
    endif()
    file(CONFIGURE OUTPUT "${gen}/qvac_coreml_templates.cpp" CONTENT
"// Generated by qvac_coreml.cmake: the Core ML templates embedded in ${target}.
#include \"templates.h\"

${decls}
namespace ${ns} {
namespace templates {

${table_body}
}  // namespace templates
}  // namespace ${ns}
" @ONLY)
    list(APPEND srcs "${gen}/qvac_coreml_templates.cpp")
    if (TARGET ${target})
        set_property(TARGET ${target} PROPERTY QVAC_COREML_TEMPLATE_IDS "${ids}")
        set_property(TARGET ${target} PROPERTY QVAC_COREML_TEMPLATE_BYTES "${total}")
    endif()
    set(${out_var} ${srcs} PARENT_SCOPE)
    list(LENGTH ids n_ids)
    if (n_ids GREATER 0)
        message(STATUS "${target}: ${n_ids} Core ML template(s) embedded (${total} bytes): ${ids}")
    endif()
endfunction()

function(qvac_coreml_add_to target)
    cmake_parse_arguments(QCA "" "NAMESPACE;PREFIX;ENABLE" "TEMPLATES" ${ARGN})
    if (NOT TARGET ${target})
        message(FATAL_ERROR "qvac_coreml_add_to: ${target} is not a target")
    endif()
    if (NOT QCA_NAMESPACE OR NOT QCA_PREFIX)
        message(FATAL_ERROR "qvac_coreml_add_to(${target}): NAMESPACE and PREFIX are required")
    endif()
    if (NOT QCA_ENABLE)
        return()  # no Core ML in this build: nothing compiled, no behaviour change
    endif()
    if (NOT APPLE)
        message(FATAL_ERROR "qvac_coreml_add_to(${target}): the Core ML runtime is Apple-only")
    endif()

    # Generation (templates, recipe interpreter, on-device compile) defaults to
    # OFF on iOS: no templates are embedded there and only the runtime half
    # (background loads of prebuilt sidecars) is compiled.
    set(gen_default ON)
    if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
        set(gen_default OFF)
    endif()
    option(${QCA_PREFIX}_COREML_GENERATE "Compile the Core ML generator and embed its templates into ${target}" ${gen_default})
    set(generate ${${QCA_PREFIX}_COREML_GENERATE})
    set(dev OFF)
    if (QVAC_COREML_DEV OR ${QCA_PREFIX}_BUILD_TESTS)
        set(dev ON)
    endif()

    set(dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    set(rt ${target}.qvac-coreml)
    _qvac_coreml_runtime_sources(srcs ON)
    if (generate)
        if (NOT TARGET ggml)
            message(FATAL_ERROR "qvac_coreml_add_to(${target}): ${QCA_PREFIX}_COREML_GENERATE needs the `ggml` target "
                                "(GgufSource reads and dequantises through ggml-base); define it first or set "
                                "${QCA_PREFIX}_COREML_GENERATE=OFF")
        endif()
        _qvac_coreml_generator_sources(gen_srcs)
        _qvac_coreml_embed_templates(${target} ${QCA_NAMESPACE} "${CMAKE_CURRENT_BINARY_DIR}/qvac-coreml-${target}"
                                     tpl_srcs ${QCA_TEMPLATES})
        list(APPEND srcs ${gen_srcs} "${dir}/src/gguf_source.cpp" "${dir}/src/apple/compile.mm" ${tpl_srcs})
    elseif (QCA_TEMPLATES)
        message(STATUS "${target}: ${QCA_PREFIX}_COREML_GENERATE is OFF; no Core ML template is embedded")
    endif()
    add_library(${rt} OBJECT ${srcs})
    _qvac_coreml_configure_runtime(${rt} ${QCA_NAMESPACE} ON)
    if (generate)
        # The interpreter's arithmetic is one IEEE operation per op on every host.
        target_compile_options(${rt} PRIVATE -ffp-contract=off)
        target_link_libraries(${rt} PRIVATE ggml)
        target_compile_definitions(${rt} PRIVATE QVAC_COREML_GENERATE=1 QVAC_COREML_APPLE_COMPILE=1)
        target_compile_definitions(${target} PRIVATE QVAC_COREML_GENERATE=1)
    endif()
    if (dev)
        target_compile_definitions(${rt} PRIVATE QVAC_COREML_DEV=1)
    endif()
    target_sources(${target} PRIVATE $<TARGET_OBJECTS:${rt}>)

    # Engine sources that adopt the runtime see the same
    # namespace and headers.
    target_compile_definitions(${target} PRIVATE QVAC_COREML_NS=${QCA_NAMESPACE})
    target_include_directories(${target} PRIVATE $<BUILD_INTERFACE:${dir}/include>)

    # PUBLIC like each engine's own Core ML frameworks, for static consumers.
    _qvac_coreml_apple_frameworks(fws ${generate})
    target_link_libraries(${target} PUBLIC ${fws})
    set(pc_libs "")
    foreach (fw IN LISTS fws)
        get_filename_component(fw_name "${fw}" NAME_WE)
        list(APPEND pc_libs "-framework ${fw_name}")
    endforeach()

    set_property(TARGET ${target} PROPERTY QVAC_COREML_NS ${QCA_NAMESPACE})
    set_property(TARGET ${target} PROPERTY QVAC_COREML_PREFIX ${QCA_PREFIX})
    set_property(TARGET ${target} PROPERTY QVAC_COREML_RUNTIME ${rt})
    set_property(TARGET ${target} PROPERTY QVAC_COREML_GENERATE ${generate})
    # For engines that write a pkg-config Libs.private (qvac-parakeet.pc).
    set_property(TARGET ${target} PROPERTY QVAC_COREML_PKGCONFIG_LIBS "${pc_libs}")
    if (generate)
        message(STATUS "${target}: Core ML runtime and generator compiled in (namespace ${QCA_NAMESPACE})")
    else()
        message(STATUS "${target}: Core ML runtime compiled in (namespace ${QCA_NAMESPACE})")
    endif()
endfunction()

function(qvac_coreml_add_tests)
    cmake_parse_arguments(QAT "REQUIRE_GGML" "" "" ${ARGN})
    if (WIN32 OR TARGET test-coreml-scheduler)
        return()  # Linux and macOS only; registered once per build tree
    endif()
    if (QAT_REQUIRE_GGML AND NOT TARGET ggml)
        message(FATAL_ERROR "qvac_coreml_add_tests(REQUIRE_GGML): no `ggml` target. The engine must define "
                            "ggml before its tests block; without it qvac-coreml-tool and the GGUF cases of "
                            "the generator tests would not be built.")
    endif()
    set(dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    find_package(Threads REQUIRED)

    set(with_objc OFF)
    if (APPLE AND CMAKE_OBJCXX_COMPILER_LOADED)
        set(with_objc ON)
    endif()
    set(fws "")
    if (with_objc)
        _qvac_coreml_apple_frameworks(fws)
    endif()

    # The tests' own copy of the runtime and the generator, in its own namespace.
    _qvac_coreml_runtime_sources(srcs ${with_objc})
    _qvac_coreml_generator_sources(gen_srcs)
    set(have_ggml OFF)
    if (TARGET ggml)
        set(have_ggml ON)
        list(APPEND gen_srcs "${dir}/src/gguf_source.cpp")
    endif()
    # Test builds embed no template (tests read template directories from
    # the source tree): an empty table.
    _qvac_coreml_embed_templates(qvac-coreml-test-runtime qvac_coreml_test
                                 "${CMAKE_CURRENT_BINARY_DIR}/qvac-coreml-test-runtime" test_tpl)
    add_library(qvac-coreml-test-runtime STATIC ${srcs} ${gen_srcs} ${test_tpl})
    _qvac_coreml_configure_runtime(qvac-coreml-test-runtime qvac_coreml_test ${with_objc})
    # Scheduler test seams (src/test_hooks.h) and the template_dir override:
    # only in this copy.
    target_compile_definitions(qvac-coreml-test-runtime PRIVATE QVAC_COREML_TEST_HOOKS=1 QVAC_COREML_DEV=1)
    target_compile_options(qvac-coreml-test-runtime PRIVATE -ffp-contract=off)
    target_link_libraries(qvac-coreml-test-runtime PUBLIC Threads::Threads ${CMAKE_DL_LIBS} ${fws})
    if (have_ggml)
        target_link_libraries(qvac-coreml-test-runtime PUBLIC ggml)
        target_compile_definitions(qvac-coreml-test-runtime PUBLIC QVAC_COREML_HAVE_GGML=1)
        # The generation glue (src/generation.cpp) with the portable pipeline;
        # tests replace the on-device compile (generation::set_compile_override).
        target_compile_definitions(qvac-coreml-test-runtime PRIVATE QVAC_COREML_GENERATE=1)
    endif()

    # A dlopen-able module with a third copy, for the dlclose lifetime case.
    add_library(qvac-coreml-test-plugin MODULE "${dir}/test/test_plugin.cpp" ${srcs})
    _qvac_coreml_configure_runtime(qvac-coreml-test-plugin qvac_coreml_plugin ${with_objc})
    target_link_libraries(qvac-coreml-test-plugin PRIVATE Threads::Threads ${CMAKE_DL_LIBS} ${fws})

    add_executable(test-coreml-scheduler "${dir}/test/test_scheduler.cpp")
    target_compile_definitions(test-coreml-scheduler PRIVATE QVAC_COREML_NS=qvac_coreml_test)
    target_include_directories(test-coreml-scheduler PRIVATE "${dir}/include" "${dir}/src" "${dir}/third_party")
    target_compile_features(test-coreml-scheduler PRIVATE cxx_std_17)
    target_link_libraries(test-coreml-scheduler PRIVATE qvac-coreml-test-runtime)
    add_dependencies(test-coreml-scheduler qvac-coreml-test-plugin)

    add_test(NAME test-coreml-scheduler
             COMMAND test-coreml-scheduler --plugin $<TARGET_FILE:qvac-coreml-test-plugin>)
    # Own label so CI runs it in a dedicated step (no SKIP path exists: any
    # non-zero exit, including 77, is a failure).
    set_tests_properties(test-coreml-scheduler PROPERTIES LABELS "unit;coreml-runtime" TIMEOUT 300)

    # The portable generator's tests: per-test temp dirs, no network, no
    # Core ML. Same label and no-skip rule as the scheduler test.
    foreach (t recipe milblob cache)
        add_executable(test-coreml-${t} "${dir}/test/test_${t}.cpp")
        target_compile_definitions(test-coreml-${t} PRIVATE QVAC_COREML_NS=qvac_coreml_test)
        target_include_directories(test-coreml-${t} PRIVATE "${dir}/include" "${dir}/src" "${dir}/third_party" "${dir}/test")
        target_compile_features(test-coreml-${t} PRIVATE cxx_std_17)
        target_compile_options(test-coreml-${t} PRIVATE -ffp-contract=off)
        target_link_libraries(test-coreml-${t} PRIVATE qvac-coreml-test-runtime)
        set(args "")
        if (t STREQUAL "recipe" AND have_ggml)
            set(args --tool $<TARGET_FILE:qvac-coreml-tool>)
        endif()
        add_test(NAME test-coreml-${t} COMMAND test-coreml-${t} ${args})
        set_tests_properties(test-coreml-${t} PROPERTIES LABELS "unit;coreml-runtime" TIMEOUT 300)
    endforeach()

    # An engine lane without ggml fails configure (REQUIRE_GGML); this case
    # configures a project that has none and expects that error.
    add_test(NAME test-coreml-require-ggml
             COMMAND ${CMAKE_COMMAND} -S "${dir}/test/require_ggml"
                     -B "${CMAKE_CURRENT_BINARY_DIR}/test-coreml-require-ggml")
    set_tests_properties(test-coreml-require-ggml PROPERTIES LABELS "unit;coreml-runtime" TIMEOUT 120
                         PASS_REGULAR_EXPRESSION "qvac_coreml_add_tests\\(REQUIRE_GGML\\): no `ggml`")

    if (have_ggml)
        # Host CLI (README.md): dump | twin | f32 | generate | digest | verify.
        add_executable(qvac-coreml-tool "${dir}/tools/qvac-coreml-tool.cpp")
        target_compile_definitions(qvac-coreml-tool PRIVATE QVAC_COREML_NS=qvac_coreml_test)
        target_include_directories(qvac-coreml-tool PRIVATE "${dir}/include" "${dir}/src" "${dir}/third_party")
        target_compile_features(qvac-coreml-tool PRIVATE cxx_std_17)
        # `twin` computes mean + sd * normal() itself: no FMA contraction, so
        # twins stay byte-identical across hosts (x86-64 and arm64), as the
        # generator's own sources are.
        target_compile_options(qvac-coreml-tool PRIVATE -ffp-contract=off)
        target_link_libraries(qvac-coreml-tool PRIVATE qvac-coreml-test-runtime)
        # test-coreml-recipe drives `verify` through the CLI as well.
        add_dependencies(test-coreml-recipe qvac-coreml-tool)
        _qvac_coreml_add_template_tests("${dir}" ${with_objc})
    else()
        message(STATUS "qvac-coreml-tool and the GGUF cases of the generator tests: not built (no ggml target)")
    endif()

    if (with_objc)
        # A process that ends while a real Core ML load is in flight exits
        # cleanly. Needs a compiled model (QVAC_COREML_TEST_EXIT_MODEL, an
        # .mlmodelc whose cold load takes long enough to exit during it, such
        # as a vocoder with dilated convolutions); without one it registers
        # DISABLED.
        set(QVAC_COREML_TEST_EXIT_MODEL "" CACHE PATH
            "qvac-coreml: compiled model (.mlmodelc) for test-coreml-exit-load")
        add_executable(test-coreml-exit-load "${dir}/test/test_exit_load.mm")
        target_compile_definitions(test-coreml-exit-load PRIVATE QVAC_COREML_NS=qvac_coreml_test)
        target_include_directories(test-coreml-exit-load PRIVATE "${dir}/include" "${dir}/src")
        target_compile_features(test-coreml-exit-load PRIVATE cxx_std_17)
        target_compile_options(test-coreml-exit-load PRIVATE -fobjc-arc)
        _qvac_coreml_apple_frameworks(exit_fws ON)
        target_link_libraries(test-coreml-exit-load PRIVATE qvac-coreml-test-runtime ${exit_fws})
        add_test(NAME test-coreml-exit-load
                 COMMAND test-coreml-exit-load --model "${QVAC_COREML_TEST_EXIT_MODEL}")
        set_tests_properties(test-coreml-exit-load PROPERTIES LABELS "fixture" TIMEOUT 1200)
        if (NOT QVAC_COREML_TEST_EXIT_MODEL OR NOT EXISTS "${QVAC_COREML_TEST_EXIT_MODEL}")
            set_property(TEST test-coreml-exit-load PROPERTY DISABLED TRUE)
        endif()
    endif()

    if (APPLE)
        # Over the budget, configure fails (a project that embeds a toy
        # template under QVAC_COREML_TEMPLATE_BUDGET_MB=0).
        add_test(NAME test-coreml-template-budget
                 COMMAND ${CMAKE_COMMAND} -S "${dir}/test/template_budget"
                         -B "${CMAKE_CURRENT_BINARY_DIR}/test-coreml-template-budget"
                         -DQVAC_COREML_TEMPLATE_BUDGET_MB=0)
        set_tests_properties(test-coreml-template-budget PROPERTIES LABELS "unit;coreml-runtime" TIMEOUT 120
                             PASS_REGULAR_EXPRESSION "over[ \n]+QVAC_COREML_TEMPLATE_BUDGET_MB=0")
    endif()
endfunction()

# The toy-template, golden (generator tripwire) and coreml_template.py tests. Needs
# qvac-coreml-tool (a `ggml` target).
function(_qvac_coreml_add_template_tests dir with_objc)
    set(root "${dir}/..")
    get_filename_component(root "${root}" ABSOLUTE)
    set(toys "${dir}/test/toy-templates/toy-stateless" "${dir}/test/toy-templates/toy-multi"
             "${dir}/test/toy-templates/toy-state")

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
        # Tripwire: every template in the tree has a golden weight.bin digest for
        # the current generator version (stdlib only).
        add_test(NAME test-coreml-golden
                 COMMAND ${Python3_EXECUTABLE} -B "${dir}/python/golden.py" check
                         --tool $<TARGET_FILE:qvac-coreml-tool> --root "${root}" --golden "${dir}/test/golden.txt")
        add_test(NAME test-coreml-golden-selftest
                 COMMAND ${Python3_EXECUTABLE} -B "${dir}/python/test_golden.py"
                         --tool $<TARGET_FILE:qvac-coreml-tool> --toy "${dir}/test/toy-templates/toy-stateless")
        set_tests_properties(test-coreml-golden test-coreml-golden-selftest PROPERTIES
                             LABELS "unit;coreml-runtime" TIMEOUT 300)
        # coreml_template.py unit checks need numpy (the template CI job runs them
        # in the pinned coremltools venv as well).
        execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
                        RESULT_VARIABLE numpy_rc OUTPUT_QUIET ERROR_QUIET)
        if (numpy_rc EQUAL 0)
            add_test(NAME test-coreml-template-py
                     COMMAND ${Python3_EXECUTABLE} -B "${dir}/python/test_coreml_template.py")
            set_tests_properties(test-coreml-template-py PROPERTIES LABELS "unit;coreml-runtime" TIMEOUT 300)
        else()
            message(STATUS "test-coreml-template-py: not registered (${Python3_EXECUTABLE} has no numpy)")
        endif()
    else()
        message(STATUS "test-coreml-golden: not registered (no Python 3 interpreter)")
    endif()

    if (APPLE AND with_objc)
        # Toy templates embedded through the engines' own path (stubs, table,
        # generator, on-device compile) in a test-only library; macOS 15+.
        add_library(qvac-coreml-toy STATIC "${dir}/test/toy/toy_lib.cpp")
        set(QVAC_COREML_TOY_COREML_GENERATE ON CACHE BOOL "test-only toy template library: always generates" FORCE)
        set(QVAC_COREML_TOY_BUILD_TESTS ON)  # the template_dir override is compiled in
        qvac_coreml_add_to(qvac-coreml-toy NAMESPACE qvac_coreml_toy PREFIX QVAC_COREML_TOY ENABLE ON
                           TEMPLATES ${toys})
        target_link_libraries(qvac-coreml-toy PUBLIC ggml)
        add_executable(test-coreml-toy "${dir}/test/test_toy.mm")
        target_compile_definitions(test-coreml-toy PRIVATE QVAC_COREML_NS=qvac_coreml_toy)
        target_include_directories(test-coreml-toy PRIVATE "${dir}/include" "${dir}/src" "${dir}/third_party" "${dir}/test")
        target_compile_features(test-coreml-toy PRIVATE cxx_std_17)
        target_compile_options(test-coreml-toy PRIVATE -fobjc-arc)
        target_link_libraries(test-coreml-toy PRIVATE qvac-coreml-toy)
        add_dependencies(test-coreml-toy qvac-coreml-tool)
        add_test(NAME test-coreml-toy
                 COMMAND test-coreml-toy --tool $<TARGET_FILE:qvac-coreml-tool> --toys "${dir}/test/toy-templates")
        # Own label: it runs Core ML (generate, compile, load, predict) and
        # needs macOS 15, so CI runs it in a dedicated step on the supported
        # OS. It fails, never skips, below that floor.
        set_tests_properties(test-coreml-toy PROPERTIES LABELS "coreml-toy" TIMEOUT 600)

        # Embedded templates stay alive through exit-time static destructors.
        # No Core ML call, so it runs with the hermetic runtime tests.
        add_executable(test-coreml-template-lifetime "${dir}/test/test_template_lifetime.cpp")
        target_compile_definitions(test-coreml-template-lifetime PRIVATE QVAC_COREML_NS=qvac_coreml_toy)
        target_include_directories(test-coreml-template-lifetime PRIVATE "${dir}/include" "${dir}/src")
        target_compile_features(test-coreml-template-lifetime PRIVATE cxx_std_17)
        target_link_libraries(test-coreml-template-lifetime PRIVATE qvac-coreml-toy)
        add_test(NAME test-coreml-template-lifetime COMMAND test-coreml-template-lifetime)
        set_tests_properties(test-coreml-template-lifetime PROPERTIES LABELS "unit;coreml-runtime" TIMEOUT 60)
    endif()
endfunction()

function(qvac_coreml_add_link_test)
    cmake_parse_arguments(QLT "" "" "TARGETS" ${ARGN})
    if (NOT APPLE OR TARGET test-coreml-link)
        return()
    endif()
    set(dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    set(gen "${CMAKE_CURRENT_BINARY_DIR}/qvac-coreml-link")
    set(probe_srcs "")
    set(objs "")
    set(xlist "")
    foreach (t IN LISTS QLT_TARGETS)
        if (NOT TARGET ${t})
            message(STATUS "test-coreml-link: not registered (${t} is not built)")
            return()
        endif()
        get_target_property(ns ${t} QVAC_COREML_NS)
        get_target_property(rt ${t} QVAC_COREML_RUNTIME)
        if (NOT ns OR NOT rt)
            message(STATUS "test-coreml-link: not registered (${t} has no Core ML runtime; enable its *_COREML option)")
            return()
        endif()
        file(CONFIGURE OUTPUT "${gen}/probe_${ns}.cpp"
             CONTENT "#define QVAC_COREML_NS ${ns}\n#include \"link_probe.inc\"\n" @ONLY)
        list(APPEND probe_srcs "${gen}/probe_${ns}.cpp")
        string(APPEND xlist " X(${ns})")
        list(APPEND objs $<TARGET_OBJECTS:${rt}>)
    endforeach()
    file(CONFIGURE OUTPUT "${gen}/qvac_coreml_link_probes.h"
         CONTENT "#pragma once\n#define QVAC_COREML_LINK_PROBES(X)${xlist}\n" @ONLY)

    # The three engine libraries plus every object of every runtime copy in
    # one image: any runtime symbol that escaped its namespace (or an
    # Objective-C class) is a duplicate and fails the link. Whole-archive
    # loading of the engine libraries is not used: tts-cpp and qvac-parakeet
    # both compile dr_wav's implementation, a pre-existing collision outside
    # the runtime.
    add_executable(test-coreml-link "${dir}/test/test_link.cpp" ${probe_srcs} ${objs})
    target_include_directories(test-coreml-link PRIVATE
        "${gen}" "${dir}/include" "${dir}/src" "${dir}/third_party" "${dir}/test")
    target_compile_features(test-coreml-link PRIVATE cxx_std_17)
    target_link_libraries(test-coreml-link PRIVATE ${QLT_TARGETS})
    add_test(NAME test-coreml-link COMMAND test-coreml-link)
    set_tests_properties(test-coreml-link PROPERTIES LABELS "unit;coreml-runtime" TIMEOUT 120)
endfunction()
