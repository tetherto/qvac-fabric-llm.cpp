find_package(Threads REQUIRED)

llama_add_compile_flags()

#
# llama-common-base
#

# Build info header

if(EXISTS "${PROJECT_SOURCE_DIR}/.git")
    set(GIT_DIR "${PROJECT_SOURCE_DIR}/.git")

    # Is git submodule
    if(NOT IS_DIRECTORY "${GIT_DIR}")
        file(READ ${GIT_DIR} REAL_GIT_DIR_LINK)
        string(REGEX REPLACE "gitdir: (.*)\n$" "\\1" REAL_GIT_DIR ${REAL_GIT_DIR_LINK})
        string(FIND "${REAL_GIT_DIR}" "/" SLASH_POS)
        if (SLASH_POS EQUAL 0)
            set(GIT_DIR "${REAL_GIT_DIR}")
        else()
            set(GIT_DIR "${PROJECT_SOURCE_DIR}/${REAL_GIT_DIR}")
        endif()
    endif()

    if(EXISTS "${GIT_DIR}/index")
        # For build-info.cpp below
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${GIT_DIR}/index")
    else()
        message(WARNING "Git index not found in git repository.")
    endif()
else()
    message(WARNING "Git repository not found; to enable automatic generation of build info, make sure Git is installed and the project is a Git repository.")
endif()

set(TEMPLATE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/build-info.cpp.in")
set(OUTPUT_FILE   "${CMAKE_CURRENT_BINARY_DIR}/build-info.cpp")

configure_file(${TEMPLATE_FILE} ${OUTPUT_FILE})

set(TARGET llama-common-base)
add_library(${TARGET} STATIC ${OUTPUT_FILE})

target_include_directories(${TARGET} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${LLAMA_INCLUDE_INSTALL_DIR}/common>)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

#
# llama-common
#

set(TARGET llama-common)

set(${TARGET}_HEADERS
    arg.h
    base64.hpp
    chat-auto-parser.h
    chat-peg-parser.h
    chat.h
    common.h
    console.h
    debug.h
    download.h
    finetune.h
    fit.h   # qvac: declarations for the fit helpers (memory_breakdown_print, get_device_memory_data)
    hf-cache.h
    http.h
    imatrix-loader.h
    json-schema-to-grammar.h
    log.h
    ngram-cache.h
    ngram-map.h
    ngram-mod.h
    peg-parser.h
    preset.h
    reasoning-budget.h
    sampling.h
    speculative.h
    subproc.h
    trie.h
    unicode.h
)

set(${TARGET}_JINJA_HEADERS
    jinja/caps.h
    jinja/lexer.h
    jinja/parser.h
    jinja/runtime.h
    jinja/string.h
    jinja/utils.h
    jinja/value.h
)

list(TRANSFORM ${TARGET}_HEADERS PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)
list(TRANSFORM ${TARGET}_JINJA_HEADERS PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)

add_library(${TARGET} STATIC
    arg.cpp
    chat-auto-parser-generator.cpp
    chat-auto-parser-helpers.cpp
    chat-diff-analyzer.cpp
    chat-peg-parser.cpp
    chat.cpp
    common.cpp
    console.cpp
    debug.cpp
    download.cpp
    finetune.cpp
    fit.cpp
    hf-cache.cpp
    imatrix-loader.cpp
    json-schema-to-grammar.cpp
    llguidance.cpp
    log.cpp
    ngram-cache.cpp
    ngram-map.cpp
    ngram-mod.cpp
    peg-parser.cpp
    preset.cpp
    reasoning-budget.cpp
    sampling.cpp
    speculative.cpp
    subproc.cpp
    trie.cpp
    unicode.cpp
    jinja/lexer.cpp
    jinja/parser.cpp
    jinja/runtime.cpp
    jinja/value.cpp
    jinja/string.cpp
    jinja/caps.cpp
    ${${TARGET}_HEADERS}
    ${${TARGET}_JINJA_HEADERS}
    )

set_target_properties(${TARGET} PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

set_target_properties(${TARGET} PROPERTIES PUBLIC_HEADER "${${TARGET}_HEADERS}")
target_link_libraries     (${TARGET} PUBLIC vendor::nlohmann vendor::sheredom)

install(FILES ${${TARGET}_JINJA_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/llama/common/jinja)

if (LLAMA_SUBPROCESS)
    target_compile_definitions(${TARGET} PUBLIC LLAMA_SUBPROCESS)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # TODO: make fine-grained exports in the future
    set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_link_libraries(${TARGET} PUBLIC  llama-common-base)
target_link_libraries(${TARGET} PRIVATE cpp-httplib)

if (LLAMA_LLGUIDANCE)
    include(ExternalProject)
    set(LLGUIDANCE_SRC ${CMAKE_BINARY_DIR}/llguidance/source)
    set(LLGUIDANCE_PATH ${LLGUIDANCE_SRC}/target/release)
    set(LLGUIDANCE_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}llguidance${CMAKE_STATIC_LIBRARY_SUFFIX}")

    ExternalProject_Add(llguidance_ext
        GIT_REPOSITORY https://github.com/guidance-ai/llguidance
        # v1.0.1:
        GIT_TAG d795912fedc7d393de740177ea9ea761e7905774
        PREFIX ${CMAKE_BINARY_DIR}/llguidance
        SOURCE_DIR ${LLGUIDANCE_SRC}
        BUILD_IN_SOURCE TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND cargo build --release --package llguidance
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS ${LLGUIDANCE_PATH}/${LLGUIDANCE_LIB_NAME} ${LLGUIDANCE_PATH}/llguidance.h
        UPDATE_COMMAND ""
    )
    target_compile_definitions(${TARGET} PUBLIC LLAMA_USE_LLGUIDANCE)

    add_library(llguidance STATIC IMPORTED)
    set_target_properties(llguidance PROPERTIES IMPORTED_LOCATION ${LLGUIDANCE_PATH}/${LLGUIDANCE_LIB_NAME})
    add_dependencies(llguidance llguidance_ext)

    target_include_directories(${TARGET} PRIVATE ${LLGUIDANCE_PATH})
    target_link_libraries(${TARGET} PRIVATE llguidance)
    if (WIN32)
        target_link_libraries(${TARGET} PRIVATE ws2_32 userenv ntdll bcrypt)
    endif()
endif()

target_include_directories(
  ${TARGET}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${CMAKE_CURRENT_SOURCE_DIR}/../vendor>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features   (${TARGET} PUBLIC cxx_std_17)
target_link_libraries     (${TARGET} PRIVATE ${LLAMA_COMMON_EXTRA_LIBS} PUBLIC llama Threads::Threads)

# qvac: rebase onto upstream b9341 renamed the common target to `llama-common`.
# Keep both names addressable so qvac-side CMakeLists (target `common`) and
# upstream/example/tool CMakeLists (target `llama-common`) both link.
if (NOT TARGET common)
    add_library(common ALIAS ${TARGET})
endif()


#
# copy the license files
#

# Check if running in GitHub Actions
if (DEFINED ENV{GITHUB_ACTIONS} AND "$ENV{GITHUB_ACTIONS}" STREQUAL "true")
    message(STATUS "Running inside GitHub Actions - copying license files")

    # Copy all files from licenses/ to build/bin/
    file(GLOB LICENSE_FILES "${CMAKE_SOURCE_DIR}/licenses/*")
    foreach(LICENSE_FILE ${LICENSE_FILES})
        get_filename_component(FILENAME ${LICENSE_FILE} NAME)
        add_custom_command(
            POST_BUILD
            TARGET ${TARGET}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${LICENSE_FILE}"
                "$<TARGET_FILE_DIR:llama>/${FILENAME}"
            COMMENT "Copying ${FILENAME} to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
        message(STATUS "Copying ${LICENSE_FILE} to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${FILENAME}")
    endforeach()
endif()
