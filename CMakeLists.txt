cmake_minimum_required(VERSION 3.14...3.28) # for add_link_options and implicit target directories.
project("llama.cpp" C CXX)
include(CheckIncludeFileCXX)

### llama.cpp version
set(LLAMA_VERSION_MAJOR 0)
set(LLAMA_VERSION_MINOR 1)
set(LLAMA_VERSION_PATCH 2)
set(LLAMA_VERSION_BASE "${LLAMA_VERSION_MAJOR}.${LLAMA_VERSION_MINOR}.${LLAMA_VERSION_PATCH}")

# whether this is a development/nightly build
# set this to OFF when making a release from a release tag (vX.Y.Z)
# ref: https://github.com/ggml-org/ggml/discussions/1579
option(LLAMA_BUILD_IS_DEV "llama: dev build" ON)

if (LLAMA_BUILD_IS_DEV)
    set(LLAMA_VERSION "${LLAMA_VERSION_BASE}-dev")
else()
    # TODO: check that the current commit is tagged correctly according to the version specified above
    set(LLAMA_VERSION "${LLAMA_VERSION_BASE}")
endif()

message(STATUS "llama.cpp version: ${LLAMA_VERSION}")

#set(CMAKE_WARN_DEPRECATED YES)
set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LLAMA_STANDALONE ON)

    include(git-vars)
else()
    set(LLAMA_STANDALONE OFF)
endif()

option(LLAMA_USE_SYSTEM_GGML "Use system libggml" OFF)
option(BUILD_LLAMA "Build, install LLAMA package artifacts as well." ON)

option(LLAMA_WASM_MEM64 "llama: use 64-bit memory in WASM builds" ON)

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    # Use 64-bit memory to support backend_get_memory queries
    # TODO: analyze performance impact, see https://spidermonkey.dev/blog/2025/01/15/is-memory64-actually-worth-using
    if (LLAMA_WASM_MEM64)
      add_compile_options("-sMEMORY64=1")
      add_link_options("-sMEMORY64=1")
    endif()
    add_link_options("-sALLOW_MEMORY_GROWTH=1")

    option(LLAMA_WASM_SINGLE_FILE "llama: embed WASM inside the generated llama.js" OFF)
    option(LLAMA_BUILD_HTML "llama: build HTML file" ON)
    if (LLAMA_BUILD_HTML)
        set(CMAKE_EXECUTABLE_SUFFIX ".html")
    endif()
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

if (LLAMA_STANDALONE)
    # enable parallel builds for msbuild
    list(APPEND CMAKE_VS_GLOBALS UseMultiToolTask=true)
    list(APPEND CMAKE_VS_GLOBALS EnforceProcessCountAcrossBuilds=true)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(LLAMA_TOOLS_INSTALL_DEFAULT OFF)
else()
    set(LLAMA_TOOLS_INSTALL_DEFAULT ${LLAMA_STANDALONE})
endif()

# subprocess spawning isn't a supported/sandbox-friendly operation on mobile OSes or in WASM
if (CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "Android" OR ANDROID
        OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten" OR EMSCRIPTEN)
    set(LLAMA_SUBPROCESS_DEFAULT OFF)
else()
    set(LLAMA_SUBPROCESS_DEFAULT ON)
endif()

#
# option list
#

# debug
option(LLAMA_ALL_WARNINGS           "llama: enable all compiler warnings"                   ON)
option(LLAMA_ALL_WARNINGS_3RD_PARTY "llama: enable all compiler warnings in 3rd party libs" OFF)

# build
option(LLAMA_FATAL_WARNINGS "llama: enable -Werror flag" OFF)

# sanitizers
option(LLAMA_SANITIZE_THREAD    "llama: enable thread sanitizer"    OFF)
option(LLAMA_SANITIZE_ADDRESS   "llama: enable address sanitizer"   OFF)
option(LLAMA_SANITIZE_UNDEFINED "llama: enable undefined sanitizer" OFF)

# utils
option(LLAMA_BUILD_COMMON "llama: build common utils library" ${LLAMA_STANDALONE})

# extra artifacts
option(LLAMA_BUILD_TESTS     "llama: build tests"                                                                ${LLAMA_STANDALONE})
option(LLAMA_BUILD_TOOLS     "llama: build tools"                                                                ${LLAMA_STANDALONE})
option(LLAMA_BUILD_EXAMPLES  "llama: build examples"                                                             ${LLAMA_STANDALONE})
option(LLAMA_BUILD_SERVER    "llama: build server example"                                                       ${LLAMA_STANDALONE})
option(LLAMA_BUILD_APP       "llama: build the unified binary"                                                   ${LLAMA_STANDALONE})
option(LLAMA_BUILD_UI        "llama: build the embedded Web UI for server"                                       ON)
option(LLAMA_USE_PREBUILT_UI "llama: use prebuilt UI from HF Bucket when available (requires LLAMA_BUILD_UI=ON)" ON)

option(LLAMA_TOOLS_INSTALL "llama: install tools" ${LLAMA_TOOLS_INSTALL_DEFAULT})
option(LLAMA_TESTS_INSTALL "llama: install tests" ON)

# specific extras
option(LLAMA_MTMD "llama: multimodal support" ${LLAMA_BUILD_TOOLS})

# 3rd party libs
option(LLAMA_OPENSSL    "llama: use openssl to support HTTPS" ON)
option(LLAMA_SUBPROCESS "llama-common: use subprocess, required by server tools and server router mode" ${LLAMA_SUBPROCESS_DEFAULT})
option(LLAMA_LLGUIDANCE "llama-common: include LLGuidance library for structured output in common utils" OFF)

# profiling
option(FORCE_GGML_VK_PERF_LOGGER "Force vk performance logging in ggml" OFF)

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

if (NOT DEFINED LLAMA_BUILD_NUMBER)
    set(LLAMA_BUILD_NUMBER        ${BUILD_NUMBER})
endif()
if (NOT DEFINED LLAMA_BUILD_COMMIT)
    set(LLAMA_BUILD_COMMIT        ${BUILD_COMMIT})
endif()

# override ggml options
set(GGML_ALL_WARNINGS   ${LLAMA_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS ${LLAMA_FATAL_WARNINGS})

# change the default for these ggml options
if (NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE_DEFAULT ON)
endif()

if (NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()

# transition helpers
function (llama_option_depr TYPE OLD)
    if (${OLD})
        set(NEW "${ARGV2}")
        if(NEW)
            message(${TYPE} "${OLD} is deprecated, use ${NEW} instead")
            set(${NEW} ON PARENT_SCOPE)
        else()
            message(${TYPE} "${OLD} is deprecated and will be ignored")
        endif()
    endif()
endfunction()

llama_option_depr(FATAL_ERROR LLAMA_CUBLAS              GGML_CUDA)
llama_option_depr(WARNING     LLAMA_CUDA                GGML_CUDA)
llama_option_depr(WARNING     LLAMA_METAL               GGML_METAL)
llama_option_depr(WARNING     LLAMA_METAL_EMBED_LIBRARY GGML_METAL_EMBED_LIBRARY)
llama_option_depr(WARNING     LLAMA_NATIVE              GGML_NATIVE)
llama_option_depr(WARNING     LLAMA_RPC                 GGML_RPC)
llama_option_depr(WARNING     LLAMA_SYCL                GGML_SYCL)
llama_option_depr(WARNING     LLAMA_SYCL_F16            GGML_SYCL_F16)
llama_option_depr(WARNING     LLAMA_CANN                GGML_CANN)
llama_option_depr(WARNING     LLAMA_CURL)

include("cmake/license.cmake")
license_add_file("llama.cpp" "LICENSE")

#
# 3rd-party
#

if (LLAMA_USE_SYSTEM_GGML)
    message(STATUS "Using system-provided libggml, skipping ggml build")
    find_package(ggml REQUIRED)
    add_library(ggml ALIAS ggml::ggml)
endif()

if (NOT TARGET ggml AND NOT LLAMA_USE_SYSTEM_GGML)
    set(GGML_BUILD_NUMBER ${LLAMA_BUILD_NUMBER})
    set(GGML_BUILD_COMMIT ${LLAMA_BUILD_COMMIT})
    add_subdirectory(ggml)
    # ... otherwise assume ggml is added by a parent CMakeLists.txt
endif()

#
# build the library
#

if (NOT BUILD_LLAMA)
    message(STATUS "BUILD_LLAMA is OFF: skipping LLAMA targets.")
else()
add_subdirectory(src)

#
# utils, programs, examples and tests
#

# vendor/ and common/ expand $<INSTALL_INTERFACE:${LLAMA_INCLUDE_INSTALL_DIR}> at configure time
include(GNUInstallDirs)
set(LLAMA_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/llama CACHE PATH "Location of header files")

# default; vendor/cpp-httplib flips this ON when it links system OpenSSL
set(LLAMA_CPP_HTTPLIB_SYSTEM_OPENSSL OFF)

add_subdirectory(vendor)

if (LLAMA_BUILD_COMMON)
    add_subdirectory(common)
endif()

if(LLAMA_BUILD_EXAMPLES OR LLAMA_BUILD_TESTS)
    add_subdirectory(common_test)
endif()

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TESTS AND NOT CMAKE_JS_VERSION)
    include(CTest)
    add_subdirectory(tests)
endif()

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_EXAMPLES)
    add_subdirectory(examples)
    add_subdirectory(pocs)
endif()

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS)
    add_subdirectory(tools)
elseif (LLAMA_MTMD)
    add_subdirectory(tools/mtmd)
endif()

if (LLAMA_BUILD_APP)
    add_subdirectory(app)
endif()

# Standalone libmtmd build without pulling in the rest of the tools/ tree.
# Useful when packaging just the mtmd library for language bindings (e.g. an
# Apple XCFramework, or a WASM build). When the full tools build is enabled,
# mtmd is already built by the tools/ subdirectory above; this hook only fires
# when LLAMA_BUILD_TOOLS is OFF to avoid double-adding the target.
option(LLAMA_BUILD_MTMD "llama: build tools/mtmd library standalone" OFF)
if (LLAMA_BUILD_MTMD AND NOT (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS))
    add_subdirectory(tools/mtmd)
endif()

endif() # NOT BUILD_LLAMA

# Automatically add all files from the 'licenses' directory
file(GLOB EXTRA_LICENSES "${CMAKE_SOURCE_DIR}/licenses/LICENSE-*")

foreach(FILE_PATH ${EXTRA_LICENSES})
    get_filename_component(FILE_NAME "${FILE_PATH}" NAME)
    string(REGEX REPLACE "^LICENSE-" "" NAME "${FILE_NAME}")
    license_add_file("${NAME}" "${FILE_PATH}")
endforeach()

# TARGET check rather than LLAMA_BUILD_COMMON: with BUILD_LLAMA=OFF the
# common/ subdirectory is never added, and license_generate() fatals on a
# missing target
if (TARGET llama-common)
    license_generate(llama-common)
endif()

#
# install
#

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if (BUILD_LLAMA)
set(LLAMA_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/llama CACHE PATH "Location of header  files")
set(LLAMA_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(LLAMA_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

set(LLAMA_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/llama.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/llama-cpp.h)

set_target_properties(llama
    PROPERTIES
        PUBLIC_HEADER "${LLAMA_PUBLIC_HEADERS}")

install(
  TARGETS llama
  EXPORT llama-targets
  PUBLIC_HEADER
    DESTINATION ${LLAMA_INCLUDE_INSTALL_DIR})

if (LLAMA_BUILD_COMMON)

  install(
      TARGETS cpp-httplib
      EXPORT llama-targets)

  # llama-common links these PUBLIC, so they must be exported regardless of
  # LLAMA_CURL
  install(
      TARGETS nlohmann sheredom
      EXPORT llama-targets)

  install(FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/vendor/nlohmann/json.hpp
      ${CMAKE_CURRENT_SOURCE_DIR}/vendor/nlohmann/json_fwd.hpp
    DESTINATION ${LLAMA_INCLUDE_INSTALL_DIR}/nlohmann)

  install(FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/vendor/sheredom/subprocess.h
    DESTINATION ${LLAMA_INCLUDE_INSTALL_DIR}/sheredom)

  install(
    TARGETS llama-common llama-common-base
    EXPORT llama-targets
    PUBLIC_HEADER
      DESTINATION ${LLAMA_INCLUDE_INSTALL_DIR}/common)

endif()

if (LLAMA_MTMD AND TARGET mtmd)

  # a static mtmd records its PRIVATE vendor deps as $<LINK_ONLY:...> usage
  # requirements, so they must be part of the export set as well
  install(
      TARGETS vendor-hash miniaudio stb
      EXPORT llama-targets)

  if (NOT LLAMA_BUILD_COMMON) # otherwise already exported above
    install(TARGETS sheredom EXPORT llama-targets)
  endif()

  install(
    TARGETS mtmd
    EXPORT llama-targets
    PUBLIC_HEADER
      DESTINATION ${LLAMA_INCLUDE_INSTALL_DIR}/mtmd)

endif()

install(
  EXPORT llama-targets
  FILE llama-targets.cmake
  NAMESPACE llama::
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/llama)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/llama-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/llama-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/llama)

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/llama-config-version.cmake
        VERSION ${LLAMA_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/llama-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/llama-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/llama)

install(
    PROGRAMS convert_hf_to_gguf.py vulkan_profiling_analyzer.py
    DESTINATION ${CMAKE_INSTALL_BINDIR})

if (IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
    set(LLAMA_PC_PREFIX "${CMAKE_INSTALL_PREFIX}")
    set(LLAMA_PC_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
else()
    set(_LLAMA_PC_PREFIX_BASE "${CMAKE_CURRENT_BINARY_DIR}/pkgconfig-prefix")
    file(RELATIVE_PATH LLAMA_PC_PREFIX_RELATIVE
        "${_LLAMA_PC_PREFIX_BASE}/${CMAKE_INSTALL_LIBDIR}/pkgconfig"
        "${_LLAMA_PC_PREFIX_BASE}")
    set(LLAMA_PC_PREFIX "\${pcfiledir}/${LLAMA_PC_PREFIX_RELATIVE}")
    set(LLAMA_PC_LIBDIR "\${prefix}/${CMAKE_INSTALL_LIBDIR}")
endif()
if (IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
    set(LLAMA_PC_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}")
else()
    set(LLAMA_PC_INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
endif()

configure_file(cmake/llama.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/llama.pc"
        @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/llama.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif() # BUILD_LLAMA
