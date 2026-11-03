set(TARGET llama-ui)

set(LLAMA_UI_HF_BUCKET "ggml-org/llama-ui" CACHE STRING "Hugging Face bucket name for prebuilt UI assets")
set(LLAMA_UI_GZIP      ON                  CACHE BOOL   "Apply gzip compress to assets to save bandwidth")

# Backward compat: forward old var to new one
if(DEFINED LLAMA_BUILD_WEBUI)
    set(LLAMA_BUILD_UI ${LLAMA_BUILD_WEBUI})
    message(DEPRECATION "LLAMA_BUILD_WEBUI is deprecated, use LLAMA_BUILD_UI instead")
endif()
if(DEFINED LLAMA_USE_PREBUILT_WEBUI)
    set(LLAMA_USE_PREBUILT_UI ${LLAMA_USE_PREBUILT_WEBUI})
    message(DEPRECATION "LLAMA_USE_PREBUILT_WEBUI is deprecated, use LLAMA_USE_PREBUILT_UI instead")
endif()
if(DEFINED LLAMA_WEBUI_HF_BUCKET)
    set(LLAMA_UI_HF_BUCKET ${LLAMA_WEBUI_HF_BUCKET})
    message(DEPRECATION "LLAMA_WEBUI_HF_BUCKET is deprecated, use LLAMA_UI_HF_BUCKET instead")
endif()

# Resolve HF asset version: explicit env var > upstream base tag > derived from build number > unset
if(DEFINED ENV{HF_WEBUI_VERSION})
    set(HF_UI_VERSION "$ENV{HF_WEBUI_VERSION}")
    message(DEPRECATION "HF_WEBUI_VERSION env var is deprecated, use HF_UI_VERSION instead")
elseif(DEFINED ENV{HF_UI_VERSION})
    set(HF_UI_VERSION "$ENV{HF_UI_VERSION}")
else()
    # In this fork LLAMA_BUILD_NUMBER counts the downstream commits too, so
    # "b${LLAMA_BUILD_NUMBER}" names a bucket tag upstream never published; the download
    # then 404s and falls back to 'latest', whose newer-era dist is missing assets this
    # tree's llama-ui-embed requires (loading.html) and the build fails. Pin to the
    # nearest reachable upstream release tag instead; the build-number derivation stays
    # as the fallback for tarball/tagless checkouts.
    execute_process(
        COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" describe --tags --abbrev=0 --match "b[0-9]*"
        OUTPUT_VARIABLE UPSTREAM_BASE_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT "${UPSTREAM_BASE_TAG}" STREQUAL "")
        set(HF_UI_VERSION "${UPSTREAM_BASE_TAG}")
    elseif(DEFINED LLAMA_BUILD_NUMBER)
        set(HF_UI_VERSION "b${LLAMA_BUILD_NUMBER}")
    else()
        set(HF_UI_VERSION "")
    endif()
endif()

if(NOT "${HF_UI_VERSION}" STREQUAL "" AND NOT HF_UI_VERSION MATCHES "^[A-Za-z0-9._-]+$")
    message(FATAL_ERROR "UI: invalid HF_UI_VERSION='${HF_UI_VERSION}' - must match ^[A-Za-z0-9._-]+$")
endif()

set(UI_CPP "${CMAKE_CURRENT_BINARY_DIR}/ui.cpp")
set(UI_H   "${CMAKE_CURRENT_BINARY_DIR}/ui.h")

# Provision assets and generate ui.cpp/ui.h natively in CMake at build time.
# The generated sources are compiled by the regular target toolchain; no
# build-time host executable is needed (works in any cross-compile setup).
# The script uses copy_if_different semantics, so the library below only
# recompiles when the generated contents actually change.
add_custom_target(llama-ui-assets ALL
    BYPRODUCTS ${UI_CPP} ${UI_H}
    COMMAND ${CMAKE_COMMAND}
        "-DUI_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DUI_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        "-DLLAMA_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
        "-DLLAMA_BUILD_NUMBER=${LLAMA_BUILD_NUMBER}"
        "-DHF_BUCKET=${LLAMA_UI_HF_BUCKET}"
        "-DHF_VERSION=${HF_UI_VERSION}"
        "-DHF_ENABLED=${LLAMA_USE_PREBUILT_UI}"
        "-DBUILD_UI=${LLAMA_BUILD_UI}"
        "-DLLAMA_UI_GZIP=${LLAMA_UI_GZIP}"
        -P "${PROJECT_SOURCE_DIR}/scripts/ui-assets.cmake"
    COMMENT "Provisioning UI assets"
    VERBATIM
)

set_source_files_properties(${UI_CPP} ${UI_H} PROPERTIES GENERATED TRUE)

add_library(${TARGET} STATIC ${UI_CPP} ${UI_H})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
add_dependencies(${TARGET} llama-ui-assets)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
)
