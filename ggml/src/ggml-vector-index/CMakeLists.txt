include(${CMAKE_CURRENT_LIST_DIR}/../ggml-vector-index-sources.cmake)

set(GGML_VECTOR_INDEX_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/../../include/ggml-vector-index.h
    ${GGML_VECTOR_INDEX_INTERNAL_HEADERS}
    ${GGML_VECTOR_INDEX_IMPL_SOURCES})

set(GGML_VECTOR_INDEX_AVX2_FLAG "")
set(GGML_VECTOR_INDEX_AVX2_CHECK_SOURCE [=[
    #include <immintrin.h>
    int main() {
        const __m128i bytes = _mm_set1_epi8(-1);
        const __m128i q16 = _mm_srai_epi16(_mm_unpacklo_epi8(bytes, bytes), 8);
        const __m128i q32_lo = _mm_srai_epi32(_mm_unpacklo_epi16(q16, q16), 16);
        const __m128i q32_hi = _mm_srai_epi32(_mm_unpackhi_epi16(q16, q16), 16);
        const __m128 lo = _mm_cvtepi32_ps(q32_lo);
        const __m128 hi = _mm_cvtepi32_ps(q32_hi);
        const __m256 q = _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
        const __m256 fma = _mm256_fmadd_ps(q, q, q);
        alignas(32) float out[8];
        _mm256_store_ps(out, fma);
        return out[0] == 0.0f ? 0 : 1;
    }
]=])

# Build this AVX2 object separately on x86 and keep runtime dispatch in the
# baseline TU.
if (GGML_SYSTEM_ARCH STREQUAL "x86")
    if (MSVC)
        check_cxx_compiler_flag("/arch:AVX2" GGML_VECTOR_INDEX_SUPPORTS_AVX2)
        if (GGML_VECTOR_INDEX_SUPPORTS_AVX2)
            set(GGML_VECTOR_INDEX_AVX2_FLAG "/arch:AVX2")
        endif()
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        check_cxx_compiler_flag("-mavx2" GGML_VECTOR_INDEX_SUPPORTS_MAVX2)
        check_cxx_compiler_flag("-mfma" GGML_VECTOR_INDEX_SUPPORTS_MFMA)
        if (GGML_VECTOR_INDEX_SUPPORTS_MAVX2 AND GGML_VECTOR_INDEX_SUPPORTS_MFMA)
            set(GGML_VECTOR_INDEX_AVX2_FLAG "-mavx2;-mfma")
        endif()
    endif()
endif()

if (GGML_VECTOR_INDEX_AVX2_FLAG)
    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
    string(REPLACE ";" " " GGML_VECTOR_INDEX_AVX2_REQUIRED_FLAGS "${GGML_VECTOR_INDEX_AVX2_FLAG}")
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${GGML_VECTOR_INDEX_AVX2_REQUIRED_FLAGS}")
    check_cxx_source_compiles(
        "${GGML_VECTOR_INDEX_AVX2_CHECK_SOURCE}"
        GGML_VECTOR_INDEX_SUPPORTS_AVX2_KERNEL)
    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
    if (NOT GGML_VECTOR_INDEX_SUPPORTS_AVX2_KERNEL)
        set(GGML_VECTOR_INDEX_AVX2_FLAG "")
    endif()
endif()

if (GGML_VECTOR_INDEX_AVX2_FLAG)
    list(APPEND GGML_VECTOR_INDEX_SOURCES
         ${GGML_VECTOR_INDEX_SOURCE_DIR}/ggml-vector-index-avx2.h
         ${GGML_VECTOR_INDEX_SOURCE_DIR}/ggml-vector-index-avx2.cpp)
    set_source_files_properties(
        ${GGML_VECTOR_INDEX_SOURCE_DIR}/ggml-vector-index-avx2.cpp
        PROPERTIES COMPILE_OPTIONS "${GGML_VECTOR_INDEX_AVX2_FLAG}")
endif()

add_library(ggml-vector-index ${GGML_VECTOR_INDEX_SOURCES})
add_library(ggml::ggml-vector-index ALIAS ggml-vector-index)
add_library(ggml::vector-index ALIAS ggml-vector-index)

set_property(TARGET ggml-vector-index PROPERTY
    GGML_VECTOR_INDEX_AVX2_COMPILE_OPTION "${GGML_VECTOR_INDEX_AVX2_FLAG}")

if (GGML_VECTOR_INDEX_AVX2_FLAG)
    target_compile_definitions(ggml-vector-index PRIVATE GGML_VEC_INDEX_HAVE_AVX2_KERNEL)
    set_property(TARGET ggml-vector-index PROPERTY GGML_VECTOR_INDEX_HAS_AVX2 TRUE)
else()
    set_property(TARGET ggml-vector-index PROPERTY GGML_VECTOR_INDEX_HAS_AVX2 FALSE)
endif()

set_target_properties(ggml-vector-index PROPERTIES
    VERSION ${GGML_VERSION}
    SOVERSION ${GGML_VERSION_MAJOR}
    PUBLIC_HEADER ${CMAKE_CURRENT_LIST_DIR}/../../include/ggml-vector-index.h
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_link_libraries(ggml-vector-index PRIVATE Threads::Threads)

set(GGML_VECTOR_INDEX_USE_ACCELERATE_QUERY OFF)
if (APPLE AND GGML_ACCELERATE)
    find_library(GGML_VECTOR_INDEX_ACCELERATE_FRAMEWORK Accelerate)
    if (GGML_VECTOR_INDEX_ACCELERATE_FRAMEWORK)
        set(GGML_VECTOR_INDEX_USE_ACCELERATE_QUERY ON)
        target_compile_definitions(
            ggml-vector-index PRIVATE
                GGML_VEC_INDEX_USE_ACCELERATE_QUERY)
        target_link_libraries(ggml-vector-index PRIVATE "-framework Accelerate")
    endif()
else()
    unset(GGML_VECTOR_INDEX_ACCELERATE_FRAMEWORK CACHE)
endif()

set(GGML_VECTOR_INDEX_PC_MATH_LIBS "")
if (DEFINED MATH_LIBRARY)
    target_link_libraries(ggml-vector-index PRIVATE ${MATH_LIBRARY})
    foreach(lib IN LISTS MATH_LIBRARY)
        if (lib MATCHES "^-" OR IS_ABSOLUTE "${lib}")
            list(APPEND GGML_VECTOR_INDEX_PC_MATH_LIBS "${lib}")
        else()
            list(APPEND GGML_VECTOR_INDEX_PC_MATH_LIBS "-l${lib}")
        endif()
    endforeach()
elseif (NOT WIN32 AND NOT DEFINED ENV{ONEAPI_ROOT})
    target_link_libraries(ggml-vector-index PRIVATE m)
    list(APPEND GGML_VECTOR_INDEX_PC_MATH_LIBS "-lm")
endif()

if (LLAMA_BUILD_TESTS)
    add_library(ggml-vector-index-test STATIC ${GGML_VECTOR_INDEX_SOURCES})
    target_compile_features(ggml-vector-index-test PRIVATE cxx_std_17)
    target_compile_definitions(ggml-vector-index-test PRIVATE GGML_VEC_INDEX_TEST_HOOKS)
    target_include_directories(
        ggml-vector-index-test
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../include)
    if (GGML_VECTOR_INDEX_AVX2_FLAG)
        target_compile_definitions(ggml-vector-index-test PRIVATE GGML_VEC_INDEX_HAVE_AVX2_KERNEL)
    endif()
    target_link_libraries(ggml-vector-index-test PRIVATE Threads::Threads)
    if (DEFINED MATH_LIBRARY)
        target_link_libraries(ggml-vector-index-test PRIVATE ${MATH_LIBRARY})
    elseif (NOT WIN32 AND NOT DEFINED ENV{ONEAPI_ROOT})
        target_link_libraries(ggml-vector-index-test PRIVATE m)
    endif()
    if (GGML_VECTOR_INDEX_USE_ACCELERATE_QUERY)
        target_compile_definitions(
            ggml-vector-index-test PRIVATE
                GGML_VEC_INDEX_USE_ACCELERATE_QUERY)
        target_link_libraries(ggml-vector-index-test PRIVATE "-framework Accelerate")
    endif()
endif()

install(
    TARGETS ggml-vector-index
    EXPORT ggml-targets
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(GGML_VECTOR_INDEX_PRIVATE_LIBS_LIST "")
foreach(lib IN LISTS CMAKE_CXX_IMPLICIT_LINK_LIBRARIES)
    if (lib STREQUAL "c++" OR lib STREQUAL "stdc++")
        list(APPEND GGML_VECTOR_INDEX_PRIVATE_LIBS_LIST "-l${lib}")
    endif()
endforeach()
if (CMAKE_THREAD_LIBS_INIT)
    list(APPEND GGML_VECTOR_INDEX_PRIVATE_LIBS_LIST "${CMAKE_THREAD_LIBS_INIT}")
endif()
if (GGML_VECTOR_INDEX_PC_MATH_LIBS)
    list(APPEND GGML_VECTOR_INDEX_PRIVATE_LIBS_LIST ${GGML_VECTOR_INDEX_PC_MATH_LIBS})
endif()
if (GGML_VECTOR_INDEX_USE_ACCELERATE_QUERY)
    list(APPEND GGML_VECTOR_INDEX_PRIVATE_LIBS_LIST "-framework" "Accelerate")
endif()
string(JOIN " " GGML_VECTOR_INDEX_PC_LIBS_PRIVATE ${GGML_VECTOR_INDEX_PRIVATE_LIBS_LIST})

configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/../../ggml-vector-index.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/ggml-vector-index.pc
    @ONLY)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/ggml-vector-index.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
