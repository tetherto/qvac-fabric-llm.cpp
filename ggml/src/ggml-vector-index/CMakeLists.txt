add_library(ggml-vector-index
            ../../include/ggml-vector-index.h
            ../ggml-vector-index-impl.h
            ../ggml-vector-index.cpp)
add_library(ggml::vector-index ALIAS ggml-vector-index)
add_library(ggml::ggml-vector-index ALIAS ggml-vector-index)

set_target_properties(ggml-vector-index PROPERTIES
    VERSION ${GGML_VERSION}
    SOVERSION ${GGML_VERSION_MAJOR}
    PUBLIC_HEADER ../../include/ggml-vector-index.h
)

target_include_directories(ggml-vector-index PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(ggml-vector-index PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS OR GGML_BACKEND_DL)
    set_target_properties(ggml-vector-index PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(ggml-vector-index PRIVATE GGML_BUILD)
    target_compile_definitions(ggml-vector-index PUBLIC  GGML_SHARED)
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
string(JOIN " " GGML_VECTOR_INDEX_PRIVATE_LIBS ${GGML_VECTOR_INDEX_PRIVATE_LIBS_LIST})

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml-vector-index.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/ggml-vector-index.pc
    @ONLY)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/ggml-vector-index.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
