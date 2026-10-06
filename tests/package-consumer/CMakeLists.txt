cmake_minimum_required(VERSION 3.19)
project(ggml-package-consumer C CXX)

# A downstream project that consumes an installed ggml package the way the
# speech engines do: ggml::all for the backends the package links statically,
# ggml::ggml-vector-index when the package ships it.
option(GGML_CONSUMER_EXPECT_CPU "the package links the CPU backend statically" OFF)

# Run-time search path for the installed shared libraries and their loaders,
# passed as a |-separated list because the driver hands it over in one argument.
set(GGML_CONSUMER_RPATH "" CACHE STRING "run-time library directories for the consumer")
string(REPLACE "|" ";" CMAKE_BUILD_RPATH "${GGML_CONSUMER_RPATH}")

find_package(ggml CONFIG REQUIRED)

get_target_property(GGML_CONSUMER_BASE_LINK ggml::ggml-base INTERFACE_LINK_LIBRARIES)
message(STATUS "consumer: ggml::ggml-base links ${GGML_CONSUMER_BASE_LINK}")

add_executable(consumer main.cpp)
target_compile_features(consumer PRIVATE cxx_std_17)
target_link_libraries(consumer PRIVATE ggml::ggml-base ggml::all)

if (GGML_CONSUMER_EXPECT_CPU)
    target_compile_definitions(consumer PRIVATE GGML_CONSUMER_EXPECT_CPU)
endif()

if (TARGET ggml::ggml-vector-index)
    target_link_libraries(consumer PRIVATE ggml::ggml-vector-index)
    target_compile_definitions(consumer PRIVATE GGML_CONSUMER_HAS_VECTOR_INDEX)
endif()

add_custom_target(run-consumer COMMAND consumer)
