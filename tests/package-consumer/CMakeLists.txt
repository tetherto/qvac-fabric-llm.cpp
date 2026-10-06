cmake_minimum_required(VERSION 3.19)
project(ggml-package-consumer C CXX)

# A downstream project that consumes an installed ggml package the way the
# speech engines do: ggml::all for the backends the package links statically,
# ggml::ggml-vector-index when the package ships it.
option(GGML_CONSUMER_EXPECT_CPU "the package links the CPU backend statically" OFF)

find_package(ggml CONFIG REQUIRED)

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
