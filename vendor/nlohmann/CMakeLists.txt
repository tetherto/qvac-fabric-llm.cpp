# header-only: interface target exposing the vendor/ root so consumers
# can include via <nlohmann/json.hpp>
add_library(nlohmann INTERFACE)
add_library(vendor::nlohmann ALIAS nlohmann)

target_include_directories(nlohmann INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:${LLAMA_INCLUDE_INSTALL_DIR}>)
