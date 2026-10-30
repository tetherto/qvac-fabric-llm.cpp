# header-only: interface target exposing the vendor/ root so consumers
# can include via <miniaudio/miniaudio.h>
add_library(miniaudio INTERFACE)
add_library(vendor::miniaudio ALIAS miniaudio)

target_include_directories(miniaudio INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:${LLAMA_INCLUDE_INSTALL_DIR}>)
