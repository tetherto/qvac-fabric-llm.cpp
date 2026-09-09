# header-only: interface target exposing the vendor/ root so consumers
# can include via <sheredom/subprocess.h>
add_library(sheredom INTERFACE)
add_library(vendor::sheredom ALIAS sheredom)

target_include_directories(sheredom INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:${LLAMA_INCLUDE_INSTALL_DIR}>)
