cmake_minimum_required(VERSION 3.25)
project(AbstractIpcCpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add Position Independent Code flag for all targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add option to build bare addon
option(BUILD_ADDON "Build bare addon" ON)
option(BUILD_TEST_BINARIES "Build test binaries" OFF)

if(ANDROID)
  set(CMAKE_CXX_EXTENSIONS OFF)    
  set(ANDROID_STL c++_static CACHE STRING "" FORCE)
  include_directories(SYSTEM "${HOST_TAG_DIR}/sysroot/usr/include/c++/v1")
  add_library(abstract_namespace_client abstract_ipc_clib/client_unix/client_unix.c)
else()
  # Add the abstract_namespace_server directory as a subdirectory
  add_subdirectory(abstract_ipc_clib)
endif()

# Include directories for finding headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add C++ library for AbstractIpcClient classes
add_library(abstract_ipc_cpp STATIC
    src/AbstractIpcClient.cpp
    src/AbstractIpcClientArray.cpp
)

target_include_directories(abstract_ipc_cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/abstract_ipc_clib
)

target_link_libraries(abstract_ipc_cpp PRIVATE abstract_namespace_client)
