cmake_minimum_required(VERSION 3.10)
project(VulkanGPUUtilizationTracker VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Vulkan package
find_package(Vulkan REQUIRED)

# Add performance logging definition for C++
option(USE_VULKAN_WRAPPERS "Use Vulkan wrapper functions" ON)

# Add the subfolder with vulkan hooks
add_subdirectory(vulkan_hooks)

# Add the qvac_lib_abstract_ipc subdirectory to get the C++ client library
add_subdirectory(qvac_lib_abstract_ipc)

# Create executable
add_executable(gpu_utilization_demo simple_compute_example.cpp)

# Add include directories
target_include_directories(
  gpu_utilization_demo
  PRIVATE ${Vulkan_INCLUDE_DIRS}
  PRIVATE vulkan_hooks
  PRIVATE qvac_lib_abstract_ipc/include
)

if(USE_VULKAN_WRAPPERS)
  # Additional compile definition only needed for the demo
  target_compile_definitions(gpu_utilization_demo PRIVATE USE_VULKAN_WRAPPERS)
  # Link order is important - link hooks BEFORE the Vulkan library
  target_link_libraries(gpu_utilization_demo PRIVATE vulkan_hooks
                                                     abstract_ipc_cpp
                                                     ${Vulkan_LIBRARIES})
else()
  target_link_libraries(gpu_utilization_demo PRIVATE abstract_ipc_cpp
                                                     ${Vulkan_LIBRARIES})
endif()

# Build instructions
message(STATUS "Build instructions:")
message(STATUS "1. Create a build directory: mkdir build && cd build")
message(STATUS "2. Run CMake: cmake ..")
message(STATUS "3. Build the project: make")
message(STATUS "4. Run the demo: ./gpu_utilization_demo")