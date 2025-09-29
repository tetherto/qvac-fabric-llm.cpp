cmake_minimum_required(VERSION 3.10)
project(VulkanHooks VERSION 1.0)

option(ENABLE_VULKAN_PERFORMANCE_LOG_TRACE "Enable performance debug logging" OFF)
option(ENABLE_VULKAN_PERFORMANCE_LOG_UTILIZATION_UPDATES "Enable utilization logging" OFF)
option(ENABLE_VULKAN_LOG_ALLOCATIONS "Enable allocation logs" OFF)
option(ENABLE_VULKAN_GPU_INFO_SERVER_TRACE "Trace requests to IPC server" OFF)
option(ENABLE_VULKAN_GPU_CALIBRATION_TRACE "Trace physical device calibration process" OFF)

# include(FetchContent)
# FetchContent_Declare(
#   spirv-reflect
#   GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Reflect.git
#   GIT_TAG vulkan-sdk-1.4.309.0
# )
#
# # Just download the source code
# FetchContent_GetProperties(spirv-reflect)
# if(NOT spirv-reflect_POPULATED)
#   FetchContent_Populate(spirv-reflect)
# endif()
#
# # Create our own spirv-reflect library with proper export settings
# add_library(spirv_reflect_wrapper STATIC
#   ${spirv-reflect_SOURCE_DIR}/spirv_reflect.c
# )
#
# target_include_directories(spirv_reflect_wrapper
#   PUBLIC
#     $<BUILD_INTERFACE:${spirv-reflect_SOURCE_DIR}>
#     $<INSTALL_INTERFACE:include/spirv-reflect>
# )

# Install the header file
# install(FILES ${spirv-reflect_SOURCE_DIR}/spirv_reflect.h
#         DESTINATION include/spirv-reflect)

# Set library for both scopes
# set(VULKAN_HOOKS_LIBRARIES spirv_reflect_wrapper)
set(VULKAN_HOOKS_LIBRARIES ${VULKAN_HOOKS_LIBRARIES} PARENT_SCOPE)

# Define source files for both current and parent scopes
set(VULKAN_HOOKS_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_device_tracker.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_memory_tracker.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_info_server.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/vk_patch.cpp
)
if(WIN32)
  list(APPEND VULKAN_HOOKS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../qvac_lib_abstract_ipc/abstract_ipc_clib/server_windows/server_windows.c)
else()
  list(APPEND VULKAN_HOOKS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../qvac_lib_abstract_ipc/abstract_ipc_clib/server_unix/server_unix.c)
endif()
# Make it available to parent scope as well
set(VULKAN_HOOKS_SOURCES ${VULKAN_HOOKS_SOURCES} PARENT_SCOPE)

# Define include directories for both current and parent scopes
set(VULKAN_HOOKS_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../qvac_lib_abstract_ipc/abstract_ipc_clib
)
if(WIN32)
  list(APPEND VULKAN_HOOKS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../qvac_lib_abstract_ipc/abstract_ipc_clib/server_windows/include)
else()
  list(APPEND VULKAN_HOOKS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../qvac_lib_abstract_ipc/abstract_ipc_clib/server_unix/include)
endif()

# Make it available to parent scope as well
set(VULKAN_HOOKS_INCLUDE_DIRS ${VULKAN_HOOKS_INCLUDE_DIRS} PARENT_SCOPE)

# Define compile definitions based on options
set(VULKAN_HOOKS_COMPILE_DEFINITIONS "")
if(ENABLE_VULKAN_PERFORMANCE_LOG_TRACE)
  list(APPEND VULKAN_HOOKS_COMPILE_DEFINITIONS VULKAN_PERFORMANCE_LOG_TRACE)
endif()
if(ENABLE_VULKAN_PERFORMANCE_LOG_UTILIZATION_UPDATES)
  list(APPEND VULKAN_HOOKS_COMPILE_DEFINITIONS VULKAN_PERFORMANCE_LOG_UTILIZATION_UPDATES)
endif()
if(ENABLE_VULKAN_LOG_ALLOCATIONS)
  list(APPEND VULKAN_HOOKS_COMPILE_DEFINITIONS VULKAN_LOG_ALLOCATIONS)
endif()
if(ENABLE_VULKAN_GPU_INFO_SERVER_TRACE)
  list(APPEND VULKAN_HOOKS_COMPILE_DEFINITIONS VULKAN_GPU_INFO_SERVER_TRACE)
endif()
if(ENABLE_VULKAN_GPU_CALIBRATION_TRACE)
  list(APPEND VULKAN_HOOKS_COMPILE_DEFINITIONS VULKAN_GPU_CALIBRATION_TRACE)
endif()
# Make it available to both current and parent scopes
set(VULKAN_HOOKS_COMPILE_DEFINITIONS ${VULKAN_HOOKS_COMPILE_DEFINITIONS} PARENT_SCOPE)

# Vulkan hooks library target
add_library(vulkan_hooks STATIC ${VULKAN_HOOKS_SOURCES})
target_link_libraries(vulkan_hooks PRIVATE ${VULKAN_HOOKS_LIBRARIES})
target_include_directories(
  vulkan_hooks
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../qvac_lib_abstract_ipc/abstract_ipc_clib
    ${CMAKE_CURRENT_SOURCE_DIR}/../qvac_lib_abstract_ipc/abstract_ipc_clib/server_unix/include
    $<INSTALL_INTERFACE:include>
)
foreach(def ${VULKAN_HOOKS_COMPILE_DEFINITIONS})
  target_compile_definitions(vulkan_hooks PRIVATE ${def})
endforeach()

# Add -fPIC flag for linking into shared libraries
set_target_properties(vulkan_hooks PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Export targets to parent scope
set(VULKAN_HOOKS_EXPORT_TARGETS vulkan_hooks PARENT_SCOPE)
