set(TARGET llama-simple)

add_executable(${TARGET} simple.cpp)

# Add the subfolder with vulkan hooks
# Add the qvac_lib_abstract_ipc subdirectory to get the C++ client library
option(USE_VULKAN_WRAPPERS "Use Vulkan wrapper functions" ON)
add_subdirectory(vulkan_patch_poc/qvac_lib_abstract_ipc)
if(USE_VULKAN_WRAPPERS)
    set(VULKAN_HOOKS_LIBS abstract_ipc_cpp)
    target_compile_definitions(${TARGET} PRIVATE USE_VULKAN_WRAPPERS)
    target_include_directories(
      ${TARGET}
      PRIVATE vulkan_patch_poc/qvac_lib_abstract_ipc/include
    )
else()
    set(VULKAN_HOOKS_LIBS "")
endif()

install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE ${VULKAN_HOOKS_LIBS} llama llama-common-test ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
