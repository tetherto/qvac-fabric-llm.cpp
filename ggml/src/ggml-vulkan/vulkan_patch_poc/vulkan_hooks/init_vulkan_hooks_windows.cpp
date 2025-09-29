#include <stdio.h>
#include <windows.h>

// Initialization function to load the real Vulkan functions
void init_vulkan_hooks() {
  static bool initialized = false;
  if (!initialized) {
    HMODULE vulkan_lib = LoadLibraryA("vulkan-1.dll");
    if (!vulkan_lib) {
      fprintf(stderr, "Failed to load Vulkan library: error code %lu\n",
              GetLastError());
      exit(1);
    }

    real_vkCreateInstance =
        (PFN_vkCreateInstance)GetProcAddress(vulkan_lib, "vkCreateInstance");
    real_vkDestroyInstance =
        (PFN_vkDestroyInstance)GetProcAddress(vulkan_lib, "vkDestroyInstance");
    real_vkCreateDevice =
        (PFN_vkCreateDevice)GetProcAddress(vulkan_lib, "vkCreateDevice");
    real_vkDestroyDevice =
        (PFN_vkDestroyDevice)GetProcAddress(vulkan_lib, "vkDestroyDevice");
    real_vkAllocateMemory =
        (PFN_vkAllocateMemory)GetProcAddress(vulkan_lib, "vkAllocateMemory");
    real_vkFreeMemory =
        (PFN_vkFreeMemory)GetProcAddress(vulkan_lib, "vkFreeMemory");

    initialized = true;
  }
}
