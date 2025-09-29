#include <dlfcn.h>
#include <stdio.h>

#ifdef ANDROID
#include <android/log.h>
#endif

// Initialization function to load the real Vulkan functions
void init_vulkan_hooks() {
  static bool initialized = false;
  if (!initialized) {
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO, "VulkanServerAddon",
                        "init_vulkan_hooks called");
    void *vulkan_lib = dlopen("libvulkan.so", RTLD_LAZY | RTLD_NODELETE);
#else
    void *vulkan_lib = dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_NODELETE);
#endif
    if (!vulkan_lib) {
#ifdef ANDROID
      __android_log_print(ANDROID_LOG_ERROR, "VulkanServerAddon",
                          "Failed to load Vulkan library: %s", dlerror());
#endif
      fprintf(stderr, "Failed to load Vulkan library: %s\n", dlerror());
      return;
    }

    real_vkCreateInstance =
        (PFN_vkCreateInstance)dlsym(vulkan_lib, "vkCreateInstance");
    real_vkDestroyInstance =
        (PFN_vkDestroyInstance)dlsym(vulkan_lib, "vkDestroyInstance");
    real_vkCreateDevice =
        (PFN_vkCreateDevice)dlsym(vulkan_lib, "vkCreateDevice");
    real_vkDestroyDevice =
        (PFN_vkDestroyDevice)dlsym(vulkan_lib, "vkDestroyDevice");
    real_vkAllocateMemory =
        (PFN_vkAllocateMemory)dlsym(RTLD_NEXT, "vkAllocateMemory");
    real_vkFreeMemory = (PFN_vkFreeMemory)dlsym(RTLD_NEXT, "vkFreeMemory");

    initialized = true;
  }
}
