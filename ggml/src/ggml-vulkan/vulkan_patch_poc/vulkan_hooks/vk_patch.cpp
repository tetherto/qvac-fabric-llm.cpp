#include "vk_patch.h"
#include "gpu_device_tracker.h"
#include "gpu_info_server.hpp"
#include "gpu_memory_tracker.h"
#include <atomic>
#include <stdexcept>
#include <stdlib.h>
#include <vulkan/vulkan_core.h>

#ifdef ANDROID
#include <android/log.h>
#endif

// Add Vulkan C++ headers for dispatcher patching
#ifdef __cplusplus
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

// Define the default dispatcher storage
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

// Forward declaration for init_vulkan_hooks
extern "C" {
    void init_vulkan_hooks();
}

namespace {
struct GpuTracker {
  GpuTracker()
      : deviceTracker(),
        memory(deviceTracker),
        gpuInfoServer(deviceTracker, memory) {}
  ~GpuTracker() = default;
  GPUDeviceTracker deviceTracker;
  GPUMemoryTracker memory;
  GpuInfoServer gpuInfoServer;
};
GpuTracker *gpuTracker = nullptr;
std::atomic_int vulkanInstanceCount{0};

// Function pointer types for original functions
typedef VkResult (*PFN_vkCreateInstance)(const VkInstanceCreateInfo *,
                                         const VkAllocationCallbacks *,
                                         VkInstance *);
typedef void (*PFN_vkDestroyInstance)(VkInstance,
                                      const VkAllocationCallbacks *);
typedef VkResult (*PFN_vkCreateDevice)(VkPhysicalDevice,
                                       const VkDeviceCreateInfo *,
                                       const VkAllocationCallbacks *,
                                       VkDevice *);
typedef void (*PFN_vkDestroyDevice)(VkDevice, const VkAllocationCallbacks *);
typedef VkResult (*PFN_vkAllocateMemory)(VkDevice, const VkMemoryAllocateInfo *,
                                         const VkAllocationCallbacks *,
                                         VkDeviceMemory *);
typedef void (*PFN_vkFreeMemory)(VkDevice, VkDeviceMemory,
                                 const VkAllocationCallbacks *);

// Original function pointers
PFN_vkCreateInstance real_vkCreateInstance = nullptr;
PFN_vkDestroyInstance real_vkDestroyInstance = nullptr;
PFN_vkCreateDevice real_vkCreateDevice = nullptr;
PFN_vkDestroyDevice real_vkDestroyDevice = nullptr;
PFN_vkAllocateMemory real_vkAllocateMemory = nullptr;
PFN_vkFreeMemory real_vkFreeMemory = nullptr;

} // namespace

#if defined(__linux__) || defined(__ANDROID__) || defined(ANDROID)
#include "init_vulkan_hooks_linux.cpp"
#elif defined(_WIN32) || defined(_WIN64)
#include "init_vulkan_hooks_windows.cpp"
#else
#error "Hooks dynamic loading not implemented for this platform"
#endif

// Dispatcher patching functions - define them BEFORE the DispatcherPatcher class
extern "C" {

void vkPatchDispatcherEarly(void* dispatcher_ptr) {
#ifdef __cplusplus
    if (!dispatcher_ptr) {
        return;
    }

    // Cast to the dispatcher type - use the correct namespace
    auto* dispatcher = static_cast<vk::detail::DispatchLoaderDynamic*>(dispatcher_ptr);

    // Patch the dispatcher's function pointers to use our hooked functions
    dispatcher->vkCreateInstance = vkCreateInstance;
    dispatcher->vkDestroyInstance = vkDestroyInstance;
    dispatcher->vkCreateDevice = vkCreateDevice;
    dispatcher->vkDestroyDevice = vkDestroyDevice;
    dispatcher->vkAllocateMemory = vkAllocateMemory;
    dispatcher->vkFreeMemory = vkFreeMemory;

#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO, "VulkanServerAddon",
                        "Early dispatcher patching completed");
#endif
#endif
}

// Override the dispatcher's init function to patch during initialization
void vkPatchDispatcherInit(void* dispatcher_ptr, void* getProcAddr_ptr) {
#ifdef __cplusplus
    if (!dispatcher_ptr || !getProcAddr_ptr) {
        return;
    }

    // Cast to the dispatcher type
    auto* dispatcher = static_cast<vk::detail::DispatchLoaderDynamic*>(dispatcher_ptr);

    // Use reinterpret_cast for function pointer (safer than static_cast)
    PFN_vkGetInstanceProcAddr getProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(getProcAddr_ptr);

    // Call the original init
    dispatcher->init(getProcAddr);

    // Immediately patch the function pointers
    vkPatchDispatcherEarly(dispatcher);
#endif
}

} // extern "C"

// NOW define the DispatcherPatcher class after the functions are defined
#ifdef __cplusplus
namespace {
// Automatic dispatcher patcher - runs before main()
class DispatcherPatcher {
public:
    DispatcherPatcher() {
        // Initialize hooks first
        init_vulkan_hooks();

        // Patch the default dispatcher immediately
        vkPatchDispatcherEarly(&VULKAN_HPP_DEFAULT_DISPATCHER);

#ifdef ANDROID
        __android_log_print(ANDROID_LOG_INFO, "VulkanServerAddon",
                            "Dispatcher patcher initialized at startup");
#endif
    }
};

// Global instance - constructor runs at startup
static DispatcherPatcher g_dispatcher_patcher;

} // namespace
#endif

double vkUtilization(uint32_t device_index) {
  (void)device_index; // Suppress unused parameter warning
  throw std::runtime_error("not implemented");
}

uint32_t vkPhysicalDeviceCount() {
  return gpuTracker->deviceTracker.getPhysicalDeviceCount();
}

uint64_t vkUsedMemory(uint32_t device_index) {
  return gpuTracker->memory.getUsedMemory(device_index);
}

uint64_t vkTotalMemory(uint32_t device_index) {
  return gpuTracker->memory.getTotalMemory(device_index);
}

// Override functions with the same name as the original Vulkan functions
extern "C" {

VkResult vkCreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkInstance *pInstance) {
#ifdef ANDROID
  __android_log_print(ANDROID_LOG_INFO, "VulkanServerAddon",
                      "vkCreateInstance called");
#endif
  init_vulkan_hooks();
#ifdef ANDROID
  __android_log_print(ANDROID_LOG_INFO, "VulkanServerAddon",
                      "init_vulkan_hooks done");
#endif
  VkResult result = real_vkCreateInstance(pCreateInfo, pAllocator, pInstance);
  if (result == VK_SUCCESS && vulkanInstanceCount.fetch_add(1) == 0 &&
      !gpuTracker) {
    gpuTracker = new GpuTracker();
    gpuTracker->gpuInfoServer.start();
  }
  return result;
}

void vkDestroyInstance(VkInstance instance,
                       const VkAllocationCallbacks *pAllocator) {
  real_vkDestroyInstance(instance, pAllocator);
  if (vulkanInstanceCount.fetch_sub(1) == 1) {
    if (gpuTracker) {
      gpuTracker->gpuInfoServer.stop();
      delete gpuTracker;
      gpuTracker = nullptr;
    }
  }
}

VkResult vkCreateDevice(VkPhysicalDevice physicalDevice,
                        const VkDeviceCreateInfo *pCreateInfo,
                        const VkAllocationCallbacks *pAllocator,
                        VkDevice *pDevice) {
  VkResult result =
      real_vkCreateDevice(physicalDevice, pCreateInfo, pAllocator, pDevice);
  gpuTracker->deviceTracker.registerDevice(result, physicalDevice, *pDevice);
  gpuTracker->memory.registerDevice(result, physicalDevice, *pDevice);
  return result;
}

VkResult vkAllocateMemory(VkDevice device,
                          const VkMemoryAllocateInfo *pAllocateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkDeviceMemory *pMemory) {
  VkResult result =
      real_vkAllocateMemory(device, pAllocateInfo, pAllocator, pMemory);
  gpuTracker->memory.trackMemoryAllocation(result, device, pAllocateInfo,
                                           pMemory);
  return result;
}

void vkFreeMemory(VkDevice device, VkDeviceMemory memory,
                  const VkAllocationCallbacks *pAllocator) {
  gpuTracker->memory.trackMemoryFree(device, memory);
  real_vkFreeMemory(device, memory, pAllocator);
}

void vkDestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator) {
  gpuTracker->memory.unregisterDevice(device);
  gpuTracker->deviceTracker.unregisterDevice(device);
  real_vkDestroyDevice(device, pAllocator);
}

} // extern "C"
