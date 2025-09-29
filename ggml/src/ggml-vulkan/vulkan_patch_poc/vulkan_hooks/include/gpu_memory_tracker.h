#pragma once
#include "gpu_device_tracker.h"
#include <atomic>
#include <cstdint>
#include <map>
#include <shared_mutex>
#include <vulkan/vulkan.h>

class GPUMemoryTracker {
public:
  GPUMemoryTracker(GPUDeviceTracker &deviceTracker);
  ~GPUMemoryTracker();

  void registerDevice(VkResult result, VkPhysicalDevice physicalDevice,
                      VkDevice device);
  void unregisterDevice(VkDevice device);

  void trackMemoryAllocation(VkResult result, VkDevice device,
                             const VkMemoryAllocateInfo *pAllocateInfo,
                             VkDeviceMemory *pMemory);
  void trackMemoryFree(VkDevice device, VkDeviceMemory memory);

  // Get current memory usage for a specific device index
  uint64_t getUsedMemory(uint32_t deviceIndex) const;

  // Get total memory capacity for a specific device index
  uint64_t getTotalMemory(uint32_t deviceIndex) const;

private:
  mutable std::shared_mutex mutex;
  std::map<VkDevice, std::atomic_uint64_t> usedMemory;
  std::map<VkDevice, uint64_t> maxMemory;
  GPUDeviceTracker &deviceTracker;
};
