#include "gpu_memory_tracker.h"
#include <algorithm>
#include <iostream>
#include <mutex>

GPUMemoryTracker::GPUMemoryTracker(GPUDeviceTracker &tracker)
    : deviceTracker(tracker) {}

GPUMemoryTracker::~GPUMemoryTracker() {}

void GPUMemoryTracker::registerDevice(VkResult result,
                                      VkPhysicalDevice physicalDevice,
                                      VkDevice device) {
  if (result != VK_SUCCESS) {
    return;
  }

  VkPhysicalDeviceMemoryProperties memoryProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

  std::unique_lock<std::shared_mutex> lock(mutex);
  usedMemory[device] = 0;
  maxMemory[device] = memoryProperties.memoryHeaps[0].size;
}

void GPUMemoryTracker::unregisterDevice(VkDevice device) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  usedMemory.erase(device);
  maxMemory.erase(device);
}

void GPUMemoryTracker::trackMemoryAllocation(
    VkResult result, VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
    VkDeviceMemory *pMemory) {

  if (result != VK_SUCCESS) {
    return;
  }

  uint64_t allocationSize = pAllocateInfo->allocationSize;

  std::shared_lock<std::shared_mutex> lock(mutex);

  auto it = usedMemory.find(device);
  if (it == usedMemory.end()) {
    return;
  }

  uint64_t old = it->second.fetch_add(allocationSize);

#ifdef VULKAN_LOG_ALLOCATIONS
  std::cout << "Allocated " << allocationSize << " bytes on device " << device
            << " (total: " << old + allocationSize << ")" << std::endl;
#endif
}

void GPUMemoryTracker::trackMemoryFree(VkDevice device, VkDeviceMemory memory) {
  std::unique_lock lock(mutex);
  auto it = usedMemory.find(device);
  if (it == usedMemory.end()) {
    return;
  }

  uint64_t allocationSize;
  vkGetDeviceMemoryCommitment(device, memory, &allocationSize);

  uint64_t old = it->second.fetch_sub(allocationSize);

#ifdef VULKAN_LOG_ALLOCATIONS
  std::cout << "Freed " << allocationSize << " bytes on device " << device
            << " (total: " << old - allocationSize << ")" << std::endl;
#endif
}

uint64_t GPUMemoryTracker::getUsedMemory(uint32_t deviceIndex) const {
  VkPhysicalDevice physicalDevice =
      deviceTracker.getPhysicalDeviceAtIndex(deviceIndex);
  if (physicalDevice == VK_NULL_HANDLE) {
    std::cerr << "Could not find memory for physical device at index "
              << deviceIndex << " (tracked physical device count is: "
              << deviceTracker.getPhysicalDeviceCount() << ")" << std::endl;
    return 0;
  }

  const std::vector<VkDevice> &devices =
      deviceTracker.getLogicalDevices(physicalDevice);
#ifdef VULKAN_GPU_INFO_SERVER_TRACE
  std::cout << "getUsedMemory: Found " << devices.size()
            << " devices for physical device at index " << deviceIndex
            << std::endl;
  std::cout << "usedMemory size: " << usedMemory.size() << std::endl;
#endif

  std::shared_lock<std::shared_mutex> lock(mutex);
  uint64_t totalUsed = 0;

  for (VkDevice device : devices) {
    auto memIt = usedMemory.find(device);
    if (memIt != usedMemory.end()) {
      totalUsed += memIt->second.load();
    }
  }

  return totalUsed;
}

uint64_t GPUMemoryTracker::getTotalMemory(uint32_t deviceIndex) const {
  VkPhysicalDevice physicalDevice =
      deviceTracker.getPhysicalDeviceAtIndex(deviceIndex);
  if (physicalDevice == VK_NULL_HANDLE) {
    std::cerr << "Could not find max memory for physical device at index "
              << deviceIndex << " (tracked physical device count is: "
              << deviceTracker.getPhysicalDeviceCount() << ")" << std::endl;
    return 0;
  }

  const std::vector<VkDevice> &devices =
      deviceTracker.getLogicalDevices(physicalDevice);
  if (devices.empty()) {
#ifdef VULKAN_LOG_ALLOCATIONS
    std::cout << "No devices found for physical device at index " << deviceIndex
              << std::endl;
#endif
    return 0;
  }

  std::shared_lock<std::shared_mutex> lock(mutex);

  // All logical devices from the same physical device should have
  // the same total memory, so we just return the first one
  VkDevice device = devices[0];
  auto memIt = maxMemory.find(device);
  if (memIt != maxMemory.end()) {
    return memIt->second;
  }

  return 0;
}
