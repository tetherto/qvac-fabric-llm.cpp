#pragma once
#include <cstdint>
#include <map>
#include <shared_mutex>
#include <vector>
#include <vulkan/vulkan.h>

class GPUDeviceTracker {
public:
  GPUDeviceTracker();
  ~GPUDeviceTracker();

  /// @brief Register a device with the device tracker
  void registerDevice(VkResult result, VkPhysicalDevice physicalDevice,
                      VkDevice device);

  /// @brief Unregister a device from the device tracker
  void unregisterDevice(VkDevice device);

  /// @brief Get physical device for a logical device
  VkPhysicalDevice getPhysicalDevice(VkDevice device) const;

  /// @brief Get all logical devices for a physical device
  const std::vector<VkDevice> &
  getLogicalDevices(VkPhysicalDevice physicalDevice) const;

  /// @brief Get physical device at index
  VkPhysicalDevice getPhysicalDeviceAtIndex(uint32_t physicalDeviceIndex) const;

  uint32_t getPhysicalDeviceCount() const;

  /// @brief Check if a physical device is a software emulation
  static VkPhysicalDeviceProperties
  isSoftwareEmulation(VkPhysicalDevice physicalDevice, bool &isEmulated);

private:
  mutable std::shared_mutex mutex;
  std::map<VkPhysicalDevice, std::vector<VkDevice>>
      emulatedPhysicalToLogicalDevices, physicalToLogicalDevices;
  std::map<VkDevice, VkPhysicalDevice> deviceToPhysicalDevice;
  std::vector<VkDevice> emptyDeviceVector;
};
