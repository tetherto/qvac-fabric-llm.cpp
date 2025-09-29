#include "gpu_device_tracker.h"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>

#ifdef VULKAN_GPU_INFO_SERVER_TRACE
#ifdef ANDROID
#include <android/log.h>
#else
#include <iostream>
#endif
#endif

GPUDeviceTracker::GPUDeviceTracker() : emptyDeviceVector() {}

GPUDeviceTracker::~GPUDeviceTracker() {}

VkPhysicalDeviceProperties
GPUDeviceTracker::isSoftwareEmulation(VkPhysicalDevice physicalDevice,
                                      bool &isEmulated) {
  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

  // Check if device name contains "llvmpipe" or other known software emulation
  // names
  const char *emulationKeywords[] = {"llvmpipe", "swrast", "lavapipe"};
  std::string deviceName(deviceProperties.deviceName);

  for (const auto &keyword : emulationKeywords) {
    if (deviceName.find(keyword) != std::string::npos) {
      isEmulated = true;
      return deviceProperties;
    }
  }

  isEmulated = false;
  return deviceProperties;
}

void GPUDeviceTracker::registerDevice(VkResult result,
                                      VkPhysicalDevice physicalDevice,
                                      VkDevice device) {
  if (result != VK_SUCCESS) {
    return;
  }

  std::unique_lock lock(mutex);

  // Check if this is a software emulation device
  bool isEmulated = false;
  VkPhysicalDeviceProperties deviceProperties =
      isSoftwareEmulation(physicalDevice, isEmulated);

#ifdef VULKAN_GPU_INFO_SERVER_TRACE
#ifdef ANDROID
  __android_log_print(ANDROID_LOG_INFO, "GpuDeviceTracker",
                      "registerDevice: Adding device %p to physical device %s",
                      device, deviceProperties.deviceName);
#else
  std::cout << "registerDevice: Adding device " << device
            << " to physical device " << deviceProperties.deviceName
            << std::endl;
#endif
#endif

  if (isEmulated) {
    emulatedPhysicalToLogicalDevices[physicalDevice].push_back(device);
  } else {
    physicalToLogicalDevices[physicalDevice].push_back(device);
  }

  deviceToPhysicalDevice[device] = physicalDevice;
}

void GPUDeviceTracker::unregisterDevice(VkDevice device) {
  std::unique_lock lock(mutex);

  auto physicalDeviceIt = deviceToPhysicalDevice.find(device);
  if (physicalDeviceIt != deviceToPhysicalDevice.end()) {
    VkPhysicalDevice physicalDevice = physicalDeviceIt->second;

    // Check if the device is in the normal or emulated map
    bool isEmulated = false;
    auto emulatedIt = emulatedPhysicalToLogicalDevices.find(physicalDevice);
    if (emulatedIt != emulatedPhysicalToLogicalDevices.end()) {
      isEmulated = true;
    }

#ifdef VULKAN_GPU_INFO_SERVER_TRACE
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO, "GpuDeviceTracker",
                        "unregisterDevice: Removing device %p", device);
#else
    std::cout << "unregisterDevice: Removing device " << device << std::endl;
#endif
#endif

    if (isEmulated) {
      auto &devices = emulatedPhysicalToLogicalDevices[physicalDevice];
      // Remove this device from the vector
      devices.erase(std::remove(devices.begin(), devices.end(), device),
                    devices.end());

      // If no more devices for this physical device, remove the entry
      if (devices.empty()) {
        emulatedPhysicalToLogicalDevices.erase(physicalDevice);
      }
    } else {
      auto &devices = physicalToLogicalDevices[physicalDevice];
      // Remove this device from the vector
      devices.erase(std::remove(devices.begin(), devices.end(), device),
                    devices.end());

      // If no more devices for this physical device, remove the entry
      if (devices.empty()) {
        physicalToLogicalDevices.erase(physicalDevice);
      }
    }

    deviceToPhysicalDevice.erase(physicalDeviceIt);
  }
}

VkPhysicalDevice GPUDeviceTracker::getPhysicalDevice(VkDevice device) const {
  std::shared_lock lock(mutex);

  auto it = deviceToPhysicalDevice.find(device);
  if (it != deviceToPhysicalDevice.end()) {
    return it->second;
  }

  return VK_NULL_HANDLE;
}

const std::vector<VkDevice> &
GPUDeviceTracker::getLogicalDevices(VkPhysicalDevice physicalDevice) const {
  std::shared_lock lock(mutex);

  // First check in hardware devices
  auto it = physicalToLogicalDevices.find(physicalDevice);
  if (it != physicalToLogicalDevices.end()) {
    return it->second;
  }

  // Then check in emulated devices
  auto emulatedIt = emulatedPhysicalToLogicalDevices.find(physicalDevice);
  if (emulatedIt != emulatedPhysicalToLogicalDevices.end()) {
    return emulatedIt->second;
  }

  return emptyDeviceVector;
}

VkPhysicalDevice
GPUDeviceTracker::getPhysicalDeviceAtIndex(uint32_t physicalDeviceIndex) const {
  std::shared_lock lock(mutex);

  if (physicalDeviceIndex >= physicalToLogicalDevices.size()) {
    return VK_NULL_HANDLE;
  }

  auto it = physicalToLogicalDevices.begin();
  std::advance(it, physicalDeviceIndex);
  return it->first;
}

uint32_t GPUDeviceTracker::getPhysicalDeviceCount() const {
  std::shared_lock lock(mutex);
  return physicalToLogicalDevices.size();
}
