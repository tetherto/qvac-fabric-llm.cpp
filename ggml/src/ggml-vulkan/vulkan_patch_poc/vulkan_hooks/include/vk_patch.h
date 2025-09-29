#pragma once

#include <vulkan/vulkan.h>

// Original Vulkan functions are overriden at vk_patch.cpp
// Below there are additional declarations.

/// @brief Get the GPU utilization for a given device index tracked by patched
/// Vulkan functions
double vkUtilization(uint32_t device_index);

/// @brief Get the number of physical devices that can be used to fetch GPU
/// utilization with vkUtilization
uint32_t vkPhysicalDeviceCount();

/// @brief Get the total GPU memory capacity in bytes for a given device index
/// @note Android has unified memory, its unnecessary to track Vulkan
/// allocations.
uint64_t vkTotalMemory(uint32_t device_index);

/// @brief Get the currently used GPU memory in bytes for a given device index
uint64_t vkUsedMemory(uint32_t device_index);
