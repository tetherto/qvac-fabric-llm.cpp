#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <vulkan/vulkan.h>
#include <atomic>

// Include the C++ IPC client classes
#include "qvac_lib_abstract_ipc/include/AbstractIpcClient.hpp"
#include "qvac_lib_abstract_ipc/include/AbstractIpcClientArray.hpp"

// Only needed if you want to directly call additional vkUtilization function
#ifdef USE_VULKAN_WRAPPERS
#include "vulkan_hooks/include/vk_patch.h"
#endif

// Thread-specific Vulkan resources struct
struct VulkanResources {
  VkPhysicalDevice physicalDevice;
  VkDevice device;
  VkQueue computeQueue;
  uint32_t computeQueueFamilyIndex;
  VkCommandPool commandPool;
  VkBuffer buffer;
  VkDeviceMemory bufferMemory;
  VkDescriptorPool descriptorPool;
  VkDescriptorSetLayout descriptorSetLayout;
  VkDescriptorSet descriptorSet;
  VkPipelineLayout pipelineLayout;
  VkPipeline computePipeline;
  VkShaderModule computeShaderModule;
};

// Shared Vulkan instance
VkInstance instance;

// Mutex for thread-safe console output
std::mutex consoleMutex;

// Global variables for memory polling
std::atomic<bool> shouldStopMemoryPolling{false};
std::atomic<double> maxMemoryUsage{0.0};
std::thread memoryPollingThread;

// Buffer size
const size_t BUFFER_SIZE = 65536 * sizeof(float); // 256KB
const int NUM_FRAMES = 5;                         // Number of frames to run

// Base workload size
const uint32_t BASE_SIZE = 8;

float workloadMultiplier = 1.0f;
int numThreads = 1;

// Memory polling configuration
const int MEMORY_POLL_INTERVAL_MS = 100; // Poll every 100ms
const int MEMORY_USAGE_REQUEST = 86;     // Request type for memory usage (from gpu_info_server.hpp)

// Read shader code from a file
std::vector<char> readFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();

  return buffer;
}

// Load shader module from SPIR-V file
VkShaderModule createShaderModule(const std::vector<char> &code,
                                  VkDevice device) {
  VkShaderModuleCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size();
  createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create shader module!");
  }

  return shaderModule;
}

void createInstance() {
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Simple Compute Example";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan instance");
  }
}

void setupDevice(VulkanResources &resources) {
  // Select first available physical device
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
  resources.physicalDevice = devices[0];

  // Find queue family with compute support
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(resources.physicalDevice,
                                           &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(
      resources.physicalDevice, &queueFamilyCount, queueFamilies.data());

  for (uint32_t i = 0; i < queueFamilyCount; i++) {
    if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      resources.computeQueueFamilyIndex = i;
      break;
    }
  }

  // Create logical device with compute queue
  float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo queueCreateInfo = {};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = resources.computeQueueFamilyIndex;
  queueCreateInfo.queueCount = 1;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  VkDeviceCreateInfo deviceCreateInfo = {};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

  if (vkCreateDevice(resources.physicalDevice, &deviceCreateInfo, nullptr,
                     &resources.device) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create logical device");
  }

  vkGetDeviceQueue(resources.device, resources.computeQueueFamilyIndex, 0,
                   &resources.computeQueue);
}

static void setupComputeResources(VulkanResources &resources) {
  // Create command pool
  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = resources.computeQueueFamilyIndex;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  vkCreateCommandPool(resources.device, &poolInfo, nullptr,
                      &resources.commandPool);

  // Create storage buffer
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = BUFFER_SIZE;
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vkCreateBuffer(resources.device, &bufferInfo, nullptr, &resources.buffer);

  // Allocate memory for buffer
  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(resources.device, resources.buffer,
                                &memRequirements);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;

  // Find memory type index
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(resources.physicalDevice, &memProperties);
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((memRequirements.memoryTypeBits & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
      allocInfo.memoryTypeIndex = i;
      break;
    }
  }

  vkAllocateMemory(resources.device, &allocInfo, nullptr,
                   &resources.bufferMemory);
  vkBindBufferMemory(resources.device, resources.buffer, resources.bufferMemory,
                     0);

  // Initialize buffer data
  float *data;
  vkMapMemory(resources.device, resources.bufferMemory, 0, BUFFER_SIZE, 0,
              (void **)&data);
  for (int i = 0; i < BUFFER_SIZE / sizeof(float); i++) {
    data[i] = static_cast<float>(i);
  }
  vkUnmapMemory(resources.device, resources.bufferMemory);

  // Create descriptor set layout
  VkDescriptorSetLayoutBinding binding = {};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &binding;
  vkCreateDescriptorSetLayout(resources.device, &layoutInfo, nullptr,
                              &resources.descriptorSetLayout);

  // Create descriptor pool
  VkDescriptorPoolSize poolSize = {};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = 1;

  VkDescriptorPoolCreateInfo descPoolInfo = {};
  descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descPoolInfo.maxSets = 1;
  descPoolInfo.poolSizeCount = 1;
  descPoolInfo.pPoolSizes = &poolSize;
  vkCreateDescriptorPool(resources.device, &descPoolInfo, nullptr,
                         &resources.descriptorPool);

  // Allocate descriptor set
  VkDescriptorSetAllocateInfo allocSetInfo = {};
  allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocSetInfo.descriptorPool = resources.descriptorPool;
  allocSetInfo.descriptorSetCount = 1;
  allocSetInfo.pSetLayouts = &resources.descriptorSetLayout;
  vkAllocateDescriptorSets(resources.device, &allocSetInfo,
                           &resources.descriptorSet);

  // Update descriptor set
  VkDescriptorBufferInfo bufferDescInfo = {};
  bufferDescInfo.buffer = resources.buffer;
  bufferDescInfo.offset = 0;
  bufferDescInfo.range = BUFFER_SIZE;

  VkWriteDescriptorSet writeDesc = {};
  writeDesc.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writeDesc.dstSet = resources.descriptorSet;
  writeDesc.dstBinding = 0;
  writeDesc.descriptorCount = 1;
  writeDesc.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writeDesc.pBufferInfo = &bufferDescInfo;
  vkUpdateDescriptorSets(resources.device, 1, &writeDesc, 0, nullptr);

  // Load shader from file
  std::string shaderPath = "compute_complex.spv";
  std::vector<char> shaderCode = readFile(shaderPath);
  resources.computeShaderModule =
      createShaderModule(shaderCode, resources.device);

  {
    std::lock_guard<std::mutex> lock(consoleMutex);
    std::cout << "Loaded shader from: " << shaderPath << " ("
              << shaderCode.size() << " bytes)" << std::endl;
  }

  // Create pipeline layout
  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &resources.descriptorSetLayout;
  vkCreatePipelineLayout(resources.device, &pipelineLayoutInfo, nullptr,
                         &resources.pipelineLayout);

  // Create compute pipeline
  VkComputePipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module = resources.computeShaderModule;
  pipelineInfo.stage.pName = "main";
  pipelineInfo.layout = resources.pipelineLayout;
  vkCreateComputePipelines(resources.device, VK_NULL_HANDLE, 1, &pipelineInfo,
                           nullptr, &resources.computePipeline);
}

void runFrame(VulkanResources &resources, int frameNumber, int threadId) {
  // Create command buffer
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = resources.commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer cmdBuffer;
  vkAllocateCommandBuffers(resources.device, &allocInfo, &cmdBuffer);

  // Record command buffer
  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmdBuffer, &beginInfo);

  // Bind compute pipeline and descriptor sets
  vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    resources.computePipeline);
  vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          resources.pipelineLayout, 0, 1,
                          &resources.descriptorSet, 0, nullptr);

  // Dispatch compute work
  {
    std::lock_guard<std::mutex> lock(consoleMutex);
    std::cout << "Thread " << threadId << ", Frame " << frameNumber
              << ": Dispatching compute work..." << std::endl;
  }

  uint32_t load_size = static_cast<uint32_t>(BASE_SIZE * workloadMultiplier);
  load_size = std::max(load_size, 1u);

  vkCmdDispatch(cmdBuffer, load_size, load_size, 1);

  vkEndCommandBuffer(cmdBuffer);

  // Submit command buffer
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmdBuffer;

  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence;
  vkCreateFence(resources.device, &fenceInfo, nullptr, &fence);

  vkQueueSubmit(resources.computeQueue, 1, &submitInfo, fence);
  vkWaitForFences(resources.device, 1, &fence, VK_TRUE, UINT64_MAX);

  // Cleanup frame resources
  vkDestroyFence(resources.device, fence, nullptr);
  vkFreeCommandBuffers(resources.device, resources.commandPool, 1, &cmdBuffer);

  // Small pause between frames to see the profiler output clearly
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

static void cleanup(VulkanResources &resources) {
  vkDestroyPipeline(resources.device, resources.computePipeline, nullptr);
  vkDestroyPipelineLayout(resources.device, resources.pipelineLayout, nullptr);
  vkDestroyShaderModule(resources.device, resources.computeShaderModule,
                        nullptr);
  vkDestroyDescriptorSetLayout(resources.device, resources.descriptorSetLayout,
                               nullptr);
  vkDestroyDescriptorPool(resources.device, resources.descriptorPool, nullptr);
  vkDestroyBuffer(resources.device, resources.buffer, nullptr);
  vkFreeMemory(resources.device, resources.bufferMemory, nullptr);
  vkDestroyCommandPool(resources.device, resources.commandPool, nullptr);

  vkDestroyDevice(resources.device, nullptr);
}

// Memory polling thread function
void memoryPollingFunction() {
  try {
    // Create IPC client array to connect to GPU info servers
    qvac_lib_abstract_ipc::AbstractIpcClientArray clientArray(
        "vulkan_hooks.gpu_info_server", 4); // Check up to 4 backends

    {
      std::lock_guard<std::mutex> lock(consoleMutex);
      std::cout << "Memory polling thread started. Found " 
                << clientArray.getCount() << " GPU info backends." << std::endl;
    }

    double currentMax = 0.0;
    
    while (!shouldStopMemoryPolling.load()) {
      try {
        // Poll memory usage from all available backends
        // Request type 86 is MEMORY_USAGE for device 0
        double memoryUsage = clientArray.sum(MEMORY_USAGE_REQUEST);
        
        if (memoryUsage > currentMax) {
          currentMax = memoryUsage;
          maxMemoryUsage.store(currentMax);
          
          {
            std::lock_guard<std::mutex> lock(consoleMutex);
            std::cout << "New max memory usage detected: " 
                      << memoryUsage << " bytes" << std::endl;
          }
        }
        
      } catch (const std::exception& e) {
        // If all backends fail, continue polling
        std::this_thread::sleep_for(std::chrono::milliseconds(MEMORY_POLL_INTERVAL_MS));
        continue;
      }
      
      std::this_thread::sleep_for(std::chrono::milliseconds(MEMORY_POLL_INTERVAL_MS));
    }
    
    {
      std::lock_guard<std::mutex> lock(consoleMutex);
      std::cout << "Memory polling thread stopped. Final max memory usage: " 
                << maxMemoryUsage.load() << " bytes" << std::endl;
    }
    
  } catch (const std::exception& e) {
    std::lock_guard<std::mutex> lock(consoleMutex);
    std::cerr << "Memory polling thread error: " << e.what() << std::endl;
  }
}

void threadFunction(int threadId) {
  try {
    VulkanResources resources = {};

    setupDevice(resources);
    setupComputeResources(resources);

    {
      std::lock_guard<std::mutex> lock(consoleMutex);
      std::cout << "Thread " << threadId << " running compute example for "
                << NUM_FRAMES << " frames..." << std::endl;
    }

    // Main loop - run multiple frames
    for (int i = 0; i < NUM_FRAMES; i++) {
      runFrame(resources, i, threadId);
    }

#ifdef USE_VULKAN_WRAPPERS
    if (threadId == 0) {
      // double utilization = vkUtilization(0);
      // std::cout << "Final GPU utilization: " << utilization << std::endl;
    }
#endif

    cleanup(resources);

  } catch (const std::exception &e) {
    std::lock_guard<std::mutex> lock(consoleMutex);
    std::cerr << "Thread " << threadId << " error: " << e.what() << std::endl;
  }
}

void printUsage(const char *programName) {
  std::cout << "Usage: " << programName << " [multiplier] [threads]"
            << std::endl;
  std::cout << "  multiplier - Multiplier for all workload sizes (default: 1.0)"
            << std::endl;
  std::cout << "  threads    - Number of threads to create (default: 1)"
            << std::endl;
}

int main(int argc, char **argv) {
  auto startTime = std::chrono::high_resolution_clock::now();

  try {
    // Set default workload multiplier
    workloadMultiplier = 1.0f;
    numThreads = 1;

    // Check if help was requested
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
      }
    }

    // Parse workload multiplier
    if (argc > 1) {
      try {
        workloadMultiplier = std::stof(argv[1]);
        if (workloadMultiplier <= 0.0f) {
          std::cerr << "Workload multiplier must be positive" << std::endl;
          return EXIT_FAILURE;
        }
      } catch (const std::exception &e) {
        std::cerr << "Error parsing workload multiplier: " << e.what()
                  << std::endl;
        printUsage(argv[0]);
        return EXIT_FAILURE;
      }
    }

    // Parse number of threads
    if (argc > 2) {
      try {
        numThreads = std::stoi(argv[2]);
        if (numThreads <= 0) {
          std::cerr << "Number of threads must be positive" << std::endl;
          return EXIT_FAILURE;
        }
      } catch (const std::exception &e) {
        std::cerr << "Error parsing number of threads: " << e.what()
                  << std::endl;
        printUsage(argv[0]);
        return EXIT_FAILURE;
      }
    }

    // Print the current settings
    std::cout << "Workload multiplier: " << workloadMultiplier << std::endl;
    std::cout << "Number of threads: " << numThreads << std::endl;


    // Create Vulkan instance (shared among all threads)
    createInstance();

    // Start memory polling thread
    shouldStopMemoryPolling.store(false);
    memoryPollingThread = std::thread(memoryPollingFunction);

    // Create and start threads
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; i++) {
      threads.emplace_back(threadFunction, i);
    }

    // Wait for all threads to complete
    for (auto &thread : threads) {
      thread.join();
    }

    // Stop memory polling and wait for it to finish
    shouldStopMemoryPolling.store(true);
    if (memoryPollingThread.joinable()) {
      memoryPollingThread.join();
    }

    // Clean up shared resources
    vkDestroyInstance(instance, nullptr);

    // Print final memory statistics
    std::cout << "\n=== Final Memory Statistics ===" << std::endl;
    std::cout << "Maximum memory usage detected: " 
              << maxMemoryUsage.load() << " bytes" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    
    // Ensure memory polling thread is stopped
    shouldStopMemoryPolling.store(true);
    if (memoryPollingThread.joinable()) {
      memoryPollingThread.join();
    }
    
    return EXIT_FAILURE;
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime)
          .count();
  std::cout << "\nTotal execution time: " << duration << " milliseconds"
            << std::endl;

  return EXIT_SUCCESS;
}
