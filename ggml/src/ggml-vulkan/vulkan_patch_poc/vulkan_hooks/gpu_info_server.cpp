#include "gpu_info_server.hpp"
#include "server_unix.h"
#include <cstring>
#include <iostream>

#ifdef ANDROID
#include <android/log.h>
#endif

// Initialize the static member
GpuInfoServer *GpuInfoServer::currentInstance = nullptr;

// Static trampoline function
double GpuInfoServer::handleRequestTrampoline(unsigned char request) {
  if (currentInstance) {
    return currentInstance->handleRequest(static_cast<RequestType>(request));
  }
  return 0.0;
}

GpuInfoServer::GpuInfoServer(GPUDeviceTracker &deviceTracker,
                             GPUMemoryTracker &memoryTracker)
    : _deviceTracker(deviceTracker), _memoryTracker(memoryTracker),
      _serverHandle(), _isRunning(false) {
  currentInstance = this; // Set the current instance
}

GpuInfoServer::~GpuInfoServer() {
  if (_isRunning) {
    stop();
  }
}

bool GpuInfoServer::start() {
  if (_isRunning) {
    return true;
  }

  if (!(abstract_ipc_server_create_socket(&_serverHandle, endpointName))) {
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_ERROR, "GpuInfoServer",
                        "Failed to create GPU info server at endpoint: %s",
                        endpointName);
#else
    std::cerr << "Failed to create GPU info server at endpoint: "
              << endpointName << std::endl;
#endif
    return false;
  }

  // Create thread to handle requests in a loop
  _isRunning = true;
  _serverThread = std::thread(&GpuInfoServer::requestHandlerLoop, this);

#ifdef VULKAN_GPU_INFO_SERVER_TRACE
#ifdef ANDROID
  __android_log_print(ANDROID_LOG_INFO, "GpuInfoServer",
                      "gpu_info_server: started.");
#else
  std::cout << "gpu_info_server: started." << std::endl;
#endif
#endif

  return true;
}

void GpuInfoServer::stop() {
  if (!_isRunning) {
    return;
  }

  // Signal thread to exit
  _isRunning = false;
  
  // Close the socket to interrupt the blocking accept() call
  abstract_ipc_server_close(&_serverHandle);
  
  // Detach the thread - it will exit when accept() fails
  if (_serverThread.joinable()) {
    _serverThread.detach();
  }
}

bool GpuInfoServer::isRunning() const { return _isRunning; }

double GpuInfoServer::handleRequest(RequestType request) {
  if (request == GpuInfoServer::RequestType::HEALTH_CHECK) {
#ifdef VULKAN_GPU_INFO_SERVER_TRACE
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO, "GpuInfoServer",
                        "Requested health status.");
#else
    std::cout << "gpu_info_server: Requested health status. " << std::endl;
#endif
#endif
    return 1.0;
  } else if (request == GpuInfoServer::RequestType::DEVICE_COUNT) {
#ifdef VULKAN_GPU_INFO_SERVER_TRACE
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO, "GpuInfoServer",
                        "Requested device count.");
#else
    std::cout << "gpu_info_server: Requested device count. " << std::endl;
#endif
#endif
    return _deviceTracker.getPhysicalDeviceCount();
  } else if (request >= GpuInfoServer::RequestType::UTILIZATION &&
             request < GpuInfoServer::RequestType::MEMORY_USAGE) {
    const char device = request - GpuInfoServer::RequestType::UTILIZATION;
#ifdef VULKAN_GPU_INFO_SERVER_TRACE
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO, "GpuInfoServer",
                        "Requested utilization for device: %u",
                        (unsigned)device);
    __android_log_print(ANDROID_LOG_INFO, "GpuInfoServer",
                        "Utilization unimplemented")
#else
    std::cout << "gpu_info_server: Requested utilization for device: "
              << (unsigned)device << std::endl;
    std::cout << "gpu_info_server: Utilization unimplemented" << std::endl;
#endif
#endif
        return 0.0;
  } else if (request >= GpuInfoServer::RequestType::MEMORY_USAGE &&
             request < GpuInfoServer::RequestType::TOTAL_AVAILABLE_MEMORY) {
    const char device = request - GpuInfoServer::RequestType::MEMORY_USAGE;
#ifdef VULKAN_GPU_INFO_SERVER_TRACE
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO, "GpuInfoServer",
                        "Requested memory for device: %u", (unsigned)device);
#else
    std::cout << "gpu_info_server: Requested memory for device: "
              << (unsigned)device << std::endl;
#endif
#endif
    return _memoryTracker.getUsedMemory(device);
  } else if (request >= GpuInfoServer::RequestType::TOTAL_AVAILABLE_MEMORY) {
    const char device =
        request - GpuInfoServer::RequestType::TOTAL_AVAILABLE_MEMORY;
#ifdef VULKAN_GPU_INFO_SERVER_TRACE
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO, "GpuInfoServer",
                        "Requested total memory for device: %u",
                        (unsigned)device);
#else
    std::cout << "gpu_info_server: Requested total memory for device: "
              << (unsigned)device << std::endl;
#endif
#endif
    return _memoryTracker.getTotalMemory(device);
  }

#ifdef VULKAN_GPU_INFO_SERVER_TRACE
#ifdef ANDROID
  __android_log_print(ANDROID_LOG_WARN, "GpuInfoServer", "Unknown request: %d",
                      request);
#else
  std::cout << "gpu_info_server: Unknown request: " << request << std::endl;
#endif
#endif

  return -2.0;
}

void GpuInfoServer::requestHandlerLoop() {
  while (_isRunning) {
    abstract_ipc_server_handle_request(&_serverHandle, handleRequestTrampoline);
  }
}
