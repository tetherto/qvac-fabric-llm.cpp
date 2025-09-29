#pragma once

#include "gpu_device_tracker.h"
#include "gpu_memory_tracker.h"
#include <thread>

extern "C" {
#ifdef ANDROID
#include "abstract_ipc_server_socket_unix.h"
#include "server_unix.h"
#else
#include "server.h"
#endif
}

class GpuInfoServer {
public:
  constexpr static char endpointName[] = "vulkan_hooks.gpu_info_server";
  enum RequestType : unsigned char {
    HEALTH_CHECK = 0,
    DEVICE_COUNT = 1,
    UTILIZATION = 2,
    MEMORY_USAGE = 86,
    TOTAL_AVAILABLE_MEMORY = 168
  };

  GpuInfoServer(GPUDeviceTracker &deviceTracker,
                GPUMemoryTracker &memoryTracker);

  ~GpuInfoServer();

  bool start();
  void stop();
  bool isRunning() const;

private:
  double handleRequest(RequestType request);

  // Add a static trampoline function
  static double handleRequestTrampoline(unsigned char request);

  // Add a static pointer to the current instance
  static GpuInfoServer *currentInstance;

  GPUDeviceTracker &_deviceTracker;
  GPUMemoryTracker &_memoryTracker;

  std::thread _serverThread;
  AbstractIPCServerSocket_t _serverHandle;
  bool _isRunning;
  void requestHandlerLoop();
};
