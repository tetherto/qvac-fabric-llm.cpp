# GPU Utilization Tracker

This project demonstrates GPU utilization tracking through a Vulkan compute example that runs a shader workload. The simple compute example (`simple_compute_example.cpp`) creates a multi-threaded Vulkan application that dispatches compute shaders.

## Overview

The application has been linked to a **patched Vulkan library** that intercepts Vulkan API calls to enable transparent GPU resource monitoring. This patching system works through function interposition, allowing real-time tracking of GPU memory usage and device utilization without modifying the application code.

## Architecture

### Build System Integration

The patching is controlled by the `USE_VULKAN_WRAPPERS` CMake option:

```cmake
if(USE_VULKAN_WRAPPERS)
  # Link hooks BEFORE the Vulkan library for proper interposition
  target_link_libraries(gpu_utilization_demo PRIVATE vulkan_hooks
                                                     abstract_ipc_cpp
                                                     ${Vulkan_LIBRARIES})
endif()
```

**Critical**: The `vulkan_hooks` library must be linked before the actual Vulkan library to ensure proper function interposition.

## Compiling

```bash
cmake -S . -B build && cmake --build build
```

### Build Options

- `USE_VULKAN_WRAPPERS=ON` (default): Enable Vulkan function patching
- `ENABLE_VULKAN_PERFORMANCE_LOG_TRACE=ON`: Enable performance debug logging
- `ENABLE_VULKAN_LOG_ALLOCATIONS=ON`: Enable memory allocation logging

## Running the Example

```bash
./build/gpu_utilization_demo [workload_multiplier] [num_threads]
```

### Example Output

```bash
Running: ./build/gpu_utilization_demo
Workload multiplier: 1
Number of threads: 1
Memory polling thread started. Found 1 GPU info backends.
Could not find memory for physical device at index 0 (tracked physical device count is: 0)
Loaded shader from: compute_complex.spv (11460 bytes)
Thread 0 running compute example for 5 frames...
Thread 0, Frame 0: Dispatching compute work...
New max memory usage detected: 262144 bytes
Thread 0, Frame 1: Dispatching compute work...
Thread 0, Frame 2: Dispatching compute work...
Thread 0, Frame 3: Dispatching compute work...
Thread 0, Frame 4: Dispatching compute work...
Memory polling thread stopped. Final max memory usage: 262144 bytes

=== Final Memory Statistics ===
Maximum memory usage detected: 262144 bytes

Total execution time: 637 milliseconds
```

## Usage in Your Application

To use the patched Vulkan functions in your own application:

```cpp
#ifdef USE_VULKAN_WRAPPERS
#include "vulkan_hooks/include/vk_patch.h"

// Get memory usage
uint64_t used_memory = vkUsedMemory(device_index);
uint64_t total_memory = vkTotalMemory(device_index);
#endif
```

Or use the IPC to poll statistics:
```c++
// Example: Query maximum GPU memory usage via IPC (as in simple_compute_example.cpp)

#include "qvac_lib_abstract_ipc/include/AbstractIpcClientArray.hpp"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    using namespace qvac_lib_abstract_ipc;

    // MEMORY_USAGE request type for device 0, as in simple_compute_example.cpp
    constexpr int MEMORY_USAGE_REQUEST = 86;

    // Create an IPC client array for the GPU info server (up to 4 backends)
    AbstractIpcClientArray clientArray("vulkan_hooks.gpu_info_server", 4);

    std::cout << "Polling maximum used memory from all available GPU info servers..." << std::endl;

    double maxMemoryUsage = 0.0;

    // Poll a few times to demonstrate (in the real app, this is done in a thread loop)
    for (int i = 0; i < 5; ++i) {
        try {
            // Query the maximum used memory across all servers
            double memoryUsage = clientArray.max(MEMORY_USAGE_REQUEST);

            if (memoryUsage > maxMemoryUsage) {
                maxMemoryUsage = memoryUsage;
                std::cout << "New max memory usage detected: " 
                          << memoryUsage << " bytes" << std::endl;
            } else {
                std::cout << "Current memory usage: " 
                          << memoryUsage << " bytes" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error polling memory usage: " << e.what() << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\n=== Final Memory Statistics ===" << std::endl;
    std::cout << "Maximum memory usage detected: " << maxMemoryUsage << " bytes" << std::endl;

    return 0;
}
```
