#pragma once

#include "AbstractIpcClient.hpp"
#include <vector>
#include <memory>
#include <stdexcept>

namespace qvac_lib_abstract_ipc {

/**
 * Response structure for array operations
 */
struct IpcResponse {
    int index;
    double response;
    std::string error;
    bool success;
    
    IpcResponse(int idx, double resp) : index(idx), response(resp), success(true) {}
    IpcResponse(int idx, const std::string& err) : index(idx), response(0.0), error(err), success(false) {}
};

/**
 * AbstractIpcClientArray provides a way to manage and query multiple AbstractIpcClient instances.
 * It allows sending requests to multiple backends and aggregating their responses.
 */
class AbstractIpcClientArray {
public:
    /**
     * Create a new AbstractIpcClientArray.
     *
     * @param baseName The base name of the server channel to connect to
     * @param maxSockets The maximum number of sockets to check (default: 64)
     * @throws std::invalid_argument if parameters are invalid
     */
    AbstractIpcClientArray(const std::string& baseName, int maxSockets = 64);

    /**
     * Destructor
     */
    ~AbstractIpcClientArray() = default;

    /**
     * Get the number of available backends.
     *
     * @returns The number of available backends
     */
    size_t getCount() const { return clients_.size(); }

    /**
     * Send a request to all available backends and get an array of responses.
     *
     * @param request The request code to send (0-255)
     * @returns Vector of responses with socket indices
     * @throws std::invalid_argument if request is out of range
     */
    std::vector<IpcResponse> sendRequest(int request);

    /**
     * Calculate the sum of all successful responses for a given request.
     *
     * @param request The request code to send
     * @returns The sum of all successful responses
     * @throws std::runtime_error If all backends failed to respond
     */
    double sum(int request);

    /**
     * Get the first successful response from any backend.
     *
     * @param request The request code to send
     * @returns The first successful response
     * @throws std::runtime_error If all backends failed to respond
     */
    double any(int request);

    /**
     * Get the maximum value from successful responses of all backends.
     *
     * @param request The request code to send
     * @returns The maximum value from all successful responses
     * @throws std::runtime_error If all backends failed to respond
     */
    double max(int request);

    /**
     * Get all successful responses for a given request.
     *
     * @param request The request code to send
     * @returns Vector of successful responses with socket indices
     * @throws std::runtime_error If all backends failed to respond
     */
    std::vector<IpcResponse> all(int request);

    /**
     * Get the average of all successful responses for a given request.
     *
     * @param request The request code to send
     * @returns The average of all successful responses
     * @throws std::runtime_error If all backends failed to respond
     */
    double average(int request);

private:
    std::string baseName_;
    int maxSockets_;
    std::vector<std::unique_ptr<AbstractIpcClient>> clients_;
    
    /**
     * Discover available backends by checking health status of each potential socket.
     */
    void discoverBackends();
};

} // namespace qvac_lib_abstract_ipc
