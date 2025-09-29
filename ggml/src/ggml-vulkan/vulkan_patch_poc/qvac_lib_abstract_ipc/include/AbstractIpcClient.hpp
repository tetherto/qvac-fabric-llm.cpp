#pragma once

#include <string>
#include <stdexcept>

extern "C" {
#include "client.h"
}

namespace qvac_lib_abstract_ipc {

/**
 * AbstractIpcClient provides a C++ interface to the abstract IPC client library.
 * It allows connecting to an abstract IPC server, sending requests, and receiving responses.
 *
 * It allows to query numeric values from the other processes using a string channel identifier.
 */
class AbstractIpcClient {
public:
    /**
     * Create a new AbstractIpcClient.
     *
     * @param baseName The base name of the server channel to connect to
     * @param socketIndex The socket index to connect to (0-63)
     * @throws std::invalid_argument if parameters are invalid
     */
    AbstractIpcClient(const std::string& baseName, int socketIndex);

    /**
     * Destructor - ensures proper cleanup
     */
    ~AbstractIpcClient() = default;

    /**
     * Send a request to the server and get the response.
     * This method handles connection, request sending, and disconnection in one step.
     *
     * @param request The request code to send (0-255)
     * @returns The response from the server
     * @throws std::invalid_argument if request is out of range
     * @throws std::runtime_error if connection or sending fails
     */
    double sendRequest(int request);

    /**
     * Get the base name used for this client.
     *
     * @returns The base name
     */
    const std::string& getBaseName() const { return baseName_; }

    /**
     * Get the socket index used for this client.
     *
     * @returns The socket index
     */
    int getSocketIndex() const { return socketIndex_; }

private:
    std::string baseName_;
    int socketIndex_;
};

} // namespace qvac_lib_abstract_ipc
