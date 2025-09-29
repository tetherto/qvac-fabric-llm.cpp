#include "AbstractIpcClient.hpp"
#include <iostream>

namespace qvac_lib_abstract_ipc {

AbstractIpcClient::AbstractIpcClient(const std::string& baseName, int socketIndex)
    : baseName_(baseName), socketIndex_(socketIndex) {
    if (socketIndex < 0 || socketIndex >= 64) {
        throw std::invalid_argument("socketIndex must be between 0 and 63");
    }
}

double AbstractIpcClient::sendRequest(int request) {
    if (request < 0 || request > 255) {
        throw std::invalid_argument("request must be between 0 and 255");
    }

    // Connect to the server
    AbstractIPCClientSocket_t socket;
    bool connected = abstract_ipc_client_connect(&socket, baseName_.c_str(), socketIndex_);
    
    if (!connected) {
        throw std::runtime_error("Failed to connect to server: " + baseName_ + ":" + std::to_string(socketIndex_));
    }

    try {
        // Send the request and get the response
        double response;
        bool success = abstract_ipc_client_send_request(&socket, static_cast<unsigned char>(request), &response);
        
        if (!success) {
            throw std::runtime_error("Failed to send request to server");
        }
        
        return response;
    } catch (...) {
        // Always close the connection when done
        abstract_ipc_client_close(&socket);
        throw;
    }
    
    // Always close the connection when done
    abstract_ipc_client_close(&socket);
}

} // namespace qvac_lib_abstract_ipc
