#include "AbstractIpcClientArray.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>

namespace qvac_lib_abstract_ipc {

AbstractIpcClientArray::AbstractIpcClientArray(const std::string& baseName, int maxSockets)
    : baseName_(baseName), maxSockets_(maxSockets) {
    if (maxSockets <= 0 || maxSockets > 64) {
        throw std::invalid_argument("maxSockets must be between 1 and 64");
    }
    
    // Discover available backends
    discoverBackends();
}

void AbstractIpcClientArray::discoverBackends() {
    const int HEALTH_CHECK_REQUEST = 0; // Assume type 0 is for health check
    
    for (int i = 0; i < maxSockets_; i++) {
        try {
            auto client = std::make_unique<AbstractIpcClient>(baseName_, i);
            // Try to connect and check health
            double result = client->sendRequest(HEALTH_CHECK_REQUEST);
            // If we reach here, the client is responsive
            if (result > 0.5) {
                clients_.push_back(std::move(client));
            } else {
                std::cerr << "Backend " << i << " is not responsive. Returned unhealthy value: " << result << std::endl;
            }
        } catch (const std::exception& e) {
            // Skip this socket if it's not available
            continue;
        }
    }
}

std::vector<IpcResponse> AbstractIpcClientArray::sendRequest(int request) {
    if (request < 0 || request > 255) {
        throw std::invalid_argument("request must be between 0 and 255");
    }
    
    std::vector<IpcResponse> responses;
    responses.reserve(clients_.size());
    
    for (auto& client : clients_) {
        try {
            double response = client->sendRequest(request);
            responses.emplace_back(client->getSocketIndex(), response);
        } catch (const std::exception& e) {
            responses.emplace_back(client->getSocketIndex(), e.what());
        }
    }
    
    return responses;
}

double AbstractIpcClientArray::sum(int request) {
    auto results = sendRequest(request);
    auto successfulResults = std::vector<IpcResponse>();
    
    for (const auto& result : results) {
        if (result.success) {
            successfulResults.push_back(result);
        }
    }
    
    if (successfulResults.empty()) {
        throw std::runtime_error("All backends failed to respond");
    }
    
    return std::accumulate(successfulResults.begin(), successfulResults.end(), 0.0,
                          [](double sum, const IpcResponse& result) {
                              return sum + result.response;
                          });
}

double AbstractIpcClientArray::any(int request) {
    auto results = sendRequest(request);
    
    for (const auto& result : results) {
        if (result.success) {
            return result.response;
        }
    }
    
    throw std::runtime_error("All backends failed to respond");
}

double AbstractIpcClientArray::max(int request) {
    auto results = sendRequest(request);
    auto successfulResults = std::vector<IpcResponse>();
    
    for (const auto& result : results) {
        if (result.success) {
            successfulResults.push_back(result);
        }
    }
    
    if (successfulResults.empty()) {
        throw std::runtime_error("All backends failed to respond");
    }
    
    return std::max_element(successfulResults.begin(), successfulResults.end(),
                           [](const IpcResponse& a, const IpcResponse& b) {
                               return a.response < b.response;
                           })->response;
}

std::vector<IpcResponse> AbstractIpcClientArray::all(int request) {
    auto results = sendRequest(request);
    std::vector<IpcResponse> successfulResults;
    
    for (const auto& result : results) {
        if (result.success) {
            successfulResults.push_back(result);
        }
    }
    
    if (successfulResults.empty()) {
        throw std::runtime_error("All backends failed to respond");
    }
    
    return successfulResults;
}

double AbstractIpcClientArray::average(int request) {
    auto results = all(request);
    if (results.empty()) {
        throw std::runtime_error("All backends failed to respond");
    }
    
    double sum = std::accumulate(results.begin(), results.end(), 0.0,
                                [](double acc, const IpcResponse& result) {
                                    return acc + result.response;
                                });
    
    return sum / results.size();
}

} // namespace qvac_lib_abstract_ipc
