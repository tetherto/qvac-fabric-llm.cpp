// Starts a managed RPC server on port 0 through the backend registry, the way
// in-process hosts do, and checks that it reports a port a client can reach.

#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <cstdint>
#include <cstdio>
#include <thread>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <winsock2.h>
#    include <ws2tcpip.h>
typedef SOCKET sockfd_t;
static bool is_valid(sockfd_t fd) { return fd != INVALID_SOCKET; }
static void close_fd(sockfd_t fd) { closesocket(fd); }
#else
#    include <arpa/inet.h>
#    include <netinet/in.h>
#    include <sys/socket.h>
#    include <unistd.h>
typedef int sockfd_t;
static bool is_valid(sockfd_t fd) { return fd >= 0; }
static void close_fd(sockfd_t fd) { close(fd); }
#endif

static bool can_connect(int port) {
    sockfd_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (!is_valid(fd)) {
        return false;
    }
    struct sockaddr_in addr = {};
    addr.sin_family         = AF_INET;
    addr.sin_port           = htons((uint16_t) port);
    addr.sin_addr.s_addr    = htonl(INADDR_LOOPBACK);
    const bool ok = connect(fd, (struct sockaddr *) &addr, sizeof(addr)) == 0;
    close_fd(fd);
    return ok;
}

template <typename T>
static T get_proc(ggml_backend_reg_t reg, const char * name) {
    T fn = (T) ggml_backend_reg_get_proc_address(reg, name);
    if (fn == nullptr) {
        fprintf(stderr, "RPC backend does not export %s\n", name);
    }
    return fn;
}

int main() {
#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        return 1;
    }
#endif
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (reg == nullptr || cpu == nullptr) {
        fprintf(stderr, "RPC backend or CPU device not available, skipping\n");
        return 77;
    }

    auto server_create   = get_proc<decltype(&ggml_backend_rpc_server_create)>(reg, "ggml_backend_rpc_server_create");
    auto server_get_port = get_proc<decltype(&ggml_backend_rpc_server_get_port)>(reg, "ggml_backend_rpc_server_get_port");
    auto server_run      = get_proc<decltype(&ggml_backend_rpc_server_run)>(reg, "ggml_backend_rpc_server_run");
    auto server_stop     = get_proc<decltype(&ggml_backend_rpc_server_stop)>(reg, "ggml_backend_rpc_server_stop");
    auto server_free     = get_proc<decltype(&ggml_backend_rpc_server_free)>(reg, "ggml_backend_rpc_server_free");
    auto rdma_supported  = get_proc<decltype(&ggml_backend_rpc_rdma_supported)>(reg, "ggml_backend_rpc_rdma_supported");
    if (!server_create || !server_get_port || !server_run || !server_stop || !server_free || !rdma_supported) {
        return 1;
    }
    printf("RDMA supported: %s\n", rdma_supported() ? "yes" : "no");

    if (server_get_port(nullptr) != -1) {
        fprintf(stderr, "get_port(NULL) did not return -1\n");
        return 1;
    }

    ggml_backend_rpc_server_t server = server_create("127.0.0.1:0", nullptr, 1, 1, &cpu);
    if (server == nullptr) {
        fprintf(stderr, "failed to create an RPC server on port 0\n");
        return 1;
    }
    const int port = server_get_port(server);

    std::thread runner([&] { server_run(server); });
    const bool connected = port > 0 && can_connect(port);
    server_stop(server);
    runner.join();
    server_free(server);

#ifdef _WIN32
    WSACleanup();
#endif
    if (!connected) {
        fprintf(stderr, "server on port 0 reported port %d, which is not reachable\n", port);
        return 1;
    }
    return 0;
}
