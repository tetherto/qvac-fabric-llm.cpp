#include "transport.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

#if defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
#    include <dlfcn.h>
#endif

#ifndef _WIN32
#    include <arpa/inet.h>
#    include <fcntl.h>
#    include <pthread.h>
#    include <sys/select.h>
#    include <sys/socket.h>
#    include <sys/wait.h>
#    include <unistd.h>

#    include <cerrno>
#    include <chrono>
#    include <csignal>
#    include <thread>

static int run_shutdown_send_child() {
    signal(SIGPIPE, SIG_DFL);

    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) {
        return 1;
    }

    sockaddr_in address     = {};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    if (bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
        perror("bind");
        close(listener);
        return 2;
    }
    if (listen(listener, 1) != 0) {
        perror("listen");
        close(listener);
        return 3;
    }

    socklen_t address_size = sizeof(address);
    if (getsockname(listener, reinterpret_cast<sockaddr *>(&address), &address_size) != 0) {
        perror("getsockname");
        close(listener);
        return 4;
    }

    socket_ptr client = socket_t::connect("127.0.0.1", ntohs(address.sin_port));
    if (client == nullptr) {
        close(listener);
        return 5;
    }

    const int peer = accept(listener, nullptr, nullptr);
    close(listener);
    if (peer < 0) {
        return 6;
    }

    client->shutdown();
    const char byte = 0;
    const bool sent = client->send_data(&byte, sizeof(byte));
    close(peer);
    return sent ? 7 : 0;
}

static bool test_shutdown_send() {
    const pid_t child = fork();
    if (child < 0) {
        perror("fork");
        return false;
    }
    if (child == 0) {
        _exit(run_shutdown_send_child());
    }

    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) {
            perror("waitpid");
            return false;
        }
    }
    if (WIFSIGNALED(status)) {
        fprintf(stderr, "send after shutdown terminated with signal %d\n", WTERMSIG(status));
        return false;
    }
    if (!WIFEXITED(status)) {
        fprintf(stderr, "send after shutdown did not exit normally\n");
        return false;
    }
    const int exit_code = WEXITSTATUS(status);
    if (exit_code != 0) {
        fprintf(stderr, "send after shutdown failed with exit code %d\n", exit_code);
        return false;
    }
    return true;
}

static bool test_accept_timeout() {
    socket_ptr server = socket_t::create_server("127.0.0.1", 0);
    if (server == nullptr) {
        fprintf(stderr, "failed to create server socket\n");
        return false;
    }

    bool       timed_out = false;
    socket_ptr client    = server->accept(1, &timed_out);
    if (client != nullptr || !timed_out) {
        fprintf(stderr, "timed accept did not report timeout\n");
        return false;
    }
    return true;
}

static void handle_signal(int) {}

static bool test_interrupted_accept_timeout() {
    socket_ptr server = socket_t::create_server("127.0.0.1", 0);
    if (server == nullptr) {
        return false;
    }

    struct sigaction action   = {};
    struct sigaction previous = {};
    action.sa_handler         = handle_signal;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGUSR1, &action, &previous) != 0) {
        return false;
    }

    int             signal_result = -1;
    const pthread_t accept_thread = pthread_self();
    std::thread     interrupter([accept_thread, &signal_result] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        signal_result = pthread_kill(accept_thread, SIGUSR1);
    });
    bool            timed_out = false;
    socket_ptr      client    = server->accept(100, &timed_out);
    interrupter.join();
    sigaction(SIGUSR1, &previous, nullptr);
    if (signal_result != 0 || client != nullptr || !timed_out) {
        fprintf(stderr, "interrupted timed accept did not report timeout\n");
        return false;
    }
    return true;
}

static bool test_high_fd_accept_timeout() {
    int    fillers[FD_SETSIZE] = {};
    size_t n_fillers           = 0;
    int    high_fd             = -1;
    while (n_fillers < FD_SETSIZE) {
        const int fd = open("/dev/null", O_RDONLY);
        if (fd < 0) {
            break;
        }
        if (fd >= FD_SETSIZE) {
            high_fd = fd;
            break;
        }
        fillers[n_fillers++] = fd;
    }

    if (high_fd < 0) {
        for (size_t i = 0; i < n_fillers; ++i) {
            close(fillers[i]);
        }
        return true;
    }
    close(high_fd);

    socket_ptr server    = socket_t::create_server("127.0.0.1", 0);
    bool       timed_out = false;
    socket_ptr client    = server == nullptr ? nullptr : server->accept(1, &timed_out);
    for (size_t i = 0; i < n_fillers; ++i) {
        close(fillers[i]);
    }
    if (server == nullptr || client != nullptr || !timed_out) {
        fprintf(stderr, "high descriptor timed accept failed\n");
        return false;
    }
    return true;
}

// open() returns the lowest free descriptor, so a leaked socket changes it
static int lowest_free_fd() {
    const int fd = open("/dev/null", O_RDONLY);
    if (fd >= 0) {
        close(fd);
    }
    return fd;
}

static bool test_failed_server_closes_socket() {
    socket_ptr server = socket_t::create_server("127.0.0.1", 0);
    if (server == nullptr) {
        fprintf(stderr, "failed to create server socket\n");
        return false;
    }

    const int  before   = lowest_free_fd();
    socket_ptr in_use   = socket_t::create_server("127.0.0.1", server->local_port());
    socket_ptr bad_host = socket_t::create_server("not-an-address", 0);
    const int  after    = lowest_free_fd();
    if (in_use != nullptr || bad_host != nullptr) {
        fprintf(stderr, "server creation unexpectedly succeeded\n");
        return false;
    }
    if (before < 0 || after != before) {
        fprintf(stderr, "failed server creation leaked a socket (fd %d -> %d)\n", before, after);
        return false;
    }
    return true;
}
#else
static bool test_transport_ref_count() {
    if (!rpc_transport_init()) {
        return false;
    }
    if (!rpc_transport_init()) {
        rpc_transport_shutdown();
        return false;
    }

    rpc_transport_shutdown();
    socket_ptr server  = socket_t::create_server("127.0.0.1", 0);
    const bool created = server != nullptr;
    server.reset();
    rpc_transport_shutdown();
    return created;
}
#endif

static bool test_ephemeral_port() {
    socket_ptr server = socket_t::create_server("127.0.0.1", 0);
    socket_ptr other  = socket_t::create_server("127.0.0.1", 0);
    const int  port   = server == nullptr ? -1 : server->local_port();
    if (port <= 0 || other == nullptr || other->local_port() <= 0 || other->local_port() == port) {
        fprintf(stderr, "port 0 servers did not get distinct ports\n");
        return false;
    }

    socket_ptr client = socket_t::connect("127.0.0.1", port, 1000);
    socket_ptr peer   = server->accept(1000);
    if (client == nullptr || peer == nullptr) {
        fprintf(stderr, "could not connect to the ephemeral port %d\n", port);
        return false;
    }
    return true;
}

// Same order as the HELLO handshake in ggml-rpc.cpp. Without RDMA both sides
// advertise zero caps and keep using TCP.
static bool test_caps_handshake() {
    socket_ptr server = socket_t::create_server("127.0.0.1", 0);
    if (server == nullptr) {
        fprintf(stderr, "failed to create server socket\n");
        return false;
    }

    bool        server_ok = false;
    std::thread server_thread([&server, &server_ok] {
        socket_ptr peer = server->accept(5000);
        if (peer == nullptr) {
            return;
        }
        uint8_t  client_caps[RPC_CONN_CAPS_SIZE] = {};
        uint8_t  server_caps[RPC_CONN_CAPS_SIZE] = {};
        uint32_t value                           = 0;
        if (!peer->recv_data(client_caps, sizeof(client_caps))) {
            return;
        }
        peer->get_caps(server_caps);
        if (!peer->send_data(server_caps, sizeof(server_caps)) || !peer->flush()) {
            return;
        }
        peer->update_caps(client_caps);
        if (!peer->recv_data(&value, sizeof(value))) {
            return;
        }
        value++;
        server_ok = peer->send_data(&value, sizeof(value)) && peer->flush();
    });

    bool       client_ok = false;
    socket_ptr client    = socket_t::connect("127.0.0.1", server->local_port(), 5000);
    if (client != nullptr) {
        uint8_t  client_caps[RPC_CONN_CAPS_SIZE] = {};
        uint8_t  server_caps[RPC_CONN_CAPS_SIZE] = {};
        uint32_t value                           = 41;
        client->get_caps(client_caps);
        if (client->send_data(client_caps, sizeof(client_caps)) && client->flush() &&
            client->recv_data(server_caps, sizeof(server_caps))) {
            client->update_caps(server_caps);
            client_ok = client->send_data(&value, sizeof(value)) && client->flush() &&
                        client->recv_data(&value, sizeof(value)) && value == 42;
        }
    }
    server_thread.join();
    if (!client_ok || !server_ok) {
        fprintf(stderr, "caps handshake or data exchange failed\n");
        return false;
    }
    return true;
}

static void set_no_rdma(bool disabled) {
#ifdef _WIN32
    _putenv_s("GGML_RPC_NO_RDMA", disabled ? "1" : "");
#else
    if (disabled) {
        setenv("GGML_RPC_NO_RDMA", "1", 1);
    } else {
        unsetenv("GGML_RPC_NO_RDMA");
    }
#endif
}

static bool test_rdma_available() {
#if defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
    // RDMA is tried only when libibverbs can be loaded
    void *     lib      = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
    const bool expected = lib != nullptr;
    if (lib != nullptr) {
        dlclose(lib);
    }
#else
    const bool expected = false;
#endif
    set_no_rdma(false);
    const bool available = rpc_transport_rdma_available();
    set_no_rdma(true);
    const bool disabled = rpc_transport_rdma_available();
    set_no_rdma(false);
    printf("RDMA available: %s (expected %s)\n", available ? "yes" : "no", expected ? "yes" : "no");
    if (available != expected || disabled) {
        fprintf(stderr, "rpc_transport_rdma_available() returned %d, %d with GGML_RPC_NO_RDMA\n", available, disabled);
        return false;
    }
    return true;
}

// tests that also run on Windows, after rpc_transport_init()
static bool test_portable() {
    return test_ephemeral_port() && test_caps_handshake() && test_rdma_available();
}

int main() {
#ifdef _WIN32
    if (!test_transport_ref_count() || !rpc_transport_init()) {
        return 1;
    }
    const bool ok = test_portable();
    rpc_transport_shutdown();
    if (!ok) {
        return 1;
    }
#else
    if (!test_shutdown_send() || !test_accept_timeout() || !test_interrupted_accept_timeout() ||
        !test_high_fd_accept_timeout() || !test_failed_server_closes_socket() || !test_portable()) {
        return 1;
    }
#endif
    return 0;
}
