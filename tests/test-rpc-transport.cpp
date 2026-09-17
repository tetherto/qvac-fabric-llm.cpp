#include "transport.h"

#include <cstdio>

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

int main() {
#ifdef _WIN32
    if (!test_transport_ref_count()) {
        return 1;
    }
#else
    if (!test_shutdown_send() || !test_accept_timeout() || !test_interrupted_accept_timeout() ||
        !test_high_fd_accept_timeout()) {
        return 1;
    }
#endif
    return 0;
}
