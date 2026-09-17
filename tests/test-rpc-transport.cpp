#include "transport.h"

#include <cstdio>

#ifndef _WIN32
#    include <arpa/inet.h>
#    include <sys/socket.h>
#    include <sys/wait.h>
#    include <unistd.h>

#    include <cerrno>
#    include <csignal>

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
#endif

int main() {
#ifndef _WIN32
    if (!test_shutdown_send()) {
        return 1;
    }
#endif
    return 0;
}
