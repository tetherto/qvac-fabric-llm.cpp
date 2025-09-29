#ifndef ABSTRACT_IPC_CLIENT_SOCKET_UNIX_H
#define ABSTRACT_IPC_CLIENT_SOCKET_UNIX_H

// These headers should be available in Linux, Android, MacOS and iOS but not in
// Windows
#include <sys/socket.h>
#include <sys/un.h>

typedef struct AbstractIPCClientSocket {
  int socket_fd;
  struct sockaddr_un addr;
  socklen_t addr_len;
} AbstractIPCClientSocket_t;

#endif
