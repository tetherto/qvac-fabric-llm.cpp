#ifndef ABSTRACT_IPC_SOCKET_UNIX_H
#define ABSTRACT_IPC_SOCKET_UNIX_H

// These headers should be available in Linux, Android, MacOS and iOS but not in
// Windows
#include <sys/socket.h>
#include <sys/un.h>

typedef struct AbstractIPCServerSocket {
  int index;
  int socket_fd;
  struct sockaddr_un addr;
  socklen_t addr_len;
} AbstractIPCServerSocket_t;

#endif
