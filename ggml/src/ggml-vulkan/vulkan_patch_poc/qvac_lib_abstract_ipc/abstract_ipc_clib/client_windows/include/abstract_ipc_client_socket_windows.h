#ifndef ABSTRACT_IPC_CLIENT_SOCKET_WINDOWS_H
#define ABSTRACT_IPC_CLIENT_SOCKET_WINDOWS_H

#include <windows.h>

typedef struct AbstractIPCClientSocket {
  HANDLE pipe_handle;
  char pipe_name[256];
} AbstractIPCClientSocket_t;

#endif
