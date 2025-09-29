#ifndef ABSTRACT_IPC_SERVER_SOCKET_WINDOWS_H
#define ABSTRACT_IPC_SERVER_SOCKET_WINDOWS_H

#include <windows.h>

typedef struct AbstractIPCServerSocket {
  HANDLE pipe_handle;
  int index;
  char pipe_name[256];
} AbstractIPCServerSocket_t;

#endif
