#ifndef ABSTRACT_IPC_SERVER_WINDOWS_H
#define ABSTRACT_IPC_SERVER_WINDOWS_H

#include "abstract_ipc_server_socket_windows.h"
#include <stdbool.h>

#ifndef ABSTRACT_IPC_MAX_CONNECTIONS
#define ABSTRACT_IPC_MAX_CONNECTIONS 5
#endif

#ifndef ABSTRACT_IPC_MAX_SOCKET_SLOTS
#define ABSTRACT_IPC_MAX_SOCKET_SLOTS 64
#endif

typedef double (*RequestHandler)(unsigned char);

bool abstract_ipc_server_create_socket(AbstractIPCServerSocket_t *result,
                                       const char *base_name);
bool abstract_ipc_server_handle_request(const AbstractIPCServerSocket_t *socket,
                                        RequestHandler request_handler);
void abstract_ipc_server_close(const AbstractIPCServerSocket_t *socket);

#endif
