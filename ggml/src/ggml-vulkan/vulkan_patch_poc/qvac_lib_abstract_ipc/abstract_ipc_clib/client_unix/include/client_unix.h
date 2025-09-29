#ifndef ABSTRACT_IPC_CLIENT_UNIX_H
#define ABSTRACT_IPC_CLIENT_UNIX_H

#include "abstract_ipc_client_socket_unix.h"
#include <stdbool.h>

bool abstract_ipc_client_connect(AbstractIPCClientSocket_t *result,
                                 const char *base_name, int socket_index);
bool abstract_ipc_client_send_request(const AbstractIPCClientSocket_t *socket,
                                      unsigned char request, double *response);
void abstract_ipc_client_close(const AbstractIPCClientSocket_t *socket);

#endif
