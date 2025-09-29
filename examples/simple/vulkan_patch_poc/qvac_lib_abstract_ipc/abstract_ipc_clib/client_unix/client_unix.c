#include "include/client_unix.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

bool abstract_ipc_client_connect(AbstractIPCClientSocket_t *result,
                                 const char *base_name, int socket_index) {
  AbstractIPCClientSocket_t client_socket;

  // Create socket
  client_socket.socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (client_socket.socket_fd < 0) {
    fprintf(stderr, "abstract_ipc_client_connect: Socket creation failed\n");
    return false;
  }

  // Setup address structure for abstract namespace
  memset(&client_socket.addr, 0, sizeof(client_socket.addr));
  client_socket.addr.sun_family = AF_UNIX;

  // Max size for sun_path
  char socket_name[108];
  // First byte should be null to make it an abstract namespace
  socket_name[0] = '\0';
  snprintf(socket_name + 1, sizeof(socket_name) - 1, "%s[%i]", base_name,
           socket_index);
  size_t socket_name_len = strlen(socket_name + 2) + 2;

  // Copy socket name (including null byte at start for abstract namespace)
  memcpy(client_socket.addr.sun_path, socket_name, socket_name_len);

  // Calculate the actual size to use with connect
  client_socket.addr_len =
      sizeof(client_socket.addr.sun_family) + socket_name_len;

  // Connect to server
  if (connect(client_socket.socket_fd, (struct sockaddr *)&client_socket.addr,
              client_socket.addr_len) < 0) {
#ifdef ABSTRACT_IPC_CLIENT_VERBOSE
    fprintf(stderr,
            "abstract_ipc_client_connect: Connection failed to socket %s\n",
            socket_name + 1);
#endif
    close(client_socket.socket_fd);
    return false;
  }

  // Return the initialized socket
  *result = client_socket;
  return true;
}

bool abstract_ipc_client_send_request(const AbstractIPCClientSocket_t *socket,
                                      unsigned char request, double *response) {
  // Send request to server
  if (write(socket->socket_fd, &request, sizeof(request)) < 0) {
    fprintf(stderr, "abstract_ipc_client_send_request: Write failed\n");
    return false;
  }

  // Receive response from server
  if (read(socket->socket_fd, response, sizeof(*response)) < 0) {
    fprintf(stderr, "abstract_ipc_client_send_request: Read failed\n");
    return false;
  }

  return true;
}

void abstract_ipc_client_close(const AbstractIPCClientSocket_t *socket) {
  close(socket->socket_fd);
}
