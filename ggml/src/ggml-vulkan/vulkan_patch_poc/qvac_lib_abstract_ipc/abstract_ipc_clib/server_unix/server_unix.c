#include "include/server_unix.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

bool abstract_ipc_server_create_socket(AbstractIPCServerSocket_t *result,
                                       const char *base_name) {
  AbstractIPCServerSocket_t server_socket;
  server_socket.socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_socket.socket_fd < 0) {
    fprintf(stderr,
            "create_abstract_ipc_server_socket: Socket creation failed\n");
    return false;
  }
  server_socket.addr.sun_family = AF_UNIX;

  {
    bool socket_bound = false;
    for (int index = 0; !socket_bound && index < ABSTRACT_IPC_MAX_SOCKET_SLOTS;
         index++) {
      // Max size for sun_path
      char socket_name[108];
      // First byte should be null to make it an abstract namespace
      socket_name[0] = '\0';
      snprintf(socket_name + 1, sizeof(socket_name) - 1, "%s[%i]", base_name,
               index);
      size_t socket_name_len = strlen(socket_name + 2) + 2;

      memcpy(server_socket.addr.sun_path, socket_name, socket_name_len);
      server_socket.addr_len =
          sizeof(server_socket.addr.sun_family) + socket_name_len;

      if (bind(server_socket.socket_fd, (struct sockaddr *)&server_socket.addr,
               server_socket.addr_len) == 0) {
        server_socket.index = index;
        socket_bound = true;
        *result = server_socket;
      }
    }

    if (!socket_bound) {
      fprintf(stderr,
              "create_abstract_namespace_server_socket: Failed to bind socket "
              "to abstract namespace after %d attempts\n",
              ABSTRACT_IPC_MAX_SOCKET_SLOTS);
      return false;
    }
  }

  if (listen(server_socket.socket_fd, ABSTRACT_IPC_MAX_CONNECTIONS) < 0) {
    fprintf(stderr, "create_abstract_namespace_server_socket: Listen failed\n");
    close(server_socket.socket_fd);
    return false;
  }

  return true;
}

bool abstract_ipc_server_handle_request(
    const AbstractIPCServerSocket_t *socket,
    double (*request_handler)(unsigned char)) {
  // Accept client connection
  socklen_t client_addr_len = sizeof(struct sockaddr_un);
  struct sockaddr_un client_addr;
  int client_fd = accept(socket->socket_fd, (struct sockaddr *)&client_addr,
                         &client_addr_len);

  if (client_fd < 0) {
    fprintf(stderr, "abstract_ipc_server_handle_request: Accept failed\n");
    return false;
  }

  // Read numeric value from client, represents the request type
  char received_value;
  if (read(client_fd, &received_value, sizeof(received_value)) < 0) {
    fprintf(stderr, "abstract_ipc_server_handle_request: Read error\n");
    close(client_fd);
    return false;
  }

  double response = request_handler(received_value);

  // Send the response back to client
  if (write(client_fd, &response, sizeof(response)) < 0) {
    fprintf(stderr, "abstract_ipc_server_handle_request: Write error\n");
    close(client_fd);
    return false;
  }

  close(client_fd);
  return true;
}

void abstract_ipc_server_close(const AbstractIPCServerSocket_t *socket) {
  close(socket->socket_fd);
}
