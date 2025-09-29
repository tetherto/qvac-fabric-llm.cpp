#include "include/server_windows.h"
#include <stdio.h>
#include <string.h>

bool abstract_ipc_server_create_socket(AbstractIPCServerSocket_t *result,
                                       const char *base_name) {
  AbstractIPCServerSocket_t server_socket;

  bool socket_created = false;
  for (int index = 0; !socket_created && index < ABSTRACT_IPC_MAX_SOCKET_SLOTS;
       index++) {
    // Create pipe name in Windows format
    snprintf(server_socket.pipe_name, sizeof(server_socket.pipe_name),
             "\\\\.\\pipe\\%s[%i]", base_name, index);

    // Create the named pipe
    server_socket.pipe_handle =
        CreateNamedPipe(server_socket.pipe_name,      // pipe name
                        PIPE_ACCESS_DUPLEX,           // read/write access
                        PIPE_TYPE_MESSAGE |           // message-type pipe
                            PIPE_READMODE_MESSAGE |   // message read mode
                            PIPE_WAIT,                // blocking mode
                        ABSTRACT_IPC_MAX_CONNECTIONS, // number of instances
                        1024,                         // output buffer size
                        1024,                         // input buffer size
                        0,                            // default timeout
                        NULL); // default security attributes

    if (server_socket.pipe_handle != INVALID_HANDLE_VALUE) {
      server_socket.index = index;
      socket_created = true;
      *result = server_socket;
    } else {
      // If error is not that the pipe exists, break the loop
      if (GetLastError() != ERROR_PIPE_BUSY) {
        continue; // Try next index
      }
    }
  }

  if (!socket_created) {
    fprintf(stderr,
            "abstract_ipc_server_create_socket: Failed to create named pipe "
            "after %d attempts\n",
            ABSTRACT_IPC_MAX_SOCKET_SLOTS);
    return false;
  }

  return true;
}

bool abstract_ipc_server_handle_request(
    const AbstractIPCServerSocket_t *socket,
    double (*request_handler)(unsigned char)) {
  // Wait for a client to connect
  if (!ConnectNamedPipe(socket->pipe_handle, NULL) &&
      GetLastError() != ERROR_PIPE_CONNECTED) {
    fprintf(stderr, "abstract_ipc_server_handle_request: Connect failed\n");
    return false;
  }

  // Read request from client
  unsigned char received_value;
  DWORD bytes_read = 0;

  if (!ReadFile(socket->pipe_handle, &received_value, sizeof(received_value),
                &bytes_read, NULL) ||
      bytes_read != sizeof(received_value)) {
    fprintf(stderr, "abstract_ipc_server_handle_request: Read error\n");
    DisconnectNamedPipe(socket->pipe_handle);
    return false;
  }

  // Process the request
  double response = request_handler(received_value);

  // Send response back to client
  DWORD bytes_written = 0;
  if (!WriteFile(socket->pipe_handle, &response, sizeof(response),
                 &bytes_written, NULL) ||
      bytes_written != sizeof(response)) {
    fprintf(stderr, "abstract_ipc_server_handle_request: Write error\n");
    DisconnectNamedPipe(socket->pipe_handle);
    return false;
  }

  // Disconnect the client
  DisconnectNamedPipe(socket->pipe_handle);
  return true;
}

void abstract_ipc_server_close(const AbstractIPCServerSocket_t *socket) {
  CloseHandle(socket->pipe_handle);
}
