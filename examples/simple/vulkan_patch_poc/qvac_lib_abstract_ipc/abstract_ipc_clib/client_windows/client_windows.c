#include "include/client_windows.h"
#include <stdio.h>
#include <string.h>

bool abstract_ipc_client_connect(AbstractIPCClientSocket_t *result,
                                 const char *base_name, int socket_index) {
  AbstractIPCClientSocket_t client_socket;

  // Create pipe name in Windows format
  snprintf(client_socket.pipe_name, sizeof(client_socket.pipe_name),
           "\\\\.\\pipe\\%s[%i]", base_name, socket_index);

  // Connect to the named pipe
  while (1) {
    client_socket.pipe_handle =
        CreateFile(client_socket.pipe_name,      // pipe name
                   GENERIC_READ | GENERIC_WRITE, // read and write access
                   0,                            // no sharing
                   NULL,                         // default security attributes
                   OPEN_EXISTING,                // opens existing pipe
                   0,                            // default attributes
                   NULL);                        // no template file

    // Break if the pipe handle is valid
    if (client_socket.pipe_handle != INVALID_HANDLE_VALUE)
      break;

    // Exit if an error other than ERROR_PIPE_BUSY occurs
    if (GetLastError() != ERROR_PIPE_BUSY) {
#ifdef ABSTRACT_IPC_CLIENT_VERBOSE
      fprintf(stderr,
              "abstract_ipc_client_connect: Connection failed to pipe %s\n",
              client_socket.pipe_name);
#endif
      return false;
    }

    // All pipe instances are busy, so wait for 20 milliseconds
    if (!WaitNamedPipe(client_socket.pipe_name, 20000)) {
      fprintf(
          stderr,
          "abstract_ipc_client_connect: Could not open pipe after waiting\n");
      return false;
    }
  }

  // Set the pipe to message-read mode and block until data is available
  DWORD pipe_mode = PIPE_READMODE_MESSAGE | PIPE_WAIT;
  if (!SetNamedPipeHandleState(client_socket.pipe_handle, // pipe handle
                               &pipe_mode,                // new pipe mode
                               NULL,  // don't set maximum bytes
                               NULL)) // don't set maximum time
  {
    fprintf(stderr,
            "abstract_ipc_client_connect: SetNamedPipeHandleState failed\n");
    CloseHandle(client_socket.pipe_handle);
    return false;
  }

  // Return the initialized socket
  *result = client_socket;
  return true;
}

bool abstract_ipc_client_send_request(const AbstractIPCClientSocket_t *socket,
                                      unsigned char request, double *response) {
  DWORD bytes_written = 0;
  DWORD bytes_read = 0;

  // Send request to server
  if (!WriteFile(socket->pipe_handle, &request, sizeof(request), &bytes_written,
                 NULL) ||
      bytes_written != sizeof(request)) {
    fprintf(stderr, "abstract_ipc_client_send_request: Write failed\n");
    return false;
  }

  // Receive response from server
  if (!ReadFile(socket->pipe_handle, response, sizeof(*response), &bytes_read,
                NULL) ||
      bytes_read != sizeof(*response)) {
    fprintf(stderr, "abstract_ipc_client_send_request: Read failed\n");
    return false;
  }

  return true;
}

void abstract_ipc_client_close(const AbstractIPCClientSocket_t *socket) {
  CloseHandle(socket->pipe_handle);
}
