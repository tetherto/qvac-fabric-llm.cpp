#ifndef ABSTRACT_IPC_SERVER_H
#define ABSTRACT_IPC_SERVER_H

#ifdef _WIN32
#include "server_windows/include/server_windows.h"
#else
#include "server_unix/include/server_unix.h"
#endif

#endif
