#ifndef ABSTRACT_IPC_CLIENT_H
#define ABSTRACT_IPC_CLIENT_H

#ifdef _WIN32
#include "client_windows/include/client_windows.h"
#else
#include "client_unix/include/client_unix.h"
#endif

#endif
