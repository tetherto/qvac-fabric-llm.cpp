cmake_minimum_required(VERSION 3.10)
project(AbstractIPC C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Detect Windows platform
if(WIN32)
    # Windows implementation
    add_library(abstract_namespace_server STATIC server_windows/server_windows.c)
    add_library(abstract_namespace_client STATIC client_windows/client_windows.c)
else()
    # Unix/Linux implementation
    add_library(abstract_namespace_server STATIC server_unix/server_unix.c)
    add_library(abstract_namespace_client STATIC client_unix/client_unix.c)
endif()
