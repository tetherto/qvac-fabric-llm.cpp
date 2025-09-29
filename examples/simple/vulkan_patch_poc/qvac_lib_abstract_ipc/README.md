
# Description
Minimal and lightweight inter-process communication based on identifiers. No ports numbers or address have to be specified, instead a string identifier is used to establish the channel. Intended to be used to query numerical values held inside any Add-on.

## Server setup
Add subdirectory, link against the `abstract_ipc_clib`, and include directories. Then use the c functions on your implementation. See `vulkan_hooks/CMakeLists.txt` as an example.

## Client setup
Build the `abstract_ipc_cpp` and use it to query values served from any other Addon that register into an specific channel.
