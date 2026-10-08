## cmake-test

This is just for manually testing/developing of a llama.cpp installation to
enable troubleshooting issues and exploration. The idea is that this can be used
after making changes to llama.cpp installation cmake configuration and then
verify it locally.

### find_package
The following will configure, build, and install llama.cpp, and the build a
project that uses find_package to use the installation.

Configuring/build/install:
```console
./build-install.sh
```
The above command will create a directory named `install` in the current directory
which will have the following files in its lib directory:
```console
$ ls install/lib/
cmake                        libqvac-ggml.so          libllama-common.so.0      libllama.so.0.1.0  llama.cpp
libqvac-ggml-base.so         libqvac-ggml.so.0        libllama-common.so.0.1.0  libmtmd.so         pkgconfig
libqvac-ggml-base.so.0       libqvac-ggml.so.0.19.0   libllama.so               libmtmd.so.0
libqvac-ggml-base.so.0.19.0  libllama-common.so       libllama.so.0             libmtmd.so.0.1.0
```

Build/run this project using the installation created above:
```console
$ ./build.sh
-- Configuring done (0.0s)
-- Generating done (0.0s)
-- Build files have been written to: /path/to/llama.cpp/examples/test-cmake/build
[100%] Built target test-cmake
[test-cmake] Using llama.cpp version 0.1.0-dev-b10335
[test-cmake] Initializing backend...
load_backend: loaded CPU backend from /path/to/llama.cpp/examples/test-cmake/install/lib/llama.cpp/libqvac-ggml-cpu-alderlake.so
[test-cmake] Backend initialized.
```

### add_subdirectory
The following will use add_subdirectory to include llama.cpp in a cmake project
and is intended to simulate projects that build llama.cpp in this way.

```console
$ USE_SUBDIR=ON ./build.sh
```
