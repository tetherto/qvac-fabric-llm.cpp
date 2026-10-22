#include <cstdio>
#include <cstdlib>
#include <exception>

#include <dlfcn.h>

int main(int argc, char ** argv) {
    if (argc != 2 || unsetenv("GGML_NO_BACKTRACE") != 0) {
        fprintf(stderr, "expected a backend library path and enabled backtraces\n");
        return 1;
    }

    const auto original_handler = std::get_terminate();
    for (int i = 0; i < 100; ++i) {
        void * handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            fprintf(stderr, "backend load %d failed: %s\n", i + 1, dlerror());
            return 1;
        }

        const auto score = reinterpret_cast<int (*)()>(dlsym(handle, "ggml_backend_score"));
        if (score == nullptr || score() != 0) {
            fprintf(stderr, "backend must log and reject the candidate\n");
            dlclose(handle);
            return 1;
        }

        if (std::get_terminate() != original_handler) {
            fprintf(stderr, "backend logging replaced the process terminate handler\n");
            dlclose(handle);
            return 1;
        }
        if (dlclose(handle) != 0) {
            fprintf(stderr, "backend unload failed: %s\n", dlerror());
            return 1;
        }

        // A retained module would hide the reload failure.
        void * resident = dlopen(argv[1], RTLD_NOW | RTLD_NOLOAD);
        if (resident != nullptr) {
            fprintf(stderr, "backend remained loaded after rejection\n");
            dlclose(resident);
            return 1;
        }
        if (std::get_terminate() != original_handler) {
            fprintf(stderr, "backend unload changed the process terminate handler\n");
            return 1;
        }
    }
    return 0;
}
