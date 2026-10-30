// Reproducer for the lost first C line of a bf16 GEMM block (README, "Known limits"): an LD_PRELOAD library that takes
// the address reservations the XRT shim makes for buffer-object mappings from a private arena, back to back, so one
// buffer's data ends on the first byte of the next. Without a guard page before C, every process then loses lines.
//
//   gcc -O2 -shared -fPIC -o bo_place_shim.so bo_place_shim.c -ldl
//   BO_PLACE=up|down            later reservations go above (up) or below (down) the earlier ones
//   BO_PLACE_MIN, BO_PLACE_MAX  sizes of the reservations to place, in bytes (default 65536 to 16777216)
//
//   LD_PRELOAD=./bo_place_shim.so BO_PLACE=down llama-embedding -m embeddinggemma-2-BF16.gguf --device XDNA0 -ngl 99
//     --poll 0 -fa on -c 4096 -b 4096 -ub 4096 --no-warmup -p "<one sentence repeated 46 times, about 600 tokens>"
//
// A build without the guard fails the op ("still hold the mark"); this one logs "running the block again" instead.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define ARENA_BYTES ((size_t) 64 << 30)
#define PAGE        ((uintptr_t) 4096)

typedef void * (*mmap_fn)(void *, size_t, int, int, int, off_t);

static mmap_fn          real_mmap;
static int              up, enabled;
static size_t           min_len = 65536, max_len = 16777216;
static uintptr_t        cur;
static __thread int     in_shim;

static void init(void) {
    real_mmap        = (mmap_fn) dlsym(RTLD_NEXT, "mmap64");
    const char * how = getenv("BO_PLACE");
    if (!how) {
        return;
    }
    up = strcmp(how, "up") == 0;
    if (getenv("BO_PLACE_MIN")) {
        min_len = (size_t) atol(getenv("BO_PLACE_MIN"));
    }
    if (getenv("BO_PLACE_MAX")) {
        max_len = (size_t) atol(getenv("BO_PLACE_MAX"));
    }
    void * a = real_mmap(NULL, ARENA_BYTES, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (a != MAP_FAILED) {
        cur     = (uintptr_t) a + (up ? 0 : ARENA_BYTES);
        enabled = 1;
    }
}

static int from_xrt(void * ret) {
    Dl_info di;
    return dladdr(ret, &di) && di.dli_fname && (strstr(di.dli_fname, "xrt") || strstr(di.dli_fname, "xdna"));
}

static void * place(void * addr, size_t len, int prot, int flags, int fd, off_t off, void * caller) {
    if (!real_mmap) {
        init();
    }
    const int reservation = fd < 0 && !addr && (flags & MAP_ANONYMOUS) && !(flags & MAP_FIXED);
    if (!enabled || in_shim || !reservation || len < min_len || len > max_len) {
        return real_mmap(addr, len, prot, flags, fd, off);
    }
    in_shim = 1;
    const int ours = from_xrt(caller);
    in_shim = 0;
    if (!ours) {
        return real_mmap(addr, len, prot, flags, fd, off);
    }
    const uintptr_t span = (len + PAGE - 1) & ~(PAGE - 1);
    uintptr_t       at   = cur;
    if (up) {
        cur += span;
    } else {
        cur -= span;
        at = cur;
    }
    return real_mmap((void *) at, len, prot, flags | MAP_FIXED, fd, off);
}

void * mmap64(void * addr, size_t len, int prot, int flags, int fd, off_t off) {
    return place(addr, len, prot, flags, fd, off, __builtin_return_address(0));
}

void * mmap(void * addr, size_t len, int prot, int flags, int fd, off_t off) {
    return place(addr, len, prot, flags, fd, off, __builtin_return_address(0));
}
