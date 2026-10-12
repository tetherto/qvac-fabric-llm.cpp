#pragma once

// True when the libibverbs functions can be called. With GGML_RPC_RDMA_DLOPEN the
// first call loads libibverbs.so.1, and RDMA is off when it or a symbol is missing.
bool ibvdrv_init();
