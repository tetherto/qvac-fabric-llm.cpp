#include "xdna-rec-post.h"

#include "xdna_design_tag.h"

#include <cstdio>
#include <filesystem>
#include <memory>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

namespace {

// The design's device and sequence symbols (kernels/post_norm.py compiled as
// a full ELF); the kernel is addressed by "<device>:<sequence>".
const char * const POST_ELF_KERNEL = "main:sequence";

// Find the ELF artifact in the kernel search dirs.
std::string post_elf_path() {
    const std::string name = std::string("post_norm_full.elf");
    for (const auto & dir : xdna_kernel_search_dirs()) {
        std::error_code ec;
        const std::string f = (dir / name).string();
        if (std::filesystem::exists(f, ec)) {
            return f;
        }
    }
    return "";
}

} // namespace

struct xdna_rec_post_elf {
    std::shared_ptr<xrt::elf>          elf;
    std::shared_ptr<xrt::hw_context>   ctx;
    std::shared_ptr<xrt::ext::kernel>  kern;
};

xdna_rec_post * xdna_rec_post_create(xdna_kernel_pool * pool) {
    if (!pool) {
        return nullptr;
    }
    const std::string path = post_elf_path();
    if (path.empty()) {
        fprintf(stderr, "xdna-rec-post: post_norm_full.elf not found\n");
        return nullptr;
    }
    xdna_rec_post * p = new xdna_rec_post;
    p->dev = pool->device;
    p->elf_handle = new xdna_rec_post_elf;
    try {
        p->elf_handle->elf = std::make_shared<xrt::elf>(path);
        p->elf_handle->ctx = std::make_shared<xrt::hw_context>(
            pool->device->device, *p->elf_handle->elf,
            xrt::hw_context::cfg_param_type{});
        p->elf_handle->kern = std::make_shared<xrt::ext::kernel>(
            *p->elf_handle->ctx, POST_ELF_KERNEL);
    } catch (const std::exception & e) {
        fprintf(stderr, "xdna-rec-post: full-ELF load failed: %s\n", e.what());
        delete p->elf_handle;
        delete p;
        return nullptr;
    }
    p->in  = xdna_buffer_alloc(p->dev, (3 * 1024 + 4) * 4);
    p->out = xdna_buffer_alloc(p->dev, 1024 * 4 + (1 + 1024 / 256) * 2112);
    if (!p->in || !p->out) {
        xdna_rec_post_free(p);
        return nullptr;
    }
    p->ok = true;
    return p;
}

bool xdna_rec_post_run(xdna_rec_post * p) {
    if (!p || !p->ok || !p->elf_handle) {
        return false;
    }
    try {
        // A fresh run per dispatch: the full-ELF path binds its buffers and
        // starts, there is no instruction stream to restart.
        xrt::run run(*p->elf_handle->kern);
        run.set_arg(0, p->in->bo);
        run.set_arg(1, p->out->bo);
        run.start();
        if (run.wait() != ERT_CMD_STATE_COMPLETED) {
            return false;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "xdna-rec-post: dispatch failed: %s\n", e.what());
        return false;
    }
    xdna_buffer_sync_from_device(p->out);
    return true;
}

void xdna_rec_post_free(xdna_rec_post * p) {
    if (!p) {
        return;
    }
    xdna_buffer_free(p->in);
    xdna_buffer_free(p->out);
    delete p->elf_handle;
    delete p;
}
