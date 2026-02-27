// the VMA implementation is compiled in this file only
#define VMA_IMPLEMENTATION
#include "ggml-vulkan-common.h"

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>

static void * portable_aligned_malloc(size_t align, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, align);
#else
    void * ptr = nullptr;
    if (posix_memalign(&ptr, align, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

static void portable_aligned_free(void * ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

ggml_backend_buffer_type_i ggml_backend_vk_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_vk_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_vk_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_vk_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

static vk_buffer ggml_vk_create_buffer(vk_device& device, const vk::BufferCreateInfo& buffer_info, const VmaAllocationCreateInfo& alloc_info) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << buffer_info.size << ")");
    if (buffer_info.size > device->max_buffer_size) {
        throw vk::OutOfDeviceMemoryError("Requested buffer size exceeds device buffer size limit");
    }

    vk_buffer buf = std::make_shared<vk_buffer_struct>();

    if (buffer_info.size == 0) {
        buf->size = 0;
        return buf;
    }

    VkResult res = vmaCreateBuffer(device->allocator, reinterpret_cast<const VkBufferCreateInfo *>(&buffer_info), &alloc_info, reinterpret_cast<VkBuffer *>(&buf->buffer), &buf->allocation, &buf->info);
    if (res != VK_SUCCESS) {
        throw vk::SystemError(static_cast<vk::Result>(res), "vmaCreateBuffer failed");
    }

    vmaGetAllocationMemoryProperties(device->allocator, buf->allocation, reinterpret_cast<VkMemoryPropertyFlags *>(&buf->memory_property_flags));

    buf->device = device;
    buf->size = buffer_info.size;

    if (device->buffer_device_address && (buffer_info.usage & vk::BufferUsageFlagBits::eShaderDeviceAddress)) {
        const vk::BufferDeviceAddressInfo addressInfo(buf->buffer);
        buf->bda_addr = device->device.getBufferAddress(addressInfo);
    }

#ifdef GGML_VULKAN_MEMORY_DEBUG
    device->memory_logger->log_allocation(buf, buf->size);
#endif
    VK_LOG_DEBUG("ggml_vk_create_buffer(buffer " << buf->buffer << ", usage " << to_string(buffer_info.usage) << ", memory property " << to_string(buf->memory_property_flags) << ")");

    return buf;
}

vk_buffer ggml_vk_create_buffer_check(vk_device& device, size_t size) {
    try {
        const vk::BufferCreateInfo buffer_info{
            vk::BufferCreateFlags(),
            size,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            vk::SharingMode::eExclusive,
            0,
            nullptr,
        };

        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.usage = (device->prefer_host_memory ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO);
        alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT;
        alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        return ggml_vk_create_buffer(device, buffer_info, alloc_info);
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }
}

static vk_buffer ggml_vk_create_buffer_aligned(vk_device& device, const vk::BufferCreateInfo& buffer_info, const VmaAllocationCreateInfo& alloc_info, VkDeviceSize alignment) {
    VK_LOG_DEBUG("ggml_vk_create_buffer_aligned(" << device->name << ", " << buffer_info.size << ", " << alignment << " )");
    if (buffer_info.size > device->max_buffer_size) {
        throw vk::OutOfDeviceMemoryError("Requested buffer size exceeds device buffer size limit");
    }

    vk_buffer buf = std::make_shared<vk_buffer_struct>();

    if (buffer_info.size == 0) {
        buf->size = 0;
        return buf;
    }

    VkResult res = vmaCreateBufferWithAlignment(device->allocator, reinterpret_cast<const VkBufferCreateInfo *>(&buffer_info), &alloc_info, alignment, reinterpret_cast<VkBuffer *>(&buf->buffer), &buf->allocation, &buf->info);
    if (res != VK_SUCCESS) {
        throw vk::SystemError(static_cast<vk::Result>(res), "vmaCreateBufferWithAlignment failed");
    }

    vmaGetAllocationMemoryProperties(device->allocator, buf->allocation, reinterpret_cast<VkMemoryPropertyFlags *>(&buf->memory_property_flags));

    buf->device = device;
    buf->size = buffer_info.size;

    if (device->buffer_device_address && (buffer_info.usage & vk::BufferUsageFlagBits::eShaderDeviceAddress)) {
        const vk::BufferDeviceAddressInfo addressInfo(buf->buffer);
        buf->bda_addr = device->device.getBufferAddress(addressInfo);
    }

#ifdef GGML_VULKAN_MEMORY_DEBUG
    device->memory_logger->log_allocation(buf, buf->size);
#endif
    VK_LOG_DEBUG("ggml_vk_create_buffer_aligned(buffer " << buf->buffer << ", usage " << to_string(buffer_info.usage) << ", memory property " << to_string(buf->memory_property_flags) << ")");

    return buf;
}

vk_buffer ggml_vk_create_buffer_device(vk_device& device, size_t size) {
    VK_LOG_MEMORY("ggml_vk_create_buffer_device(" << size << ")");

    vk_buffer buf;

    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    if (device->buffer_device_address) {
        usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
    }

    const vk::BufferCreateInfo buffer_create_info{
        vk::BufferCreateFlags(),
        size,
        usage,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
    };

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = (device->prefer_host_memory ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO);
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;

    try {
        buf = ggml_vk_create_buffer(device, buffer_create_info, alloc_info);
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }

    return buf;
}

void ggml_vk_destroy_buffer(vk_buffer& buf) {
    if (buf == nullptr) {
        return;
    }

    if (buf->device != nullptr) {
        buf->device->memory_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
static size_t ggml_vk_host_allocation_find(const void * ptr) {
    for (size_t i = 0; i < vk_instance.host_allocations.size(); i++) {
        const uint8_t * base = (const uint8_t *) vk_instance.host_allocations[i].base;
        if (ptr >= base && ptr < base + vk_instance.host_allocations[i].size) {
            return i;
        }
    }
    return SIZE_MAX;
}

void * ggml_vk_host_malloc(size_t size) {
    VK_LOG_MEMORY("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    if (vk_instance.multi_device) {
        const size_t align      = std::max<size_t>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        const size_t alloc_size = (size + align - 1) / align * align;

        void * ptr = portable_aligned_malloc(align, alloc_size);
        if (ptr == nullptr) {
            return nullptr;
        }
        vk_host_allocation entry;
        entry.base = ptr;
        entry.size = alloc_size;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(entry));
        return ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    const vk::BufferCreateInfo buffer_info{
        vk::BufferCreateFlags(),
        size,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
    };

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = (dev0->prefer_host_memory ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO);
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;

    vk_buffer buf;
    try {
        buf = ggml_vk_create_buffer_aligned(dev0, buffer_info, alloc_info, TENSOR_ALIGNMENT);
    } catch (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed to allocate pinned memory (%s)\n", e.what());
        return nullptr;
    }

    if (!(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_buffer(buf);
        return nullptr;
    }

    vk_host_allocation entry;
    entry.base             = buf->info.pMappedData;
    entry.size             = size;
    entry.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(entry));
    return buf->info.pMappedData;
}

void ggml_vk_host_free(void * ptr) {
    if (ptr == nullptr) {
        return;
    }
    VK_LOG_MEMORY("ggml_vk_host_free(" << ptr << ")");

    vk_host_allocation entry;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        const size_t index = ggml_vk_host_allocation_find(ptr);
        if (index == SIZE_MAX) {
            fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
            return;
        }
        entry = std::move(vk_instance.host_allocations[index]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + index);
    }

    for (auto & view : entry.views) {
        ggml_vk_destroy_buffer(view);
    }
    if (vk_instance.multi_device) {
        portable_aligned_free(entry.base);
    }
}

void ggml_vk_host_get(const vk_device& device, const void * ptr, vk_buffer& buf, size_t& buf_offset) {
    buf = nullptr;
    buf_offset = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    const size_t index = ggml_vk_host_allocation_find(ptr);
    if (index == SIZE_MAX) {
        return;
    }
    vk_host_allocation & entry = vk_instance.host_allocations[index];

    if (entry.views[device->idx] == nullptr) {
        const uint32_t bit = 1u << device->idx;
        if (!vk_instance.multi_device || !vk_instance.device_host_buffer_import[device->idx] || (vk_instance.host_import_failed & bit)) {
            return;
        }
        GGML_ASSERT(device->external_memory_host);
        // const only propagates from the const context; the pointee is mutable
        entry.views[device->idx] = ggml_vk_buffer_from_host_ptr(const_cast<vk_device&>(device), entry.base, entry.size);
        if (entry.views[device->idx] == nullptr) {
            vk_instance.host_import_failed |= bit;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            return;
        }
    }

    buf = entry.views[device->idx];
    buf_offset = (const uint8_t *) ptr - (const uint8_t *) entry.base;
}

void ggml_vk_ensure_sync_staging_buffer(vk_device& device, size_t size) {
    if (device->sync_staging == nullptr || device->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(device->sync_staging);
        device->sync_staging = ggml_vk_create_buffer_check(device, size);
    }
}

void ggml_vk_ensure_sync_staging_buffer(ggml_backend_vk_context * ctx, size_t size) {
    if (ctx->sync_staging == nullptr || ctx->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_buffer_check(ctx->device, size);
    }
}

static vk_context ggml_vk_begin_sync_ctx(vk_device & device, vk_command_pool & pool) {
    vk_context subctx = ggml_vk_create_temporary_context(pool);
    ggml_vk_ctx_begin(device, subctx);
    return subctx;
}

static void ggml_vk_submit_sync(vk_device & device, vk_context & subctx) {
    ggml_vk_ctx_end(subctx);

    for (auto & cpy : subctx->in_memcpys) {
        memcpy(cpy.dst, cpy.src, cpy.n);
    }

    for (auto & mset : subctx->memsets) {
        memset(mset.dst, mset.val, mset.n);
    }

    ggml_vk_submit(subctx, device->fence);
    VK_CHECK(device->device.waitForFences({ device->fence }, true, UINT64_MAX), "vk_submit_sync waitForFences", device);
    device->device.resetFences({ device->fence });
    ggml_vk_queue_command_pools_cleanup(device);

    for (auto & cpy : subctx->out_memcpys) {
        memcpy(cpy.dst, cpy.src, cpy.n);
    }
}

static void ggml_vk_buffer_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_buffer& dst, size_t offset, const ggml_tensor * tensor, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::endl;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(ctx->device, tensor->data, buf, buf_offset);

    const uint64_t ne0 = tensor->ne[0];
    const uint64_t ne1 = tensor->ne[1];
    const uint64_t ne2 = tensor->ne[2];
    const uint64_t ne3 = tensor->ne[3];
    const uint64_t nb0 = tensor->nb[0];
    const uint64_t nb1 = tensor->nb[1];
    const uint64_t nb2 = tensor->nb[2];
    const uint64_t nb3 = tensor->nb[3];
    const ggml_type type = tensor->type;
    const uint64_t ts = ggml_type_size(type);
    const uint64_t bs = ggml_blck_size(type);

    const uint64_t dstnb0 = ts;
    const uint64_t dstnb1 = dstnb0*(ne0/bs);
    const uint64_t dstnb2 = dstnb1*ne1;
    const uint64_t dstnb3 = dstnb2*ne2;

    const uint64_t ne = ggml_nelements(tensor);

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices;

        for (uint64_t i3 = 0; i3 < ne3; i3++) {
            for (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                if (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_offset + i3*nb3 + i2*nb2, offset + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } else {
                    for (uint64_t i1 = 0; i1 < ne1; i1++) {
                        if (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_offset + i3*nb3 + i2*nb2 + i1*nb1, offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } else {
                            const uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                            const uint64_t d_off = offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            for (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        return;
    }

    if (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_buffer& staging = ctx->device->sync_staging;
    const uint64_t copy_size = ts*ne/bs;
    ggml_vk_ensure_sync_staging_buffer(ctx->device, copy_size);
    VkBufferCopy buf_copy{ 0, offset, copy_size };

    ggml_vk_sync_buffers(ctx, subctx);
    vkCmdCopyBuffer(subctx->s->buffer->buf, (VkBuffer)staging->buffer, (VkBuffer)dst->buffer, 1, &buf_copy);

    for (uint64_t i3 = 0; i3 < ne3; i3++) {
        for (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            if (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->info.pMappedData + i3*dstnb3 + i2*dstnb2, (const uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2, dstnb2, &subctx->in_memcpys);
            } else {
                for (uint64_t i1 = 0; i1 < ne1; i1++) {
                    if (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->info.pMappedData + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (const uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->in_memcpys);
                    } else {
                        const uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                        const uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        for (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->info.pMappedData + d_off + i0*dstnb0, (const uint8_t *) tensor->data + s_off + i0*nb0, dstnb0, &subctx->in_memcpys);
                        }
                    }
                }
            }
        }
    }
}

bool ggml_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << width << ", " << height << ")");
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_offset);

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices(1);
        if (width == spitch && width == dpitch) {
            // Only do single write if stride is equal
            slices[0].srcOffset = buf_offset;
            slices[0].dstOffset = offset;
            slices[0].size = width * height;
        } else {
            slices.resize(height);
            for (size_t i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * dpitch;
                slices[i].size = width;
            }
        }

        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        return true;
    }
    VK_LOG_DEBUG("STAGING");

    if (!sync_staging) {
        // copy was not handled caller needs to fall back
        return false;
    }

    // Staging buffer required
    const size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(dst->device, staging_size);

    vk_buffer& staging_buffer = dst->device->sync_staging;

    std::vector<vk::BufferCopy> slices(1);
    if (width == dpitch) {
        slices[0].srcOffset = 0;
        slices[0].dstOffset = offset;
        slices[0].size = staging_size;
    } else {
        slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = i * width;
            slices[i].dstOffset = offset + i * dpitch;
            slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(staging_buffer->buffer, dst->buffer, slices);

    if (width == spitch) {
        deferred_memcpy((uint8_t *)staging_buffer->info.pMappedData, src, staging_size, &subctx->in_memcpys);
    } else {
        for (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *)staging_buffer->info.pMappedData + i * width, (const uint8_t *) src + i * spitch, width, &subctx->in_memcpys);
        }
    }
    return true;
}

bool ggml_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t size, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    return ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, size, size, size, 1, sync_staging);
}

void ggml_vk_buffer_write_2d(vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << width << ", " << height << ")");
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        if (width == spitch && width == dpitch) {
            memcpy((uint8_t *)dst->info.pMappedData + offset, src, width * height);
        } else {
            for (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *)dst->info.pMappedData + offset + i * dpitch, (const uint8_t *) src + i * spitch, width);
            }
        }
    } else {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_begin_sync_ctx(dst->device, dst->device->transfer_queue->cmd_pool);
        bool ret = ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, spitch, dpitch, width, height, true);
        GGML_ASSERT(ret);
        ggml_vk_submit_sync(dst->device, subctx);
    }
}

void ggml_vk_buffer_write(vk_buffer& dst, size_t offset, const void * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_buffer_write_2d(dst, offset, src, size, size, size, 1);
}

bool ggml_vk_buffer_read_2d_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << offset << ", width=" << width << ", height=" << height << ")");
    GGML_ASSERT(width > 0);
    GGML_ASSERT(height > 0);
    GGML_ASSERT(src != nullptr);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_offset);

    std::vector<vk::BufferCopy> slices(1);
    if (width == spitch && width == dpitch) {
        // Only do single write if stride is equal
        slices[0].srcOffset = offset;
        slices[0].dstOffset = buf_offset;
        slices[0].size = width * height;
    } else {
        slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = offset + i * spitch;
            slices[i].dstOffset = buf_offset + i * dpitch;
            slices[i].size = width;
        }
    }

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(src->buffer, buf->buffer, slices);

        return true;
    }
    VK_LOG_DEBUG("STAGING");

    if (!sync_staging) {
        // copy was not handled caller needs to fall back
        return false;
    }

    // Fall back to staging buffer
    const size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(src->device, staging_size);

    vk_buffer& staging_buffer = src->device->sync_staging;

    std::vector<vk::BufferCopy> staging_slices(1);
    if (width == spitch) {
        staging_slices[0].srcOffset = offset;
        staging_slices[0].dstOffset = 0;
        staging_slices[0].size = staging_size;
    } else {
        staging_slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            staging_slices[i].srcOffset = offset + i * spitch;
            staging_slices[i].dstOffset = i * width;
            staging_slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(src->buffer, staging_buffer->buffer, staging_slices);

    if (width == dpitch) {
        deferred_memcpy(dst, staging_buffer->info.pMappedData, staging_size, &subctx->out_memcpys);
    } else {
        for (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (const uint8_t *) staging_buffer->info.pMappedData + i * width, width, &subctx->out_memcpys);
        }
    }
    return true;
}

static bool ggml_vk_buffer_read_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t size, bool sync_staging = false) {
    return ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, size, size, size, 1, sync_staging);
}

static bool ggml_vk_buffer_host_read_direct(const vk_buffer & buf) {
    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // UMA: uncached host-visible memory is write-combined, CPU reads are slow (except Nvidia, based on benchmarks)
    const bool slow_host_read = buf->device->vendor_id != VK_VENDOR_ID_NVIDIA &&
                                !(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCached);
    return (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) && buf->device->uma && !slow_host_read;
}

void ggml_vk_buffer_read_2d(vk_buffer& src, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->buffer << ", " << offset << ", " << width << ", " << height << ")");

    if (ggml_vk_buffer_host_read_direct(src)) {
        GGML_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_begin_sync_ctx(src->device, src->device->compute_queue->cmd_pool);
        subctx->s->buffer->buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost,
            {},
            { { vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eHostRead } },
            {}, {});
        ggml_vk_submit_sync(src->device, subctx);

        if (width == spitch && width == dpitch) {
            memcpy(dst, (const uint8_t *) src->info.pMappedData + offset, width * height);
        } else {
            for (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *) dst + i * dpitch, (const uint8_t *) src->info.pMappedData + offset + i * spitch, width);
            }
        }
    } else {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_begin_sync_ctx(src->device, src->device->transfer_queue->cmd_pool);
        bool ret = ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, spitch, dpitch, width, height, true);
        GGML_ASSERT(ret);
        ggml_vk_submit_sync(src->device, subctx);
    }
}

void ggml_vk_buffer_read(vk_buffer& src, size_t offset, void * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->buffer << ", " << offset << ", " << size << ")");
    ggml_vk_buffer_read_2d(src, offset, dst, size, size, size, 1);
}

void ggml_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    VkBufferCopy bc{ src_offset, dst_offset, size };

    vkCmdCopyBuffer(ctx->s->buffer->buf, (VkBuffer)src->buffer, (VkBuffer)dst->buffer, 1, &bc);
}

void ggml_vk_buffer_copy(vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    if (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_begin_sync_ctx(src->device, src->device->transfer_queue->cmd_pool);
        ggml_vk_buffer_copy_async(subctx, dst, dst_offset, src, src_offset, size);
        ggml_vk_submit_sync(src->device, subctx);
    } else {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_buffer(src->device, size);

        // Copy to src staging buffer
        ggml_vk_buffer_copy(src->device->sync_staging, 0, src, src_offset, size);

        // Copy to dst buffer
        ggml_vk_buffer_write(dst, dst_offset, src->device->sync_staging->info.pMappedData, size);
    }
}

void ggml_vk_buffer_memset_async(vk_context& ctx, vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << offset << ", " << c << ", " << size << ")");

    // Host memsets run at submit time, before earlier GPU work in the batch.
    // Only UMA may take this path, as upstream does. On ReBAR, an earlier
    // node that aliases the range would overwrite the memset.
    if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->info.pMappedData + offset, c, size, &ctx->memsets);
        return;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
}

void ggml_vk_buffer_memset(vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << offset << ", " << c << ", " << size << ")");

    if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->info.pMappedData + offset, c, size);
        return;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_begin_sync_ctx(dst->device, dst->device->transfer_queue->cmd_pool);
    subctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
    ggml_vk_submit_sync(dst->device, subctx);
}

static size_t ggml_vk_staging_size_n(const vk_device & device, const std::vector<vk_buffer_region> & regions) {
    size_t total = 0;
    size_t max   = 0;
    for (const auto & r : regions) {
        vk_buffer pinned;
        size_t pinned_offset;
        ggml_vk_host_get(device, r.host, pinned, pinned_offset);
        if (pinned != nullptr) {
            continue;
        }
        total += r.size;
        max = std::max(max, r.size);
    }
    return std::max(max, std::min(total, (size_t) device->suballocation_block_size));
}

// returns number of regions recorded, stops when staging is full
size_t ggml_vk_buffer_write_n_async(vk_context& subctx, const vk_buffer_region * regions, size_t n, vk_buffer * staging, size_t * staging_offset) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_n_async(" << n << ")");
    ggml_vk_sync_buffers(nullptr, subctx);

    size_t i = 0;
    for (; i < n; i++) {
        const vk_buffer_region & r = regions[i];

        vk_buffer pinned = nullptr;
        size_t pinned_offset = 0;
        ggml_vk_host_get(r.buf->device, r.host, pinned, pinned_offset);

        if (pinned != nullptr) {
            subctx->s->buffer->buf.copyBuffer(pinned->buffer, r.buf->buffer, { vk::BufferCopy{ pinned_offset, r.offset, r.size } });
            continue;
        }

        if (staging == nullptr || *staging_offset + r.size > (*staging)->size) {
            break;
        }

        subctx->s->buffer->buf.copyBuffer((*staging)->buffer, r.buf->buffer, { vk::BufferCopy{ *staging_offset, r.offset, r.size } });
        deferred_memcpy((uint8_t *) (*staging)->info.pMappedData + *staging_offset, r.host, r.size, &subctx->in_memcpys);
        *staging_offset += r.size;
    }
    return i;
}

// returns number of regions recorded, stops when staging is full
size_t ggml_vk_buffer_read_n_async(vk_context& subctx, const vk_buffer_region * regions, size_t n, vk_buffer * staging, size_t * staging_offset) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_n_async(" << n << ")");
    ggml_vk_sync_buffers(nullptr, subctx);

    size_t i = 0;
    for (; i < n; i++) {
        const vk_buffer_region & r = regions[i];

        vk_buffer pinned = nullptr;
        size_t pinned_offset = 0;
        ggml_vk_host_get(r.buf->device, r.host, pinned, pinned_offset);

        if (pinned != nullptr) {
            subctx->s->buffer->buf.copyBuffer(r.buf->buffer, pinned->buffer, { vk::BufferCopy{ r.offset, pinned_offset, r.size } });
            continue;
        }

        if (staging == nullptr || *staging_offset + r.size > (*staging)->size) {
            break;
        }

        subctx->s->buffer->buf.copyBuffer(r.buf->buffer, (*staging)->buffer, { vk::BufferCopy{ r.offset, *staging_offset, r.size } });
        deferred_memcpy(r.host, (const uint8_t *) (*staging)->info.pMappedData + *staging_offset, r.size, &subctx->out_memcpys);
        *staging_offset += r.size;
    }
    return i;
}

void ggml_vk_buffer_copy_n_async(vk_context& subctx, const vk_buffer_copy_region * regions, size_t n) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_n_async(" << n << ")");
    ggml_vk_sync_buffers(nullptr, subctx);

    for (size_t i = 0; i < n; i++) {
        const vk_buffer_copy_region & r = regions[i];
        GGML_ASSERT(r.src->device == r.dst->device);
        subctx->s->buffer->buf.copyBuffer(r.src->buffer, r.dst->buffer, { vk::BufferCopy{ r.src_offset, r.dst_offset, r.size } });
    }
}

void ggml_vk_buffer_write_n(const vk_buffer_region * regions, size_t n) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_n(" << n << ")");
    std::vector<vk_buffer_region> pending;

    for (size_t i = 0; i < n; i++) {
        const vk_buffer_region & r = regions[i];
        if (r.size == 0) {
            continue;
        }
        if (r.buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
            GGML_ASSERT(r.buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);
            memcpy((uint8_t *) r.buf->info.pMappedData + r.offset, r.host, r.size);
            continue;
        }
        GGML_ASSERT(pending.empty() || r.buf->device == pending[0].buf->device);
        pending.push_back(r);
    }

    if (pending.empty()) {
        return;
    }

    vk_device device = pending[0].buf->device;
    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    const size_t staging_size = ggml_vk_staging_size_n(device, pending);
    if (staging_size > 0) {
        ggml_vk_ensure_sync_staging_buffer(device, staging_size);
    }

    for (size_t i = 0; i < pending.size();) {
        vk_context subctx = ggml_vk_begin_sync_ctx(device, device->transfer_queue->cmd_pool);
        size_t staging_offset = 0;
        const size_t recorded = ggml_vk_buffer_write_n_async(subctx, pending.data() + i, pending.size() - i, &device->sync_staging, &staging_offset);
        GGML_ASSERT(recorded > 0);
        ggml_vk_submit_sync(device, subctx);
        i += recorded;
    }
}

void ggml_vk_buffer_read_n(const vk_buffer_region * regions, size_t n) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_n(" << n << ")");
    std::vector<vk_buffer_region> direct;
    std::vector<vk_buffer_region> pending;

    for (size_t i = 0; i < n; i++) {
        const vk_buffer_region & r = regions[i];
        if (r.size == 0) {
            continue;
        }
        GGML_ASSERT((direct.empty() && pending.empty()) || r.buf->device == (direct.empty() ? pending[0] : direct[0]).buf->device);
        if (ggml_vk_buffer_host_read_direct(r.buf)) {
            GGML_ASSERT(r.buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);
            direct.push_back(r);
        } else {
            pending.push_back(r);
        }
    }

    if (direct.empty() && pending.empty()) {
        return;
    }

    vk_device device = (direct.empty() ? pending[0] : direct[0]).buf->device;
    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    if (!direct.empty()) {
        vk_context subctx = ggml_vk_begin_sync_ctx(device, device->compute_queue->cmd_pool);
        subctx->s->buffer->buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost,
            {},
            { { vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eHostRead } },
            {}, {});
        ggml_vk_submit_sync(device, subctx);

        for (const auto & r : direct) {
            memcpy(r.host, (const uint8_t *) r.buf->info.pMappedData + r.offset, r.size);
        }
    }

    if (pending.empty()) {
        return;
    }

    const size_t staging_size = ggml_vk_staging_size_n(device, pending);
    if (staging_size > 0) {
        ggml_vk_ensure_sync_staging_buffer(device, staging_size);
    }

    for (size_t i = 0; i < pending.size();) {
        vk_context subctx = ggml_vk_begin_sync_ctx(device, device->transfer_queue->cmd_pool);
        size_t staging_offset = 0;
        const size_t recorded = ggml_vk_buffer_read_n_async(subctx, pending.data() + i, pending.size() - i, &device->sync_staging, &staging_offset);
        GGML_ASSERT(recorded > 0);
        ggml_vk_submit_sync(device, subctx);
        i += recorded;
    }
}

void ggml_vk_buffer_copy_n(const vk_buffer_copy_region * regions, size_t n) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_n(" << n << ")");
    std::vector<vk_buffer_copy_region> same_device;

    for (size_t i = 0; i < n; i++) {
        vk_buffer_copy_region r = regions[i];
        if (r.size == 0) {
            continue;
        }
        if (r.src->device != r.dst->device) {
            ggml_vk_buffer_copy(r.dst, r.dst_offset, r.src, r.src_offset, r.size);
            continue;
        }
        GGML_ASSERT(same_device.empty() || r.dst->device == same_device[0].dst->device);
        same_device.push_back(std::move(r));
    }

    if (same_device.empty()) {
        return;
    }

    vk_device device = same_device[0].dst->device;
    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    vk_context subctx = ggml_vk_begin_sync_ctx(device, device->transfer_queue->cmd_pool);
    ggml_vk_buffer_copy_n_async(subctx, same_device.data(), same_device.size());
    ggml_vk_submit_sync(device, subctx);
}

ggml_backend_buffer_i ggml_backend_vk_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_vk_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_vk_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_vk_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_buffer_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ ggml_backend_vk_buffer_set_tensor_n,
    /* .get_tensor_n    = */ ggml_backend_vk_buffer_get_tensor_n,
    /* .cpy_tensor_n    = */ ggml_backend_vk_buffer_cpy_tensor_n,
};

vk_buffer ggml_vk_buffer_from_host_ptr(vk_device & device, void * ptr, size_t size) {
    if (!device->external_memory_host) {
        return {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    if (uptr & (device->min_imported_host_pointer_alignment - 1)) {
        return {};
    }
    if (size & (device->min_imported_host_pointer_alignment - 1)) {
        return {};
    }

    // buffer_from_host_ptr capability is disabled; external memory import not supported with VMA
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    return {};
}

