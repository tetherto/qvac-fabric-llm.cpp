#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <vector>

struct htp_memory_count {
    uint64_t bytes = 0;
    bool valid = true;

    void add(uint64_t value) {
        if (value > std::numeric_limits<uint64_t>::max() - bytes) {
            valid = false;
        } else {
            bytes += value;
        }
    }

    void array(uint64_t count, uint64_t size) {
        if (size && count > std::numeric_limits<uint64_t>::max() / size) {
            valid = false;
        } else {
            add(count * size);
        }
    }

    void pages(uint64_t value, uint64_t page) {
        if (!page || value > std::numeric_limits<uint64_t>::max() - (page - 1)) {
            valid = false;
        } else {
            add((value + page - 1) / page * page);
        }
    }
};

class htp_runtime_activity {
    inline static std::shared_mutex mutex;
    inline static thread_local unsigned depth = 0;
    std::shared_lock<std::shared_mutex> lock;

public:
    htp_runtime_activity() : lock(mutex, std::defer_lock) {
        if (depth == 0) {
            lock.lock();
        }
        ++depth;
    }
    ~htp_runtime_activity() { --depth; }
    htp_runtime_activity(const htp_runtime_activity &) = delete;
    htp_runtime_activity & operator=(const htp_runtime_activity &) = delete;

    static bool active() { return depth != 0; }
    static std::unique_lock<std::shared_mutex> snapshot() {
        return std::unique_lock<std::shared_mutex>(mutex);
    }
};

template <typename Container>
inline void htp_runtime_count_hash(htp_memory_count & count, const Container & values) {
    count.array(values.bucket_count(), sizeof(void *));
    count.array(values.size(), sizeof(typename Container::value_type) + 2 * sizeof(void *));
}

template <typename Node>
inline void htp_runtime_count_nodes(htp_memory_count & count, const std::vector<Node> & nodes) {
    // fused/inputs/outputs are fixed-capacity htp_opvec members stored inline in Node.
    count.array(nodes.capacity(), sizeof(Node));
    for (const auto & node : nodes) {
        count.array(node.dummy.capacity(), sizeof(node.dummy[0]));
        count.array(node.dummy.size(), sizeof(*node.dummy[0]) + 2 * sizeof(void *));
    }
}

inline uint64_t htp_runtime_max_buffer_payload(uint64_t configured, uint64_t mmap_limit, uint64_t guard) {
    return mmap_limit >= guard ? std::min(configured, mmap_limit - guard) : 0;
}

inline void htp_runtime_future_metadata(htp_memory_count & count, uint64_t nodes,
        uint64_t batch, uint64_t depth, uint64_t node_size, uint64_t tensor_size) {
    constexpr uint64_t graph_copies = 4;
    constexpr uint64_t capacity_growth = 2;
    constexpr uint64_t node_pointers = 64;
    constexpr uint64_t node_name_bytes = 512;
    htp_memory_count per_node;
    per_node.add(node_size);
    per_node.array(node_pointers, sizeof(void *));
    per_node.array(2, tensor_size + 2 * sizeof(void *));
    per_node.add(node_name_bytes);
    if (!per_node.valid || per_node.bytes > UINT64_MAX / (graph_copies * capacity_growth)) {
        count.valid = false;
        return;
    }
    count.array(nodes, graph_copies * capacity_growth * per_node.bytes);
    htp_memory_count queued;
    queued.array(std::min(nodes, batch), capacity_growth * per_node.bytes);
    if (!queued.valid) {
        count.valid = false;
        return;
    }
    count.array(depth, queued.bytes);
}
