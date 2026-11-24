#pragma once

#include <cstddef>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

namespace tts_cpp::detail {

// Shape-keyed results that keep only the most recently added `capacity`
// entries: a decoder kept loaded sees a new length on almost every reply.
// An evicted entry stays valid for callers still holding it. Not thread-safe.
template <typename Key>
class shape_result_cache {
public:
    using entry = std::shared_ptr<const std::vector<float>>;

    explicit shape_result_cache(size_t capacity) : capacity_(capacity) {}

    entry find(const Key & key) const {
        const auto it = entries_.find(key);
        return it == entries_.end() ? nullptr : it->second;
    }

    entry insert(const Key & key, std::vector<float> values) {
        if (entry existing = find(key)) {
            return existing;
        }
        entry added = std::make_shared<const std::vector<float>>(std::move(values));
        entries_.emplace(key, added);
        order_.push_back(key);
        if (order_.size() > capacity_) {
            entries_.erase(order_.front());
            order_.pop_front();
        }
        return added;
    }

    size_t size() const { return entries_.size(); }

    void clear() {
        entries_.clear();
        order_.clear();
    }

private:
    size_t capacity_;
    std::unordered_map<Key, entry> entries_;
    std::deque<Key> order_;
};

} // namespace tts_cpp::detail
