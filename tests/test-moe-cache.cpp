#include "llama-moe-cache.h"

#include <vector>

struct test_slot_owner {
    int32_t layer = -1;
    int32_t expert = -1;
};

static void apply_fills(
        std::vector<test_slot_owner> & owners,
        const std::vector<llama_moe_cache_lru_fill> & fills) {
    for (const llama_moe_cache_lru_fill & fill : fills) {
        GGML_ASSERT(fill.slot >= 0);
        GGML_ASSERT(size_t(fill.slot) < owners.size());
        owners[fill.slot] = { fill.layer, fill.expert };
    }
}

static void check_owners(
        int32_t layer,
        const int32_t * ids,
        size_t n_ids,
        const int32_t * remapped_ids,
        const std::vector<test_slot_owner> & owners) {
    for (size_t i = 0; i < n_ids; ++i) {
        GGML_ASSERT(remapped_ids[i] >= 0);
        GGML_ASSERT(size_t(remapped_ids[i]) < owners.size());
        GGML_ASSERT(owners[remapped_ids[i]].layer == layer);
        GGML_ASSERT(owners[remapped_ids[i]].expert == ids[i]);
    }
}

static void test_cold_and_hit() {
    llama_moe_cache_lru lru(2, 8, 4);
    std::vector<test_slot_owner> owners(lru.capacity());
    const int32_t ids[] = { 1, 2 };
    int32_t remapped[2];
    std::vector<llama_moe_cache_lru_fill> fills;
    llama_moe_cache_lru_stats stats;

    GGML_ASSERT(lru.plan(0, ids, 2, remapped, fills, stats));
    GGML_ASSERT(fills.size() == 2);
    GGML_ASSERT(stats.hits == 0);
    GGML_ASSERT(stats.misses == 2);
    GGML_ASSERT(stats.evictions == 0);
    GGML_ASSERT(remapped[0] != remapped[1]);
    apply_fills(owners, fills);
    check_owners(0, ids, 2, remapped, owners);

    GGML_ASSERT(lru.plan(0, ids, 2, remapped, fills, stats));
    GGML_ASSERT(fills.empty());
    GGML_ASSERT(stats.hits == 2);
    GGML_ASSERT(stats.misses == 0);
    GGML_ASSERT(stats.evictions == 0);
    check_owners(0, ids, 2, remapped, owners);
}

static void test_duplicate_ids() {
    llama_moe_cache_lru lru(1, 8, 3);
    const int32_t ids[] = { 4, 4, 6 };
    int32_t remapped[3];
    std::vector<llama_moe_cache_lru_fill> fills;
    llama_moe_cache_lru_stats stats;

    GGML_ASSERT(lru.plan(0, ids, 3, remapped, fills, stats));
    GGML_ASSERT(fills.size() == 2);
    GGML_ASSERT(stats.hits == 0);
    GGML_ASSERT(stats.misses == 2);
    GGML_ASSERT(stats.evictions == 0);
    GGML_ASSERT(remapped[0] == remapped[1]);
    GGML_ASSERT(remapped[0] != remapped[2]);
}

static void test_global_lru() {
    llama_moe_cache_lru lru(2, 8, 2);
    const int32_t first[] = { 1, 2 };
    std::vector<test_slot_owner> owners(lru.capacity());
    const int32_t second[] = { 3 };
    const int32_t hit[] = { 2 };
    int32_t remapped[2];
    std::vector<llama_moe_cache_lru_fill> fills;
    llama_moe_cache_lru_stats stats;

    GGML_ASSERT(lru.plan(0, first, 2, remapped, fills, stats));
    apply_fills(owners, fills);
    check_owners(0, first, 2, remapped, owners);
    GGML_ASSERT(lru.plan(1, second, 1, remapped, fills, stats));
    apply_fills(owners, fills);
    check_owners(1, second, 1, remapped, owners);
    GGML_ASSERT(stats.hits == 0);
    GGML_ASSERT(stats.misses == 1);
    GGML_ASSERT(stats.evictions == 1);

    GGML_ASSERT(lru.plan(0, hit, 1, remapped, fills, stats));
    check_owners(0, hit, 1, remapped, owners);
    GGML_ASSERT(stats.hits == 1);
    GGML_ASSERT(stats.misses == 0);
    GGML_ASSERT(stats.evictions == 0);
}

static void test_current_plan_pinning() {
    llama_moe_cache_lru lru(1, 8, 2);
    const int32_t first[] = { 0, 1 };
    std::vector<test_slot_owner> owners(lru.capacity());
    const int32_t second[] = { 1, 2 };
    int32_t remapped[2];
    std::vector<llama_moe_cache_lru_fill> fills;
    llama_moe_cache_lru_stats stats;

    GGML_ASSERT(lru.plan(0, first, 2, remapped, fills, stats));
    const int32_t slot_one = remapped[1];
    apply_fills(owners, fills);
    check_owners(0, first, 2, remapped, owners);

    GGML_ASSERT(lru.plan(0, second, 2, remapped, fills, stats));
    apply_fills(owners, fills);
    GGML_ASSERT(stats.hits == 1);
    GGML_ASSERT(stats.misses == 1);
    GGML_ASSERT(stats.evictions == 1);
    GGML_ASSERT(remapped[0] == slot_one);
    GGML_ASSERT(remapped[0] != remapped[1]);
    check_owners(0, second, 2, remapped, owners);
}

static void test_capacity() {
    llama_moe_cache_lru lru(1, 8, 2);
    const int32_t too_many[] = { 1, 2, 3 };
    int32_t remapped[3];
    std::vector<llama_moe_cache_lru_fill> fills;
    llama_moe_cache_lru_stats stats;

    GGML_ASSERT(!lru.plan(0, too_many, 3, remapped, fills, stats));
}

int main() {
    test_cold_and_hit();
    test_duplicate_ids();
    test_global_lru();
    test_current_plan_pinning();
    test_capacity();
    return 0;
}
