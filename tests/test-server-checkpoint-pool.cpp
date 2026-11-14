#include "server-task.h"

#include <cstdio>
#include <iterator>
#include <list>

static constexpr size_t   TGT_BYTES   = 4096;
static constexpr size_t   DFT_BYTES   = 512;
static constexpr size_t   SPEC_BYTES  = 16;
static constexpr int      ID_TASK_OLD = 7;
static constexpr int64_t  N_TOKENS    = 100;

using checkpoint_list = std::list<common_prompt_checkpoint>;

static void fill_checkpoint(common_prompt_checkpoint & ckpt) {
    ckpt.id_task = ID_TASK_OLD;
    ckpt.update_pos(N_TOKENS, 0, (llama_pos) N_TOKENS - 1);
    ckpt.data_tgt.assign(TGT_BYTES, 1);
    ckpt.data_dft.assign(DFT_BYTES, 2);
    ckpt.data_spec.assign(SPEC_BYTES, 3);
}

// discard() moves the node out of the list and returns the next element
static bool check_discard_keeps_order() {
    server_checkpoint_pool pool;
    checkpoint_list checkpoints(3);
    auto second = std::next(checkpoints.begin());

    auto next = pool.discard(checkpoints, checkpoints.begin());

    return next == second && checkpoints.size() == 2 && pool.spare.size() == 1;
}

// a checkpoint added after a discard reuses its memory and drops its optional speculative state
static bool check_add_reuses_storage() {
    server_checkpoint_pool pool;
    checkpoint_list checkpoints(1);
    fill_checkpoint(checkpoints.front());
    const uint8_t * tgt = checkpoints.front().data_tgt.data();
    const uint8_t * dft = checkpoints.front().data_dft.data();

    pool.discard(checkpoints, checkpoints.begin());
    auto & cur = pool.add(checkpoints);

    return checkpoints.size() == 1 && &cur == &checkpoints.back() && pool.spare.empty() &&
        cur.data_tgt.data() == tgt && cur.data_tgt.size() == TGT_BYTES &&
        cur.data_dft.data() == dft && cur.data_dft.size() == DFT_BYTES &&
        cur.data_spec.empty() && cur.id_task == -1;
}

static bool check_add_without_spare() {
    server_checkpoint_pool pool;
    checkpoint_list checkpoints;

    auto & cur = pool.add(checkpoints);

    return checkpoints.size() == 1 && cur.empty() && cur.data_spec.empty() && cur.id_task == -1;
}

static bool check_clear_releases_spare() {
    server_checkpoint_pool pool;
    checkpoint_list checkpoints(1);
    fill_checkpoint(checkpoints.front());

    pool.discard(checkpoints, checkpoints.begin());
    pool.clear();

    return pool.spare.empty() && pool.add(checkpoints).empty();
}

int main() {
    const struct {
        bool (*fn)();
        const char * name;
    } checks[] = {
        { check_discard_keeps_order,  "discard returns the next checkpoint" },
        { check_add_reuses_storage,   "add reuses discarded storage without stale speculative state" },
        { check_add_without_spare,    "add creates a fresh checkpoint when the pool is empty" },
        { check_clear_releases_spare, "clear releases the spare checkpoints" },
    };

    int failures = 0;
    for (const auto & check : checks) {
        if (!check.fn()) {
            fprintf(stderr, "FAIL: %s\n", check.name);
            ++failures;
        }
    }

    return failures == 0 ? 0 : 1;
}
