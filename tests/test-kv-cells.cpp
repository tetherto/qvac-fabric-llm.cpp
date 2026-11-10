#include "llama-kv-cells.h"

#include <algorithm>
#include <random>
#include <vector>

static constexpr uint32_t CELL_0 = 0;
static constexpr llama_seq_id SEQ_0 = 0;

static constexpr uint32_t T_AXIS     = 0;
static constexpr uint32_t Y_AXIS     = 1;
static constexpr uint32_t X_AXIS     = 2;
static constexpr uint32_t OTHER_AXIS = 3;

static constexpr llama_pos EMPTY_POS = -1;

static constexpr llama_pos EXT_X = 7;
static constexpr llama_pos EXT_Y = 5;

static constexpr llama_pos TOKEN_POS = 10;
static constexpr llama_pos POS_SHIFT = -3;
static constexpr llama_pos SHIFTED_TOKEN_POS = TOKEN_POS + POS_SHIFT;

static void test_scalar_shift_does_not_touch_ext() {
    llama_kv_cells cells;
    cells.resize(1);

    llama_kv_cell_ext ext;
    ext.x = EXT_X;
    ext.y = EXT_Y;

    cells.pos_set(CELL_0, TOKEN_POS);
    cells.ext_set(CELL_0, ext);
    cells.seq_add(CELL_0, SEQ_0);

    const bool removed = cells.pos_add(CELL_0, POS_SHIFT);

    GGML_ASSERT(!removed);
    GGML_ASSERT(cells.pos_get(CELL_0) == SHIFTED_TOKEN_POS);
    GGML_ASSERT(cells.ext_get(CELL_0).x == EXT_X);
    GGML_ASSERT(cells.ext_get(CELL_0).y == EXT_Y);
    GGML_ASSERT(cells.get_shift(CELL_0, T_AXIS) == POS_SHIFT);
    GGML_ASSERT(cells.get_shift(CELL_0, Y_AXIS) == 0);
    GGML_ASSERT(cells.get_shift(CELL_0, X_AXIS) == 0);
    GGML_ASSERT(cells.get_shift(CELL_0, OTHER_AXIS) == 0);
}

static void test_mrope_shift_updates_active_axes() {
    llama_kv_cells cells;
    cells.resize(1);

    llama_kv_cell_ext ext;
    ext.x = EXT_X;
    ext.y = EXT_Y;

    cells.pos_set(CELL_0, TOKEN_POS);
    cells.ext_set(CELL_0, ext);
    cells.seq_add(CELL_0, SEQ_0);

    const bool removed = cells.pos_add(CELL_0, POS_SHIFT, true);

    GGML_ASSERT(!removed);
    GGML_ASSERT(cells.pos_get(CELL_0) == SHIFTED_TOKEN_POS);
    GGML_ASSERT(cells.ext_get(CELL_0).x == EXT_X + POS_SHIFT);
    GGML_ASSERT(cells.ext_get(CELL_0).y == EXT_Y + POS_SHIFT);
    GGML_ASSERT(cells.get_shift(CELL_0, T_AXIS) == POS_SHIFT);
    GGML_ASSERT(cells.get_shift(CELL_0, Y_AXIS) == POS_SHIFT);
    GGML_ASSERT(cells.get_shift(CELL_0, X_AXIS) == POS_SHIFT);
    GGML_ASSERT(cells.get_shift(CELL_0, OTHER_AXIS) == 0);
}

static void test_mrope_image_grid_shift_preserves_relative_positions() {
    static constexpr uint32_t IMAGE_GRID_WIDTH = 2;
    static constexpr uint32_t IMAGE_GRID_CELLS = IMAGE_GRID_WIDTH * IMAGE_GRID_WIDTH;
    static constexpr llama_pos IMAGE_ORIGIN_POS = 100;
    static constexpr llama_pos IMAGE_SHIFT = -20;

    llama_kv_cells cells;
    cells.resize(IMAGE_GRID_CELLS);

    for (uint32_t i = 0; i < cells.size(); ++i) {
        const llama_pos y = IMAGE_ORIGIN_POS + i/IMAGE_GRID_WIDTH;
        const llama_pos x = IMAGE_ORIGIN_POS + i%IMAGE_GRID_WIDTH;

        llama_kv_cell_ext ext;
        ext.x = x;
        ext.y = y;

        cells.pos_set(i, IMAGE_ORIGIN_POS);
        cells.ext_set(i, ext);
        cells.seq_add(i, SEQ_0);
    }

    GGML_ASSERT(cells.seq_pos_min(SEQ_0) == IMAGE_ORIGIN_POS);
    GGML_ASSERT(cells.seq_pos_max(SEQ_0) == IMAGE_ORIGIN_POS);

    for (uint32_t i = 0; i < cells.size(); ++i) {
        const bool removed = cells.pos_add(i, IMAGE_SHIFT, true);
        GGML_ASSERT(!removed);
    }

    GGML_ASSERT(cells.seq_pos_min(SEQ_0) == IMAGE_ORIGIN_POS + IMAGE_SHIFT);
    GGML_ASSERT(cells.seq_pos_max(SEQ_0) == IMAGE_ORIGIN_POS + IMAGE_SHIFT);

    for (uint32_t i = 0; i < cells.size(); ++i) {
        const llama_pos y = IMAGE_ORIGIN_POS + IMAGE_SHIFT + i/IMAGE_GRID_WIDTH;
        const llama_pos x = IMAGE_ORIGIN_POS + IMAGE_SHIFT + i%IMAGE_GRID_WIDTH;

        GGML_ASSERT(cells.pos_get(i) == IMAGE_ORIGIN_POS + IMAGE_SHIFT);
        GGML_ASSERT(cells.ext_get(i).x == x);
        GGML_ASSERT(cells.ext_get(i).y == y);
        GGML_ASSERT(cells.get_shift(i, T_AXIS) == IMAGE_SHIFT);
        GGML_ASSERT(cells.get_shift(i, Y_AXIS) == IMAGE_SHIFT);
        GGML_ASSERT(cells.get_shift(i, X_AXIS) == IMAGE_SHIFT);
        GGML_ASSERT(cells.get_shift(i, OTHER_AXIS) == 0);
    }

    GGML_ASSERT(cells.ext_get(1).x - cells.ext_get(0).x == 1);
    GGML_ASSERT(cells.ext_get(IMAGE_GRID_WIDTH).y - cells.ext_get(0).y == 1);
}

static void test_mrope_image_grid_negative_shift_clears_cells() {
    static constexpr uint32_t IMAGE_GRID_WIDTH = 2;
    static constexpr uint32_t IMAGE_GRID_CELLS = IMAGE_GRID_WIDTH * IMAGE_GRID_WIDTH;
    static constexpr llama_pos START_POS = 1;
    static constexpr llama_pos CLEARING_SHIFT = -2;

    llama_kv_cells cells;
    cells.resize(IMAGE_GRID_CELLS);

    for (uint32_t i = 0; i < cells.size(); ++i) {
        llama_kv_cell_ext ext;
        ext.x = START_POS + i%IMAGE_GRID_WIDTH;
        ext.y = START_POS + i/IMAGE_GRID_WIDTH;

        cells.pos_set(i, START_POS);
        cells.ext_set(i, ext);
        cells.seq_add(i, SEQ_0);
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        const bool removed = cells.pos_add(i, CLEARING_SHIFT, true);
        GGML_ASSERT(removed);
        GGML_ASSERT(cells.is_empty(i));
    }

    GGML_ASSERT(cells.get_used() == 0);
    GGML_ASSERT(cells.seq_pos_min(SEQ_0) == EMPTY_POS);
    GGML_ASSERT(cells.seq_pos_max(SEQ_0) == EMPTY_POS);
}

static void test_scalar_div_tracks_position_delta() {
    static constexpr llama_pos DIV_EXT_X = 9;
    static constexpr llama_pos DIV_EXT_Y = 5;
    static constexpr llama_pos DIV_TOKEN_POS = 11;
    static constexpr int DIVISOR = 2;

    llama_kv_cells cells;
    cells.resize(1);

    llama_kv_cell_ext ext;
    ext.x = DIV_EXT_X;
    ext.y = DIV_EXT_Y;

    cells.pos_set(CELL_0, DIV_TOKEN_POS);
    cells.ext_set(CELL_0, ext);
    cells.seq_add(CELL_0, SEQ_0);

    cells.pos_div(CELL_0, DIVISOR);

    GGML_ASSERT(cells.pos_get(CELL_0) == DIV_TOKEN_POS/DIVISOR);
    GGML_ASSERT(cells.ext_get(CELL_0).x == DIV_EXT_X);
    GGML_ASSERT(cells.ext_get(CELL_0).y == DIV_EXT_Y);
    GGML_ASSERT(cells.get_shift(CELL_0, T_AXIS) == DIV_TOKEN_POS - DIV_TOKEN_POS/DIVISOR);
    GGML_ASSERT(cells.get_shift(CELL_0, Y_AXIS) == 0);
    GGML_ASSERT(cells.get_shift(CELL_0, X_AXIS) == 0);
    GGML_ASSERT(cells.get_shift(CELL_0, OTHER_AXIS) == 0);
}

static void test_mrope_negative_shift_clears_cell() {
    static constexpr llama_pos START_POS = 1;
    static constexpr llama_pos START_EXT_X = 3;
    static constexpr llama_pos START_EXT_Y = 2;
    static constexpr llama_pos CLEARING_SHIFT = -2;

    llama_kv_cells cells;
    cells.resize(1);

    llama_kv_cell_ext ext;
    ext.x = START_EXT_X;
    ext.y = START_EXT_Y;

    cells.pos_set(CELL_0, START_POS);
    cells.ext_set(CELL_0, ext);
    cells.seq_add(CELL_0, SEQ_0);

    const bool removed = cells.pos_add(CELL_0, CLEARING_SHIFT, true);

    GGML_ASSERT(removed);
    GGML_ASSERT(cells.is_empty(CELL_0));
}

// the kv_rows lists come from per-sequence cell bitmaps that every cell update has to keep in sync,
// compare them with a scan of the cells after random updates
static void test_seq_cells_match_brute_force() {
    static constexpr uint32_t     N_CELLS = 300; // several 64-bit words, the last one partial
    static constexpr llama_seq_id N_SEQS  = 6;

    std::mt19937 rng(42);

    const auto rnd = [&](int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(rng);
    };

    llama_kv_cells cells;
    cells.resize(N_CELLS);

    // the mask lets slice attend to cell c
    const auto attends = [&](const llama_kv_rows_slice & slice, int32_t c) {
        if (cells.is_empty(c) || cells.pos_get(c) < slice.p_min) {
            return false;
        }
        for (const llama_seq_id t : slice.seq_ids) {
            if (cells.seq_has(c, t)) {
                return true;
            }
        }
        return false;
    };

    std::vector<int32_t> data;

    for (int step = 0; step < 20000; ++step) {
        const uint32_t     i = rnd(0, N_CELLS - 1);
        const llama_seq_id s = rnd(0, N_SEQS - 1);

        switch (rnd(0, 8)) {
            case 0:
            case 1:
                if (cells.is_empty(i)) {
                    cells.pos_set(i, rnd(0, 500));
                    cells.seq_add(i, s);
                }
                break;
            case 2:
                if (!cells.is_empty(i) && !cells.seq_has(i, s)) {
                    cells.seq_add(i, s);
                }
                break;
            case 3:
                if (!cells.is_empty(i) && cells.seq_has(i, s)) {
                    cells.seq_rm(i, s);
                }
                break;
            case 4:
                if (!cells.is_empty(i)) {
                    cells.seq_keep(i, s);
                }
                break;
            case 5:
                if (!cells.is_empty(i)) {
                    cells.pos_add(i, rnd(-50, 50), rnd(0, 1) == 1);
                }
                break;
            case 6:
                if (!cells.is_empty(i)) {
                    cells.pos_div(i, 2);
                }
                break;
            case 7:
                if (!cells.is_empty(i) && rnd(0, 3) == 0) {
                    cells.rm(i);
                }
                break;
            case 8:
                {
                    // save/restore of a range, as in state_read
                    cells.reset_shift();
                    const uint32_t n = rnd(1, 80);
                    const uint32_t j = rnd(0, N_CELLS - n);
                    const uint32_t k = rnd(0, N_CELLS - n);
                    cells.set(k, cells.cp(j, n));
                } break;
        }

        if (step == 10000) {
            cells.reset();
        }

        if (step % 97 != 0) {
            continue;
        }

        // random slices: a few sequences each, with or without the SWA cutoff
        std::vector<llama_kv_rows_slice> slices(rnd(1, 4));
        for (auto & slice : slices) {
            for (llama_seq_id t = 0; t < N_SEQS; ++t) {
                if (rnd(0, 2) == 0) {
                    slice.seq_ids.push_back(t);
                }
            }
            if (slice.seq_ids.empty()) {
                slice.seq_ids.push_back(s);
            }
            slice.p_min = rnd(0, 1) ? -1 : rnd(0, 400);
        }

        // the cells each slice attends to, in cell order
        std::vector<std::vector<int32_t>> listed(slices.size());
        size_t n_longest = 1;
        for (size_t is = 0; is < slices.size(); ++is) {
            for (uint32_t c = 0; c < N_CELLS; ++c) {
                if (attends(slices[is], c)) {
                    listed[is].push_back(c);
                }
            }
            n_longest = std::max(n_longest, listed[is].size());
        }

        // lists as long as the cache over a view of the whole cache, as long as the longest list over a view of the used
        // cells (both shorter in llama_kv_cache), and in between
        const uint32_t n_used = cells.used_max_p1();
        const std::pair<int64_t, uint32_t> shapes[] = {
            { N_CELLS, N_CELLS },
            { (int64_t) n_longest, n_used },
            { rnd((int) n_longest, N_CELLS), (uint32_t) rnd((int) n_used, N_CELLS) },
        };

        for (const auto & [n_per_seq, n_kv] : shapes) {
            data.assign(slices.size()*n_per_seq, -2);
            cells.get_seq_cells(slices, data.data(), n_per_seq, n_kv);

            for (size_t is = 0; is < slices.size(); ++is) {
                const int32_t * list = data.data() + is*n_per_seq;
                const size_t    n    = listed[is].size();

                GGML_ASSERT(std::equal(listed[is].begin(), listed[is].end(), list));

                for (int64_t k = n; k < n_per_seq; ++k) {
                    if (list[k] < 0) {
                        // -1 only once the view ends, then to the end of the list
                        GGML_ASSERT((k == 0 ? 0 : list[k - 1] + 1) >= (int32_t) n_kv);
                        GGML_ASSERT(std::all_of(list + k, list + n_per_seq, [](int32_t c) { return c == -1; }));
                        break;
                    }

                    // padding: in the K/V view, hidden by the mask, and consecutive with the entry before it
                    GGML_ASSERT(list[k] < (int32_t) n_kv);
                    GGML_ASSERT(!attends(slices[is], list[k]));
                    GGML_ASSERT(k == 0 || list[k] == list[k - 1] + 1);
                }
            }
        }

        for (llama_seq_id t = 0; t < N_SEQS; ++t) {
            const llama_pos pm = rnd(0, 400);

            uint32_t n_all = 0;
            uint32_t n_pm  = 0;
            for (uint32_t c = 0; c < N_CELLS; ++c) {
                if (!cells.is_empty(c) && cells.seq_has(c, t)) {
                    n_all += 1;
                    n_pm  += cells.pos_get(c) >= pm;
                }
            }

            GGML_ASSERT(cells.seq_n_cells(t)     == n_all);
            GGML_ASSERT(cells.seq_n_cells(t, pm) == n_pm);
        }
    }
}

int main() {
    test_scalar_shift_does_not_touch_ext();
    test_mrope_shift_updates_active_axes();
    test_mrope_image_grid_shift_preserves_relative_positions();
    test_mrope_image_grid_negative_shift_clears_cells();
    test_scalar_div_tracks_position_delta();
    test_mrope_negative_shift_clears_cell();
    test_seq_cells_match_brute_force();

    return 0;
}
