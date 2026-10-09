#include "llama-kv-cells.h"

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

int main() {
    test_scalar_shift_does_not_touch_ext();
    test_mrope_shift_updates_active_axes();
    test_mrope_image_grid_shift_preserves_relative_positions();
    test_mrope_image_grid_negative_shift_clears_cells();
    test_scalar_div_tracks_position_delta();
    test_mrope_negative_shift_clears_cell();

    return 0;
}
