// Top-k indices of f32 rows: each work-group of the first pass keeps the k largest values of one chunk of a row,
// one work-group per row then keeps the k largest of those candidates. TOP_K_WG_SIZE comes from the host.

#define TOP_K_NONE -1

// a beats b: larger value, or equal value at a smaller index; TOP_K_NONE marks no entry
inline bool top_k_beats(float av, int ai, float bv, int bi) {
    return ai != TOP_K_NONE && (bi == TOP_K_NONE || av > bv || (av == bv && ai < bi));
}

inline bool top_k_taken(local const uint * taken, int p) {
    return (taken[p >> 5] >> (p & 31)) & 1u;
}

inline void top_k_clear(local uint * taken, int n, int lid) {
    for (int w = lid; w < (n + 31) / 32; w += TOP_K_WG_SIZE) {
        taken[w] = 0u;
    }
}

// Best entry not taken yet among this work-item's positions; entry p has index ids[p], or id0 + p without ids.
inline void top_k_scan(global const float * v, global const int * ids, int id0, int n, local const uint * taken,
                       int lid, float * best_v, int * best_i, int * best_p) {
    *best_v = 0.0f;
    *best_i = TOP_K_NONE;
    *best_p = TOP_K_NONE;
    for (int p = lid; p < n; p += TOP_K_WG_SIZE) {
        const int id = ids ? ids[p] : id0 + p;
        if (!top_k_taken(taken, p) && top_k_beats(v[p], id, *best_v, *best_i)) {
            *best_v = v[p];
            *best_i = id;
            *best_p = p;
        }
    }
}

// Tree reduction of the per-item bests; the work-group winner ends in slot 0.
inline void top_k_reduce(local float * red_v, local int * red_i, local int * red_p, int lid) {
    for (int s = TOP_K_WG_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s && top_k_beats(red_v[lid + s], red_i[lid + s], red_v[lid], red_i[lid])) {
            red_v[lid] = red_v[lid + s];
            red_i[lid] = red_i[lid + s];
            red_p[lid] = red_p[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// k rounds, each taking the largest entry not taken yet; writes its value (when out_v is set) and index.
inline void top_k_select(global const float * v, global const int * ids, int id0, int n, int k,
                         global float * out_v, global int * out_i, local uint * taken,
                         local float * red_v, local int * red_i, local int * red_p) {
    const int lid = get_local_id(0);
    top_k_clear(taken, n, lid);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int r = 0; r < k; ++r) {
        float best_v;
        int   best_i, best_p;
        top_k_scan(v, ids, id0, n, taken, lid, &best_v, &best_i, &best_p);
        red_v[lid] = best_v;
        red_i[lid] = best_i;
        red_p[lid] = best_p;
        barrier(CLK_LOCAL_MEM_FENCE);
        top_k_reduce(red_v, red_i, red_p, lid);
        if (lid == 0) {
            if (red_p[0] != TOP_K_NONE) {
                taken[red_p[0] >> 5] |= 1u << (red_p[0] & 31);
            }
            if (out_v) {
                out_v[r] = red_v[0];
            }
            out_i[r] = red_i[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

kernel void kernel_top_k_f32_partial(
        global const char * src0,
        ulong                offset0,
        ulong                nb01,
        int                  ne00,
        int                  k,
        int                  chunk,
        global float       * cand_value,
        global int         * cand_index,
        local uint         * taken) {
    local float red_v[TOP_K_WG_SIZE];
    local int   red_i[TOP_K_WG_SIZE];
    local int   red_p[TOP_K_WG_SIZE];

    const int ichunk = get_group_id(0);
    const int row    = get_group_id(1);
    const int c0     = ichunk * chunk;
    const int out0   = (row * (int) get_num_groups(0) + ichunk) * k;

    global const float * x = (global const float *)(src0 + offset0 + row * nb01) + c0;
    top_k_select(x, 0, c0, min(chunk, ne00 - c0), k, cand_value + out0, cand_index + out0, taken, red_v, red_i, red_p);
}

kernel void kernel_top_k_f32_merge(
        global const float * cand_value,
        global const int   * cand_index,
        int                  n_cand,
        int                  k,
        global char        * dst,
        ulong                offsetd,
        ulong                nb1,
        local uint         * taken) {
    local float red_v[TOP_K_WG_SIZE];
    local int   red_i[TOP_K_WG_SIZE];
    local int   red_p[TOP_K_WG_SIZE];

    const int row = get_group_id(1);

    global int * out = (global int *)(dst + offsetd + row * nb1);
    top_k_select(cand_value + row * n_cand, cand_index + row * n_cand, 0, n_cand, k, 0, out, taken, red_v, red_i, red_p);
}
