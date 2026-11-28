// Register-prefetch loaders (MM_PREFETCH variants of mul_mm.comp): fetch_* load a K tile into registers, store_*_raw dequantise
// it into LDS. pos_a is in LOAD_VEC_A units and advances by BK / LOAD_VEC_A per K step (BK = 32).

#if !defined(COOPMAT) || !defined(FLOAT16) || defined(MUL_MAT_ID)
#error "MM_PREFETCH variants are dense fp16 KHR coopmat only"
#endif

// The q5_K qh dword and the iq4_xs d / scale words serve a whole superblock, so the prefetch loop loads them once per
// superblock. a_fetch_all forces full loads for the prologue and for the non-prefetch K loops.
bool a_fetch_all = true;

#if defined(DATA_A_Q6_K) && LOAD_VEC_A == 8
// 210-byte blocks are only 2-byte aligned: fetch8 reads the 8 bytes at b as two dwords and a word, all inside [b - 2, b + 8)
// (a group at the start of an odd block reads the last 2 bytes of the previous block). A lane covers 8 consecutive k.
uvec3 fetch8(const uint b) {
    const uint b4 = b & ~3u;
    return uvec3(data_a_u32[b4 / 4], data_a_u32[b4 / 4 + 1], uint(data_a_u16[(b + 6) / 2]));
}
uvec2 unpack8b(const uvec3 r, const bool odd) {
    return odd ? uvec2((r.x >> 16) | (r.y << 16), (r.y >> 16) | (r.z << 16)) : r.xy;
}
#define A_RAW_T uvec4[2]   // [0] = ql fetch8, qh.x; [1] = qh.yz, d bits | scale << 16, 0
// byte offset of the lane's row block plus its 8-byte group
uint a_lane_off(const uint row, const uint col) { return (col * (p.stride_a / QUANT_K)) * 210 + 8 * row; }
void fetch_a(const uint pos_a, const uint row, const uint col, const uint lane_off, out A_RAW_T raw) {
    const uint g = (pos_a % 32) / 4;       // 32-k group in the block
    const uint n = g / 4;
    const uint m = g % 4;
    const uint base = (pos_a / 32) * 210 + lane_off - 8 * row;
    const uint ib  = base / 210;
    const uint bql = base + 64 * n + 32 * (m & 1) + 8 * row;
    const uint bqh = base + 128 + 32 * n + 8 * row;
    const uint is  = 8 * n + 2 * m + row / 2;
    const uvec3 ql = fetch8(bql);
    const uvec3 qh = fetch8(bqh);
    raw[0] = uvec4(ql.x, ql.y, ql.z, qh.x);
    raw[1] = uvec4(qh.y, qh.z, uint(float16BitsToUint16(data_a[ib].d)) | (uint(uint8_t(data_a[ib].scales[is])) << 16), 0);
}
void store_a_raw(const uint pos_a, const uint row, const uint col, const uint sidx, const A_RAW_T raw) {
    const uint m = ((pos_a % 32) / 4) % 4;
    const uint base = (pos_a / 32) * 210 + (col * (p.stride_a / QUANT_K)) * 210;
    const bool odd = (base & 2) != 0;
    const uvec2 ql = unpack8b(raw[0].xyz, odd);
    const uvec2 qh = unpack8b(uvec3(raw[0].w, raw[1].x, raw[1].y), odd);
    const int   sc = int(int8_t(uint8_t(raw[1].z >> 16)));
    const float d  = float(uint16BitsToFloat16(uint16_t(raw[1].z & 0xFFFF)));
    const float dscale = d * float(sc);
    const uint nib = 4 * (m >> 1);
    const uint hsh = 2 * m;
    const uint q0 = ((ql.x >> nib) & 0x0F0F0F0F) | (((qh.x >> hsh) & 0x03030303) << 4);
    const uint q1 = ((ql.y >> nib) & 0x0F0F0F0F) | (((qh.y >> hsh) & 0x03030303) << 4);
    const vec4 v0 = (vec4(unpack8(q0)) - 32.0f) * dscale;
    const vec4 v1 = (vec4(unpack8(q1)) - 32.0f) * dscale;
    buf_a[sidx]     = FLOAT_TYPEV2(v0.xy);
    buf_a[sidx + 1] = FLOAT_TYPEV2(v0.zw);
    buf_a[sidx + 2] = FLOAT_TYPEV2(v1.xy);
    buf_a[sidx + 3] = FLOAT_TYPEV2(v1.zw);
}

#elif defined(DATA_A_IQ4_XS) && LOAD_VEC_A == 8
// 136-byte superblock: d | scales_h (dword 0), scales_l (dword 1), qs = 8 sub-blocks of 16 bytes (low nibbles k 0..15, high
// nibbles k 16..31). Lane row covers k 8 * row .. 8 * row + 7 of the sub-block; every field is dword aligned.
#define A_RAW_T uvec4   // xy: the lane's 8 qs bytes, z: d | scales_h << 16, w: scales_l
uint a_lane_off(const uint row, const uint col) { return (col * (p.stride_a / QUANT_K)) * 136 + 8 * (row % 2); }
void fetch_a(const uint pos_a, const uint row, const uint col, const uint lane_off, inout A_RAW_T raw) {
    const uint sb   = (pos_a / 32) * 136 + lane_off - 8 * (row % 2);
    const uint ib32 = (pos_a / 4) % 8;
    const uint q    = (sb + 8 + 16 * ib32 + 8 * (row % 2)) / 4;
    raw.xy = uvec2(data_a_u32[q], data_a_u32[q + 1]);
    if (a_fetch_all || ib32 == 0) {
        raw.zw = uvec2(data_a_u32[sb / 4], data_a_u32[sb / 4 + 1]);
    }
}
void store_a_raw(const uint pos_a, const uint row, const uint col, const uint sidx, const A_RAW_T raw) {
    const uint ib32 = (pos_a / 4) % 8;
    const uint sl = (raw.w >> (4 * ib32)) & 0xF;
    const uint sh = (raw.z >> (16 + 2 * ib32)) & 3;
    const float d = unpackHalf2x16(raw.z).x * float(int(sl | (sh << 4)) - 32);
    const uint nib = 4 * (row / 2);
    const uint q0 = (raw.x >> nib) & 0x0F0F0F0F;
    const uint q1 = (raw.y >> nib) & 0x0F0F0F0F;
    const vec4 v0 = d * vec4(float(kvalues_iq4nl[q0 & 0xFF]), float(kvalues_iq4nl[(q0 >> 8) & 0xFF]), float(kvalues_iq4nl[(q0 >> 16) & 0xFF]), float(kvalues_iq4nl[q0 >> 24]));
    const vec4 v1 = d * vec4(float(kvalues_iq4nl[q1 & 0xFF]), float(kvalues_iq4nl[(q1 >> 8) & 0xFF]), float(kvalues_iq4nl[(q1 >> 16) & 0xFF]), float(kvalues_iq4nl[q1 >> 24]));
    buf_a[sidx]     = FLOAT_TYPEV2(v0.xy);
    buf_a[sidx + 1] = FLOAT_TYPEV2(v0.zw);
    buf_a[sidx + 2] = FLOAT_TYPEV2(v1.xy);
    buf_a[sidx + 3] = FLOAT_TYPEV2(v1.zw);
}

#elif (defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)) && LOAD_VEC_A == 4
// Lane row covers k 4 * row .. 4 * row + 3 of a 32-k sub-block. The (d, m) pair of each sub-block comes from scache_dm,
// which scache_refill decodes once per superblock for every row of the tile.
#if defined(DATA_A_Q4_K)
#define A_RAW_T uint
#define A_BLOCK_DWORDS 36   // dm (1), scales (3), qs (32)
#define A_QS_DWORD 4
#else
#define A_RAW_T uvec2       // x: qs dword, y: qh dword
#define A_BLOCK_DWORDS 44   // dm (1), scales (3), qh (8), qs (32)
#define A_QH_DWORD 4
#define A_QS_DWORD 12
#endif
uint a_lane_off(const uint row, const uint col) { return (col * (p.stride_a / QUANT_K)) * A_BLOCK_DWORDS + row; }
void fetch_a(const uint pos_a, const uint row, const uint col, const uint lane_off, inout A_RAW_T raw) {
    const uint kb  = pos_a / 64;
    const uint sub = (pos_a % 64) / 8;
#if defined(DATA_A_Q4_K)
    raw = data_a_u32[kb * A_BLOCK_DWORDS + lane_off + A_QS_DWORD + 8 * (sub / 2)];
#else
    raw.x = data_a_u32[kb * A_BLOCK_DWORDS + lane_off + A_QS_DWORD + 8 * (sub / 2)];
    // bit sub of the lane's qh dword is the fifth bit of sub-block sub
    if (a_fetch_all || sub == 0) {
        raw.y = data_a_u32[kb * A_BLOCK_DWORDS + lane_off + A_QH_DWORD];
    }
#endif
}
void store_a_raw(const uint pos_a, const uint row, const uint col, const uint sidx, const A_RAW_T raw) {
    const uint sub = (pos_a % 64) / 8;
    const vec2 dm = vec2(scache_dm[col * 8 + sub]);
#if defined(DATA_A_Q4_K)
    const vec4 q = vec4(unpack8((raw >> (4 * (sub % 2))) & 0x0F0F0F0F));
#else
    const uint qs = (raw.x >> (4 * (sub % 2))) & 0x0F0F0F0F;
    const uint qh = ((raw.y >> sub) & 0x01010101) << 4;
    const vec4 q = vec4(unpack8(qs | qh));
#endif
    buf_a[sidx]     = FLOAT_TYPEV2(fma(dm.x, q.x, dm.y), fma(dm.x, q.y, dm.y));
    buf_a[sidx + 1] = FLOAT_TYPEV2(fma(dm.x, q.z, dm.y), fma(dm.x, q.w, dm.y));
}

// Decode the 8 (d * scale, -dmin * min) pairs of every tile row at each superblock boundary (pos_a is at the boundary).
void scache_refill(const uint pos_a, const uint block) {
    if (block % QUANT_K != 0) {
        return;
    }
    for (uint x = gl_LocalInvocationID.x; x < BM * 8; x += gl_WorkGroupSize.x) {
        const uint r  = x / 8;
        const uint is = x % 8;
        const uint ib = (pos_a + r * (p.stride_a / LOAD_VEC_A)) / 64;

        const vec2 loadd = vec2(data_a[ib].dm);
        const uvec3 scales = uvec3(data_a_packed32[ib].scales[0], data_a_packed32[ib].scales[1], data_a_packed32[ib].scales[2]);
        const uint scalesoffs = (is & 3) * 8;
        const uint scidx0 = (is < 4) ? 0 : 2;
        const uint scidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
        const uint mbidx0 = (is < 4) ? 1 : 2;
        const uint mbidxshift0 = (is < 4) ? scalesoffs : scalesoffs + 4;
        const uint mbidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
        const uint sc    = ((scales[scidx0] >> scalesoffs) & 0xF) | ((scales[0] >> scidxshift1) & 0x30);
        const uint mbyte = ((scales[mbidx0] >> mbidxshift0) & 0xF) | ((scales[1] >> mbidxshift1) & 0x30);

        scache_dm[x] = SCACHE_T(loadd.x * sc, -loadd.y * mbyte);
    }
    barrier();
}
#define SCACHE_REFILL(pos, blk) scache_refill(pos, blk);

#else
#error "MM_PREFETCH: no prefetch loader for this A type and LOAD_VEC_A"
#endif

void load_a_to_shmem(const uint pos_a, const uint row, const uint col, const uint idx_m, const bool bounds_check, const uint block, const uint end_k) {
    A_RAW_T raw;
    fetch_a(pos_a, row, col, a_lane_off(row, col), raw);
    store_a_raw(pos_a, row, col, a_shmem_index(col, row * LOAD_VEC_A / 2), raw);
}

#if LOAD_VEC_B != 8
#error "MM_PREFETCH: the B prefetch loads 8 values per lane"
#endif
void fetch_b(const uint pos_b, const uint row, const uint col, out B_TYPE raw) {
    raw = data_b[pos_b + col * p.stride_b / LOAD_VEC_B + row];
}
void store_b_raw(const uint buf_idx, const B_TYPE raw) {
    const FLOAT_TYPEV8 bb = FLOAT_TYPEV8(raw);
    buf_b[buf_idx + 0] = bb[0].xy;
    buf_b[buf_idx + 1] = bb[0].zw;
    buf_b[buf_idx + 2] = bb[1].xy;
    buf_b[buf_idx + 3] = bb[1].zw;
}
