#pragma once

// Adapted from lukdmine/llama.cpp gdn-fused-v2, b6ceb2f6703ed70c3ddf62330b3f57aeae487207.
// Each block carries a 32-value slice of one head through 16-token chunks.
// BF16 high/residual products and an FP32 inverse solve preserve recurrent precision.
// Explicit block barriers order shared-buffer reuse.
namespace gdn_value_split {

constexpr int CHUNK     = 16;   // tokens per chunk = one mma M tile
constexpr int DV_TILE   = 32;   // state columns per block
constexpr int DD        = 128;  // head dim; the fragment geometry is specialised to it
constexpr int NWARPS    = 8;
constexpr int NTHREADS  = NWARPS * 32;
constexpr int MINBLOCKS = 1;

using bf16                    = __nv_bfloat16;
using frag                    = uint2;
// Every operand has a high plane and a residual plane at this fixed element
// offset. High planes occupy 14208 BF16 elements; reserve padding to 14336.
constexpr int    LO_OFFSET    = 14336;
constexpr size_t SHARED_BYTES = 2 * LO_OFFSET * sizeof(bf16);

__device__ __forceinline__ void store1(bf16 * p, float x) {
    const bf16 hi = __float2bfloat16_rn(x);
    *p            = hi;
    p[LO_OFFSET]  = __float2bfloat16_rn(x - __bfloat162float(hi));
}

__device__ __forceinline__ void store2(bf16 * p, float2 x) {
    auto   hi                                          = __float22bfloat162_rn(x);
    float2 h                                           = __bfloat1622float2(hi);
    auto   lo                                          = __float22bfloat162_rn(make_float2(x.x - h.x, x.y - h.y));
    *reinterpret_cast<__nv_bfloat162 *>(p)             = hi;
    *reinterpret_cast<__nv_bfloat162 *>(p + LO_OFFSET) = lo;
}

__device__ __forceinline__ void
rawmma(float c[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void raw_ldsm4(const void * p, unsigned & r0, unsigned & r1, unsigned & r2, unsigned & r3) {
    unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a)
                 : "memory");
}

__device__ __forceinline__ void raw_ldsm2(const void * p, unsigned & r0, unsigned & r1) {
    unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a) : "memory");
}

__device__ __forceinline__ void raw_ldsm2t(const void * p, unsigned & r0, unsigned & r1) {
    unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(a)
                 : "memory");
}

__device__ __forceinline__ void raw_ldsm4t(const void * p, unsigned & r0, unsigned & r1, unsigned & r2, unsigned & r3) {
    unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a)
                 : "memory");
}

__device__ __forceinline__ void mma16816(float c[4], frag a0, frag a1, frag a2, frag a3, frag b0, frag b1) {
    // Add cross-products before the main product; omit the second-order low*low
    // term.
    rawmma(c, a0.y, a1.y, a2.y, a3.y, b0.x, b1.x);
    rawmma(c, a0.x, a1.x, a2.x, a3.x, b0.y, b1.y);
    rawmma(c, a0.x, a1.x, a2.x, a3.x, b0.x, b1.x);
}

__device__ __forceinline__ void ldsm4(const bf16 * p, frag & r0, frag & r1, frag & r2, frag & r3) {
    raw_ldsm4(p, r0.x, r1.x, r2.x, r3.x);
    raw_ldsm4(p + LO_OFFSET, r0.y, r1.y, r2.y, r3.y);
}

__device__ __forceinline__ void ldsm2(const bf16 * p, frag & r0, frag & r1) {
    raw_ldsm2(p, r0.x, r1.x);
    raw_ldsm2(p + LO_OFFSET, r0.y, r1.y);
}

__device__ __forceinline__ void ldsm2t(const bf16 * p, frag & r0, frag & r1) {
    raw_ldsm2t(p, r0.x, r1.x);
    raw_ldsm2t(p + LO_OFFSET, r0.y, r1.y);
}

__device__ __forceinline__ void ldsm4t(const bf16 * p, frag & r0, frag & r1, frag & r2, frag & r3) {
    raw_ldsm4t(p, r0.x, r1.x, r2.x, r3.x);
    raw_ldsm4t(p + LO_OFFSET, r0.y, r1.y, r2.y, r3.y);
}

// Bank-conflict swizzle of the 16-byte column unit within a row. umask = row
// width in 16B units minus one (7 for 256 B rows, 3 for 64 B rows).
__device__ __forceinline__ int swz(int r, int c16, int umask) {
    return c16 ^ (r & umask);
}

static __global__ void __launch_bounds__(NTHREADS, MINBLOCKS) gdn_fused(int kT,
                                                                        int kHk,
                                                                        int kHv,
                                                                        int kD,
                                                                        const float * __restrict__ qp,
                                                                        const float * __restrict__ kp,
                                                                        const float * __restrict__ vp,
                                                                        const float * __restrict__ gp,
                                                                        const float * __restrict__ bp,
                                                                        float * __restrict__ op,
                                                                        const float * __restrict__ s0p,
                                                                        float * __restrict__ stp,
                                                                        long long v_tok_stride) {
#if !defined(AMPERE_MMA_AVAILABLE)
    GGML_UNUSED_VARS(kT, kHk, kHv, kD, qp, kp, vp, gp, bp, op, s0p, stp, v_tok_stride);
    NO_DEVICE_CODE;
#else
    extern __shared__ bf16 arena[];
    int                    offset = 0;
    bf16 *                 Qs     = arena + offset;
    offset += CHUNK * DD;
    bf16 * Kh = arena + offset;
    offset += 2 * CHUNK * DD;
    bf16 * P = arena + offset;
    offset += CHUNK * CHUNK;
    bf16 * VbWh = arena + offset;
    offset += 2 * CHUNK * DV_TILE;
    bf16 * Sh = arena + offset;
    offset += DD * DV_TILE;
    bf16 * Vn2 = arena + offset;
    offset += 2 * CHUNK * DV_TILE;
    bf16 * Vn = arena + offset;
    offset += CHUNK * DV_TILE;

    // Preserve the measured shared-memory bank layout between value and inverse tiles.
    offset += 640;
    bf16 * Th = arena + offset;
    offset += CHUNK * CHUNK;
    bf16 * TCc = arena + offset;
    offset += CHUNK * CHUNK;
    __shared__ __align__(16) float A[CHUNK * CHUNK];  // FP32 gram and diagonal inverse workspace.
    __shared__ float               B[CHUNK], Cc[CHUNK], EX[CHUNK], Gc[CHUNK];
    __shared__ float               C2[2 * CHUNK];     // parity double-buffered (read after bar2)
    volatile __shared__ float      s_gl;

    __nv_bfloat16 * const Vb = VbWh;
    __nv_bfloat16 * const Wh = VbWh + CHUNK * DV_TILE;
    // Each warp half (0-3, 4-7) gets its own Vn2 copy: warps w and w+4 compute
    // the same tile, and separate copies make every store single-writer (clean
    // under racecheck) instead of an identical-byte WAW.

    // blockIdx.y selects the sequence: every pointer is offset to sequence b
    // here, so the rest of the kernel is single-sequence.
    {
        const long long b = blockIdx.y;
        qp += b * kT * kHk * kD;
        kp += b * kT * kHk * kD;
        vp += b * kT * v_tok_stride;
        gp += b * kT * kHv;
        bp += b * kT * kHv;
        op += b * kT * kHv * kD;
        s0p += b * kHv * kD * kD;
        stp += b * kHv * kD * kD;
    }
    const int   warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int   h     = blockIdx.x / (kD / DV_TILE);
    const int   dv0   = (blockIdx.x % (kD / DV_TILE)) * DV_TILE;
    const int   qkh   = h % kHk;
    const float scale = 1.f / sqrtf((float) kD);

    const int             nt  = warp & 3;
    const int             mt0 = 4 * (warp >> 2);
    const int             gr = lane >> 2, tig = lane & 3;
    const int             jbase  = 8 * nt + 2 * tig;
    const bool            o_warp = warp >= 4;                              // o-tile == state n-tile nt
    __nv_bfloat16 * const Vn2g   = Vn2 + (warp >> 2) * (CHUNK * DV_TILE);  // this warp half's copy
    const int             w2     = warp - 2;                               // warps 2-3: P half / Vb rows

    float  Sreg[4][4];
    float4 pq, pk;
    float  pgv, pbv;

    auto sts_sh = [&]() {
#    pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const int d0 = 16 * (mt0 + mt) + gr;
            const int c0 = swz(d0, nt, 3) << 3, c8 = swz(d0 + 8, nt, 3) << 3;
            store2(&Sh[d0 * DV_TILE + c0 + 2 * tig], make_float2((Sreg[mt][0]), (Sreg[mt][1])));
            store2(&Sh[(d0 + 8) * DV_TILE + c8 + 2 * tig], make_float2((Sreg[mt][2]), (Sreg[mt][3])));
        }
    };

    // Warp 0. Every exponential has a non-positive argument; the masked
    // intra-chunk factors exp(G_i - G_j) are formed at their use sites from Gc.
    auto decay_tables = [&](int rows, int buf) {
        float s = pgv;
#    pragma unroll
        for (int o = 1; o < 16; o <<= 1) {
            const float y = __shfl_up_sync(~0u, s, o);
            if (lane >= o)
                s += y;
        }
        const float glt = __shfl_sync(~0u, s, rows - 1);
        if (lane == 0)
            s_gl = glt;
        if (lane < CHUNK) {
            const float ex         = __expf(s);
            B[lane]                = pbv;
            EX[lane]               = ex;
            Gc[lane]               = s;
            Cc[lane]               = pbv * ex;
            C2[buf * CHUNK + lane] = __expf(glt - s);
        }
    };

    auto prefetch = [&](int tn) {
        const int t = tn + warp;
        if (t < kT) {
            const long long base = ((long long) t * kHk + qkh) * kD + 4 * lane;
            pq                   = *reinterpret_cast<const float4 *>(qp + base);
            pk                   = *reinterpret_cast<const float4 *>(kp + base);
        } else {
            pq = pk = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        if (warp == 0) {
            const int  tg = tn + lane;
            const bool ok = lane < CHUNK && tg < kT;
            pgv           = ok ? gp[(long long) tg * kHv + h] : 0.f;
            pbv           = ok ? bp[(long long) tg * kHv + h] : 0.f;
        }
    };

    auto sts_row4 = [&](__nv_bfloat16 * tile, int i, int j4, float4 r) {
        bf16 * d = &tile[i * DD + (swz(i, j4 >> 3, 7) << 3) + (j4 & 7)];
        store2(d + 0, make_float2((r.x), (r.y)));
        store2(d + 2, make_float2((r.z), (r.w)));
    };

    auto stage_qk = [&](int t0, int buf) {
        __nv_bfloat16 * const Khb = Kh + buf * CHUNK * DD;
        const int             j4  = 4 * lane;
        const int             t1  = t0 + warp + NWARPS;
        float4                q1 = make_float4(0.f, 0.f, 0.f, 0.f), k1 = q1;
        if (t1 < kT) {
            const long long base = ((long long) t1 * kHk + qkh) * kD + j4;
            q1                   = *reinterpret_cast<const float4 *>(qp + base);
            k1                   = *reinterpret_cast<const float4 *>(kp + base);
        }
        sts_row4(Qs, warp, j4, make_float4(pq.x * scale, pq.y * scale, pq.z * scale, pq.w * scale));
        sts_row4(Khb, warp, j4, pk);
        sts_row4(Qs, warp + NWARPS, j4, make_float4(q1.x * scale, q1.y * scale, q1.z * scale, q1.w * scale));
        sts_row4(Khb, warp + NWARPS, j4, k1);
    };

    // Warps 0-1: A = -beta_i exp(G_i - G_j) (k_i . k_j), j < i; warp w stores
    // columns 8w..8w+7.
    auto gram_A_half = [&](const __nv_bfloat16 * khb) {
        float acc[4] = { 0.f, 0.f, 0.f, 0.f };
        frag  a0, a1, a2, a3;
        ldsm4(&khb[(lane & 15) * DD + (swz(lane & 15, lane >> 4, 7) << 3)], a0, a1, a2, a3);
#    pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const frag A0 = a0, A1 = a1, A2 = a2, A3 = a3;
            if (ks + 1 < 8) {
                const int k16 = 16 * (ks + 1);
                ldsm4(&khb[(lane & 15) * DD + (swz(lane & 15, (k16 >> 3) + (lane >> 4), 7) << 3)], a0, a1, a2, a3);
            }
            if (warp == 0)
                mma16816(acc, A0, A1, A2, A3, A0, A2);
            else
                mma16816(acc, A0, A1, A2, A3, A1, A3);
        }
        const int   i0 = gr, i1 = i0 + 8;
        const int   jj = 8 * warp + 2 * tig;
        const float mA = -B[i0], mB = -B[i1];
        const float g0 = Gc[i0], g1 = Gc[i1];
        const float gj0 = Gc[jj], gj1 = Gc[jj + 1];
        *reinterpret_cast<float2 *>(&A[i0 * CHUNK + jj]) = make_float2(
            (jj < i0) ? mA * __expf(g0 - gj0) * acc[0] : 0.f, (jj + 1 < i0) ? mA * __expf(g0 - gj1) * acc[1] : 0.f);
        *reinterpret_cast<float2 *>(&A[i1 * CHUNK + jj]) = make_float2(
            (jj < i1) ? mB * __expf(g1 - gj0) * acc[2] : 0.f, (jj + 1 < i1) ? mB * __expf(g1 - gj1) * acc[3] : 0.f);
        __syncwarp();
    };

    // One lane per inverse column; keep the Gram matrix read-only during the
    // solve.
    auto invert_T = [&]() {
        if (warp == 0 && lane < CHUNK) {
            float tc[CHUNK];
#    pragma unroll
            for (int r = 0; r < CHUNK; ++r) {
                float value = r == lane ? 1.f : 0.f;
                if (r > lane) {
                    value = A[r * CHUNK + lane];
#    pragma unroll
                    for (int m = 1; m < CHUNK; ++m) {
                        if (m > lane && m < r)
                            value += A[r * CHUNK + m] * tc[m];
                    }
                }
                tc[r] = value;
                store1(Th + r * CHUNK + lane, value);
                store1(TCc + r * CHUNK + lane, -value * Cc[lane]);
            }
        }
    };

    // Warps 2-3: P[i][j] = exp(G_i - G_j) (q_i . k_j), j <= i; warp w stores
    // columns 8(w-2)..+7. The B chain loads only this warp's 8 token rows of Kh
    // (ldsm2, both matrices = same rows).
    auto p_gram_half = [&](const __nv_bfloat16 * khb) {
        float     acc[4] = { 0.f, 0.f, 0.f, 0.f };
        frag      a0, a1, a2, a3, b0, b1;
        const int rr = 8 * w2 + (lane & 7);
        ldsm4(&Qs[(lane & 15) * DD + (swz(lane & 15, lane >> 4, 7) << 3)], a0, a1, a2, a3);
        ldsm2(&khb[rr * DD + (swz(rr, ((lane >> 3) & 1), 7) << 3)], b0, b1);
#    pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const frag A0 = a0, A1 = a1, A2 = a2, A3 = a3;
            const frag B0 = b0, B1 = b1;
            if (ks + 1 < 8) {
                const int k16 = 16 * (ks + 1);
                ldsm4(&Qs[(lane & 15) * DD + (swz(lane & 15, (k16 >> 3) + (lane >> 4), 7) << 3)], a0, a1, a2, a3);
                ldsm2(&khb[rr * DD + (swz(rr, (k16 >> 3) + ((lane >> 3) & 1), 7) << 3)], b0, b1);
            }
            mma16816(acc, A0, A1, A2, A3, B0, B1);
        }
        const int   i0 = gr, i1 = i0 + 8;
        const int   jj = 8 * w2 + 2 * tig;
        const float g0 = Gc[i0], g8 = Gc[i1];
        const float gj0 = Gc[jj], gj1 = Gc[jj + 1];
        store2(&P[i0 * CHUNK + jj], make_float2(((jj <= i0) ? __expf(g0 - gj0) * acc[0] : 0.f),
                                                ((jj + 1 <= i0) ? __expf(g0 - gj1) * acc[1] : 0.f)));
        store2(&P[i1 * CHUNK + jj], make_float2(((jj <= i1) ? __expf(g8 - gj0) * acc[2] : 0.f),
                                                ((jj + 1 <= i1) ? __expf(g8 - gj1) * acc[3] : 0.f)));
        __syncwarp();
    };

    // ---- prologue: s0 loads (ggml [v][k] -> fragment S[k][v]) overlapped with
    // chunk 0 staging ----
    float2 st0[4], st1[4];
    {
        const float * __restrict__ s0h = s0p + (long long) h * kD * kD + (long long) (dv0 + jbase) * kD;
#    pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const int d0 = 16 * (mt0 + mt) + gr;
            st0[mt]      = make_float2(s0h[d0], s0h[kD + d0]);
            st1[mt]      = make_float2(s0h[d0 + 8], s0h[kD + d0 + 8]);
        }
    }
    prefetch(0);
    stage_qk(0, 0);
    if (warp == 0) {
        decay_tables(min(CHUNK, kT), 0);
    }
#    pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        Sreg[mt][0] = st0[mt].x;
        Sreg[mt][1] = st0[mt].y;
        Sreg[mt][2] = st1[mt].x;
        Sreg[mt][3] = st1[mt].y;
    }
    sts_sh();
    __syncthreads();

    const int NC = (kT + CHUNK - 1) / CHUNK;
    for (int c = 0; c < NC; ++c) {
        const int             t0  = c * CHUNK;
        const __nv_bfloat16 * khb = Kh + (c & 1) * CHUNK * DD;

        if (c + 1 < NC) {
            prefetch(t0 + CHUNK);  // consumed after bar2
        }
        const float dl      = __expf(s_gl);
        float       acco[4] = { 0.f, 0.f, 0.f, 0.f };

        // ---- Wwin: gram + inverse (warps 0-1); beta*v + P halves (warps 2-3); Q S
        // + K S (warps 4-7) ----
        if (warp < 2) {
            gram_A_half(khb);
        }
        if (warp >= 2 && warp < 4) {
            const int     iv   = w2 * 8 + (lane >> 2);
            const int     j4v  = 4 * (lane & 3);
            const int     tv   = t0 + iv;
            const float * vrow = vp + (long long) tv * v_tok_stride + (long long) h * kD + dv0 + j4v;
            const float4  v0   = (tv < kT) ? *reinterpret_cast<const float4 *>(vrow) : make_float4(0.f, 0.f, 0.f, 0.f);
            const float4  v1 =
                (tv < kT) ? *reinterpret_cast<const float4 *>(vrow + 16) : make_float4(0.f, 0.f, 0.f, 0.f);
            p_gram_half(khb);
            const float  biv = B[iv];
            bf16 * const v2a = &Vb[iv * DV_TILE + (swz(iv, j4v >> 3, 3) << 3) + (j4v & 7)];
            store2(v2a + 0, make_float2((biv * v0.x), (biv * v0.y)));
            store2(v2a + 2, make_float2((biv * v0.z), (biv * v0.w)));
            const int    j4w = j4v + 16;
            bf16 * const v2b = &Vb[iv * DV_TILE + (swz(iv, j4w >> 3, 3) << 3) + (j4w & 7)];
            store2(v2b + 0, make_float2((biv * v1.x), (biv * v1.y)));
            store2(v2b + 2, make_float2((biv * v1.z), (biv * v1.w)));
        }
        if (o_warp) {
            float acc2[4] = { 0.f, 0.f, 0.f, 0.f };
            frag  a0, a1, a2, a3, bh0, bh1;
            ldsm4(&Qs[(lane & 15) * DD + (swz(lane & 15, lane >> 4, 7) << 3)], a0, a1, a2, a3);
            ldsm2t(&Sh[(lane & 15) * DV_TILE + (swz(lane & 15, nt, 3) << 3)], bh0, bh1);
#    pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const frag A0 = a0, A1 = a1, A2 = a2, A3 = a3;
                const frag BH0 = bh0, BH1 = bh1;
                if (ks + 1 < 8) {
                    const int k16 = 16 * (ks + 1);
                    ldsm4(&Qs[(lane & 15) * DD + (swz(lane & 15, (k16 >> 3) + (lane >> 4), 7) << 3)], a0, a1, a2, a3);
                    ldsm2t(&Sh[(k16 + (lane & 15)) * DV_TILE + (swz(k16 + (lane & 15), nt, 3) << 3)], bh0, bh1);
                }
                mma16816(acc2, A0, A1, A2, A3, BH0, BH1);
            }
            const int i0 = gr, i1 = i0 + 8;
            acco[0] = EX[i0] * acc2[0];
            acco[1] = EX[i0] * acc2[1];
            acco[2] = EX[i1] * acc2[2];
            acco[3] = EX[i1] * acc2[3];

            float accA[4] = { 0.f, 0.f, 0.f, 0.f };
            ldsm4(&khb[(lane & 15) * DD + (swz(lane & 15, lane >> 4, 7) << 3)], a0, a1, a2, a3);
            ldsm2t(&Sh[(lane & 15) * DV_TILE + (swz(lane & 15, nt, 3) << 3)], bh0, bh1);
#    pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const frag A0 = a0, A1 = a1, A2 = a2, A3 = a3;
                const frag BH0 = bh0, BH1 = bh1;
                if (ks + 1 < 8) {
                    const int k16 = 16 * (ks + 1);
                    ldsm4(&khb[(lane & 15) * DD + (swz(lane & 15, (k16 >> 3) + (lane >> 4), 7) << 3)], a0, a1, a2, a3);
                    ldsm2t(&Sh[(k16 + (lane & 15)) * DV_TILE + (swz(k16 + (lane & 15), nt, 3) << 3)], bh0, bh1);
                }
                mma16816(accA, A0, A1, A2, A3, BH0, BH1);
            }
            store2(&Wh[i0 * DV_TILE + (swz(i0, nt, 3) << 3) + 2 * tig], make_float2((accA[0]), (accA[1])));
            store2(&Wh[i1 * DV_TILE + (swz(i1, nt, 3) << 3) + 2 * tig], make_float2((accA[2]), (accA[3])));
        }
        __syncthreads();  // Gram and Q/state products are ready.
        invert_T();
        __syncthreads();  // Inverse high/low planes are ready.

        // ---- Ph2: vnew tile nt = Th*Vb + TCc*Wh, produced by every warp that
        // consumes it ----
        const int c2b = (c & 1) * CHUNK;
        frag      pa0 = make_uint2(0, 0), pa1 = make_uint2(0, 0), pa2 = make_uint2(0, 0), pa3 = make_uint2(0, 0);
        if (o_warp) {
            ldsm4(&P[(lane & 15) * CHUNK + ((lane >> 4) << 3)], pa0, pa1, pa2, pa3);
        }
        {
            float acc[4] = { 0.f, 0.f, 0.f, 0.f };
            frag  a0, a1, a2, a3, b0, b1, b2, b3;
            ldsm2t(&Vb[(lane & 15) * DV_TILE + (swz(lane & 15, nt, 3) << 3)], b0, b1);
            ldsm2t(&Wh[(lane & 15) * DV_TILE + (swz(lane & 15, nt, 3) << 3)], b2, b3);
            ldsm4(Th + (lane & 15) * 16 + ((lane >> 4) << 3), a0, a1, a2, a3);
            mma16816(acc, a0, a1, a2, a3, b0, b1);
            ldsm4(TCc + (lane & 15) * 16 + ((lane >> 4) << 3), a0, a1, a2, a3);
            mma16816(acc, a0, a1, a2, a3, b2, b3);
            const int i0 = gr, i1 = i0 + 8;
            const int c0 = (swz(i0, nt, 3) << 3) + 2 * tig, c1 = (swz(i1, nt, 3) << 3) + 2 * tig;
            store2(&Vn2g[i0 * DV_TILE + c0], make_float2((C2[c2b + i0] * acc[0]), (C2[c2b + i0] * acc[1])));
            store2(&Vn2g[i1 * DV_TILE + c1], make_float2((C2[c2b + i1] * acc[2]), (C2[c2b + i1] * acc[3])));
            if (o_warp) {
                store2(&Vn[i0 * DV_TILE + c0], make_float2((acc[0]), (acc[1])));
                store2(&Vn[i1 * DV_TILE + c1], make_float2((acc[2]), (acc[3])));
            }
        }
        __syncthreads();

        // ---- Ph3: S = dl*S + K^T Vn2 (all warps, own tiles);  o = o_inter + P Vn
        // (warps 4-7) ----
        {
#    pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                Sreg[mt][0] *= dl;
                Sreg[mt][1] *= dl;
                Sreg[mt][2] *= dl;
                Sreg[mt][3] *= dl;
            }
            frag a0, a1, a2, a3, B0, B1;
            frag OB0 = make_uint2(0, 0), OB1 = make_uint2(0, 0);
            ldsm2t(&Vn2g[(lane & 15) * DV_TILE + (swz(lane & 15, nt, 3) << 3)], B0, B1);
            if (o_warp) {
                ldsm2t(&Vn[(lane & 15) * DV_TILE + (swz(lane & 15, nt, 3) << 3)], OB0, OB1);
            }
            ldsm4t(&khb[(lane & 15) * DD + (swz(lane & 15, 2 * mt0 + (lane >> 4), 7) << 3)], a0, a1, a2, a3);
#    pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const frag A0 = a0, A1 = a1, A2 = a2, A3 = a3;
                if (mt + 1 < 4) {
                    ldsm4t(&khb[(lane & 15) * DD + (swz(lane & 15, 2 * (mt0 + mt + 1) + (lane >> 4), 7) << 3)], a0, a1,
                           a2, a3);
                }
                mma16816(Sreg[mt], A0, A2, A1, A3, B0, B1);
                if (o_warp && mt == 0) {
                    mma16816(acco, pa0, pa1, pa2, pa3, OB0, OB1);
                }
            }
            if (o_warp) {
                const int jj = 8 * nt + 2 * tig;
                const int tA = t0 + gr, tB = tA + 8;
                if (tA < kT) {
                    *reinterpret_cast<float2 *>(op + ((long long) tA * kHv + h) * kD + dv0 + jj) =
                        make_float2(acco[0], acco[1]);
                }
                if (tB < kT) {
                    *reinterpret_cast<float2 *>(op + ((long long) tB * kHv + h) * kD + dv0 + jj) =
                        make_float2(acco[2], acco[3]);
                }
            }
            __syncthreads();
            if (c + 1 < NC) {
                sts_sh();
            }
        }
        __syncthreads();  // Finish consuming this chunk before reusing its input
                          // buffers.
        // Stage the next chunk after all current-chunk reads have finished.
        if (c + 1 < NC) {
            stage_qk(t0 + CHUNK, (c + 1) & 1);
            if (warp == 0) {
                decay_tables(min(CHUNK, kT - (t0 + CHUNK)), (c + 1) & 1);
            }
        }

        __syncthreads();
    }

    // st[v][k] <- S[k][v], into ggml's [v][k]
    {
        float * __restrict__ sth = stp + (long long) h * kD * kD + (long long) (dv0 + jbase) * kD;
#    pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const int d0     = 16 * (mt0 + mt) + gr;
            sth[d0]          = Sreg[mt][0];
            sth[kD + d0]     = Sreg[mt][1];
            sth[d0 + 8]      = Sreg[mt][2];
            sth[kD + d0 + 8] = Sreg[mt][3];
        }
    }
#endif
}

}  // namespace gdn_value_split
