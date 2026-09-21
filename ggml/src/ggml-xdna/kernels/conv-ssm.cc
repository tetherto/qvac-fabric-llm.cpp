// Causal depthwise conv1d for the GDN layers (ggml SSM_CONV).
//
//     out[t][c] = sum_{i<KW} x[c][t+i] * w[c][i]
//
// There is no reduction across channels, so the op is embarrassingly parallel
// over c and every core owns a channel slice outright.
//
// The vectorisation goes along t, not c. ggml hands the input in as
// {d_conv-1+n_t, d_inner} with the *time* axis contiguous inside a channel row,
// so the KW-wide sliding window is KW contiguous loads and the whole thing is KW
// broadcast-scalar MACs. Vectorising along c instead would read with a stride of
// (n_t + KW - 1) floats per lane.
//
// That leaves the output channel-major, while ggml's dst is {d_inner, n_t} with
// *channels* contiguous, so the host has to reorder it. Neither a core nor the
// drain descriptor can: a core would pay strided scalar stores per element, and
// a transposing drain measured 2.8 GB/s against 12.9.
//
// x and w arrive as one buffer, x first: a column's mem tile has to serve the
// split to its four cores and the join back from them, and three separate
// distributions (x, w, out) exceed its DMA channel budget -- aiecc says so
// outright ("no MemTile has sufficient DMA capacity for 1 input/4 output
// channels"). One stream carrying both operands is the same trick the attn
// design uses to put K and V on one fifo.
//
// CONV_TOKEN_MAJOR flips all of the above. The channel-major convention costs
// the host two full transposes per call, because both of this op's neighbours
// in the graph are token-major: the projection that feeds it is
// [tokens][channels] and ggml's dst is {d_inner, n_t}, so the pack gathers with
// a 24 KB stride and the scatter scatters with one. Taking x as [XROWT][C] and
// writing out as [T][C] removes both -- the host copies whole rows. The
// arithmetic is identical -- KW*T vector MACs either way -- and one lane per
// channel needs no sliding window, so the shuffle disappears too. It buys that
// for a KW-1 row halo per core, since the design then splits a column's tokens
// across its cores rather than its channels.
//
// Compile flags: DIM_C (channels per core per tile), DIM_T (tokens per tile),
// DIM_KW (kernel width), CONV_TOKEN_MAJOR (layout, see above).

#include <aie_api/aie.hpp>

// Element type of the packed xw/out buffers. bf16 halves every BO stream of the
// conv - the pack write, the dispatch read and write and the scatter read -
// while the arithmetic is unchanged: the accumulators stay accfloat and only
// the operands and the stored result are narrowed.
#ifdef CONV_BF16
typedef bfloat16 conv_elem;
#else
typedef float conv_elem;
#endif

#ifndef DIM_C
#error "DIM_C must be defined"
#endif
#ifndef DIM_T
#error "DIM_T must be defined"
#endif
#ifndef DIM_KW
#error "DIM_KW must be defined"
#endif

// 512-bit vectors: 16 f32 lanes.
#define VLEN 16

#ifdef CONV_TOKEN_MAJOR

// Rows are whole vectors of channels, so nothing pads: a row is DIM_C floats
// and every row start is DIM_C-aligned.
#define XROWT (DIM_T + DIM_KW - 1)

static_assert(DIM_C % VLEN == 0, "DIM_C must be a multiple of the vector length");

// x is [XROWT][C] and w is [KW][C], so tap i of channel lane c sits at
// (t+i)*C + c: every load is contiguous and aligned, and the KW taps are KW
// whole vectors rather than a window slid across two.
extern "C" void ggml_xdna_conv_apply(const conv_elem * __restrict xw,
                                     conv_elem * __restrict       out) {
    const conv_elem * const w = xw + (size_t) XROWT * DIM_C;

    for (unsigned t = 0; t < DIM_T; t++) {
        const conv_elem * xt = xw + (size_t) t * DIM_C;
        conv_elem *       ot = out + (size_t) t * DIM_C;

        for (unsigned c = 0; c < DIM_C; c += VLEN) {
            aie::accum<accfloat, VLEN> acc = aie::zeros<accfloat, VLEN>();
            for (unsigned i = 0; i < DIM_KW; i++) {
                acc = aie::mac(acc,
                               aie::load_v<VLEN>(xt + (size_t) i * DIM_C + c),
                               aie::load_v<VLEN>(w + (size_t) i * DIM_C + c));
            }
            aie::store_v(ot + c, acc.template to_vector<conv_elem>());
        }
    }
}

#else

// Channel rows pad to VLEN so xc+c*XROW stays aligned; T+KW-1 is 259 for the
// default tile and a misaligned load_v slides.
static_assert(DIM_T % VLEN == 0, "DIM_T must be a multiple of the vector length");

#define XROW (((DIM_T + DIM_KW - 1 + VLEN - 1) / VLEN) * VLEN)
static_assert(XROW % VLEN == 0, "XROW must be a multiple of the vector length");
static_assert(XROW >= DIM_T + VLEN, "XROW must cover the aligned pair of loads");
static_assert(DIM_KW <= VLEN, "shuffle tap must fit in one vector");

extern "C" void ggml_xdna_conv_apply(const conv_elem * __restrict xw,
                                     conv_elem * __restrict       out) {
    const conv_elem * const w = xw + (size_t) DIM_C * XROW;

    for (unsigned c = 0; c < DIM_C; c++) {
        const conv_elem * xc = xw + (size_t) c * XROW;
        const conv_elem * wc = w + (size_t) c * DIM_KW;
        conv_elem *       oc = out + (size_t) c * DIM_T;

        for (unsigned t = 0; t < DIM_T; t += VLEN) {
            const aie::vector<conv_elem, VLEN> x_lo = aie::load_v<VLEN>(xc + t);
            const aie::vector<conv_elem, VLEN> x_hi = aie::load_v<VLEN>(xc + t + VLEN);
            aie::accum<accfloat, VLEN> acc = aie::zeros<accfloat, VLEN>();
            // Tap i needs x[t+i : t+i+VLEN]. load_v at t+i is misaligned for
            // i>0; two aligned loads plus shuffle_down_fill is the sliding window.
            for (unsigned i = 0; i < DIM_KW; i++) {
                acc = aie::mac(acc, aie::shuffle_down_fill(x_lo, x_hi, i),
                               aie::broadcast<conv_elem, VLEN>(wc[i]));
            }
            aie::store_v(oc + t, acc.template to_vector<conv_elem>());
        }
    }
}

#endif
