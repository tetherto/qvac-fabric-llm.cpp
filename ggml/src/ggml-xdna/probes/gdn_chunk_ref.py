# The chunked gated delta rule against ggml's
# token recurrence (ggml_compute_forward_gated_delta_net), on the host, with
# and without bf16 rounding of every matmul operand (the state kept exact):
#   python3 probes/gdn_chunk_ref.py [tokens] [chunk] [doubling|inverse] [gate scale]
# ("doubling": T by products of rounded matmuls, as the array would form it)
# Per chunk (G = exp of the cumulative gate, D[t, s] = G_t / G_s):
#   T = (I + tril(beta K K^T * D, -1))^-1
#   U = T (beta V),  W = T (beta G K),  Delta = U - W S0
#   O = scale (G Q S0 + tril(Q K^T * D) Delta)
#   S = G_C S0 + ((G_C / G) K)^T Delta
import numpy as np
import sys
try:
    import ml_dtypes
    bf = lambda x: x.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float64)
except ImportError:
    def bf(x):
        a = np.asarray(x, np.float32).copy()
        b = a.view(np.uint32)
        b[:] = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
        return a.astype(np.float64)
rng = np.random.default_rng(0)
d, T = 128, int(sys.argv[1]) if len(sys.argv) > 1 else 512
C = int(sys.argv[2]) if len(sys.argv) > 2 else 64
DOUBLING = len(sys.argv) > 3 and sys.argv[3] == "doubling"
GS = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3     # the gate's scale: 0.003 is weak decay
nrm = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
q = nrm(rng.standard_normal((T, d)))
k = nrm(rng.standard_normal((T, d)))
v = rng.standard_normal((T, d))
beta = 1 / (1 + np.exp(-rng.standard_normal(T)))
g = -np.log1p(np.exp(rng.standard_normal(T))) * GS
S0 = rng.standard_normal((d, d)) * 0.1
scale = 1 / np.sqrt(d)


def ref():
    S = S0.copy()
    o = np.zeros((T, d))
    for t in range(T):
        S *= np.exp(g[t])
        delta = beta[t] * (v[t] - S.T @ k[t])
        S += np.outer(k[t], delta)
        o[t] = scale * (S.T @ q[t])
    return o, S


def chunked(rnd):
    S = S0.copy()
    o = np.zeros((T, d))
    for c0 in range(0, T, C):
        K, Q, V, b, gg = k[c0:c0 +C], q[c0:c0 +C], v[c0:c0 +C], beta[c0:c0 +C], g[c0:c0 +C]
        n = len(b)
        gam = np.cumsum(gg)
        G = np.exp(gam)
        Dm = np.exp(gam[:, None] - gam[None, :])                 # G_t / G_s
        L = np.tril(b[:, None] * Dm * (rnd(K) @ rnd(K).T), -1)
        if DOUBLING:
            # (I + L)^-1 for a strictly lower L: (I - L)(I + L^2)(I + L^4)...,
            # every product a matmul of rounded operands
            Tm = np.eye(n) - L
            P = rnd(L) @ rnd(L)
            for _ in range(int(np.ceil(np.log2(n))) - 1):
                Tm = rnd(Tm) @ rnd(np.eye(n) + P)
                P = rnd(P) @ rnd(P)
        else:
            Tm = np.linalg.inv(np.eye(n) + L)
        U = rnd(Tm) @ rnd(b[:, None] * V)
        W = rnd(Tm) @ rnd((b * G)[:, None] * K)
        Sb = rnd(S)
        Dl = U - rnd(W) @ Sb
        A = np.tril(Dm * (rnd(Q) @ rnd(K).T))
        o[c0:c0 +n] = scale * (G[:, None] * (rnd(Q) @ Sb) + rnd(A) @ rnd(Dl))
        S = G[-1] * S + rnd((np.exp(gam[-1] - gam))[:, None] * K).T @ rnd(Dl)
    return o, S


def array_form():
    """As the array holds it: the state a bf16 hi/lo pair S_st with the decay
    a scalar sigma (S = sigma * S_st, renormalized by powers of two), T by
    doubling, every matmul operand bf16."""
    rnd = bf
    hi = bf(S0)
    lo = bf(S0 - hi)
    sig = 1.0
    o = np.zeros((T, d))
    for c0 in range(0, T, C):
        K, Q, V, b, gg = k[c0:c0 +C], q[c0:c0 +C], v[c0:c0 +C], beta[c0:c0 +C], g[c0:c0 +C]
        n = len(b)
        gam = np.cumsum(gg)
        G = np.exp(gam)
        Dm = np.exp(gam[:, None] - gam[None, :])
        L = np.tril(b[:, None] * Dm * (rnd(K) @ rnd(K).T), -1)
        Tm = np.eye(n) - L
        P = rnd(L) @ rnd(L)
        for _ in range(int(np.ceil(np.log2(n))) - 1):
            Tm = rnd(Tm) @ rnd(np.eye(n) + P)
            P = rnd(P) @ rnd(P)
        U = rnd(Tm) @ rnd(b[:, None] * V)
        W = rnd(Tm) @ rnd((b * G * sig)[:, None] * K)
        Dl = U - rnd(W) @ hi
        A = np.tril(Dm * (rnd(Q) @ rnd(K).T))
        o[c0:c0 +n] = scale * (rnd((G * sig)[:, None] * Q) @ hi + rnd(A) @ rnd(Dl))
        sig *= G[-1]
        new = hi + lo + rnd((np.exp(gam[-1] - gam) / sig)[:, None] * K).T @ rnd(Dl)
        if sig < 2.0 ** -32:
            e = np.floor(np.log2(sig))
            new *= 2.0 ** e
            sig /= 2.0 ** e
        hi = bf(new)
        lo = bf(new - hi)
    return o, sig * (hi + lo)


o_r, S_r = ref()
for name, fn in [("exact", lambda: chunked(lambda x: x)), ("bf16 operands", lambda: chunked(bf)),
                 ("array form", array_form)]:
    o_c, S_c = fn()
    e = lambda a, b: np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean())
    print(f"C={C} T={T} {name:14s}: out rel {e(o_c, o_r):.2e}  state rel {e(S_c, S_r):.2e}")
