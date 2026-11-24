// Recipe interpreter: evaluation, the ops, number formats and blob tolerance.
//
// Compiled with -ffp-contract=off (qvac_coreml.cmake): every arithmetic op is
// one IEEE operation in its operand dtype, with no fused multiply-add, so the
// bytes are the same on every host. Plain portable loops only.

#include "qvac_coreml/recipe.h"

#include "milblob.h"
#include "sha256.h"
#include "shape.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <type_traits>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the Core ML generator writes little-endian blobs from host-order floats"
#endif

namespace QVAC_COREML_NS {
namespace recipe {

// ---- binary16 ------------------------------------------------------------------

uint16_t f32_to_f16(float x) {
    // numpy npy_floatbits_to_halfbits: round to nearest, ties to even.
    uint32_t f;
    std::memcpy(&f, &x, 4);
    const uint16_t h_sgn = static_cast<uint16_t>((f & 0x80000000u) >> 16);
    uint32_t f_exp = f & 0x7f800000u;
    uint32_t f_sig;
    if (f_exp >= 0x47800000u) {  // overflow, Inf or NaN
        if (f_exp == 0x7f800000u) {
            f_sig = f & 0x007fffffu;
            if (f_sig != 0) {
                uint16_t ret = static_cast<uint16_t>(0x7c00u + (f_sig >> 13));
                if (ret == 0x7c00u) ++ret;  // keep NaN a NaN
                return static_cast<uint16_t>(h_sgn + ret);
            }
        }
        return static_cast<uint16_t>(h_sgn + 0x7c00u);
    }
    if (f_exp <= 0x38000000u) {  // subnormal half or zero
        if (f_exp < 0x33000000u) return h_sgn;
        f_exp >>= 23;
        f_sig = 0x00800000u + (f & 0x007fffffu);
        f_sig >>= (113 - f_exp);
        if (((f_sig & 0x00003fffu) != 0x00001000u) || (f & 0x000007ffu)) f_sig += 0x00001000u;
        return static_cast<uint16_t>(h_sgn + (f_sig >> 13));
    }
    const uint16_t h_exp = static_cast<uint16_t>((f_exp - 0x38000000u) >> 13);
    f_sig = f & 0x007fffffu;
    if ((f_sig & 0x00003fffu) != 0x00001000u) f_sig += 0x00001000u;
    uint16_t h_sig = static_cast<uint16_t>(f_sig >> 13);
    h_sig = static_cast<uint16_t>(h_sig + h_exp);  // a carry into the exponent is correct (up to Inf)
    return static_cast<uint16_t>(h_sgn + h_sig);
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) {
            f = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                --exp;
            }
            man &= 0x3ffu;
            f = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1f) {
        f = sign | 0x7f800000u | (man << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

namespace {

int64_t ordered16(uint16_t b) { return (b & 0x8000u) ? -static_cast<int64_t>(b & 0x7fffu) : static_cast<int64_t>(b); }
int64_t ordered32(uint32_t b) {
    return (b & 0x80000000u) ? -static_cast<int64_t>(b & 0x7fffffffu) : static_cast<int64_t>(b);
}
bool finite16(uint16_t b) { return (b & 0x7c00u) != 0x7c00u; }

}  // namespace

int64_t ulp_distance_f16(uint16_t a, uint16_t b) {
    if (!finite16(a) || !finite16(b)) return -1;
    const int64_t d = ordered16(a) - ordered16(b);
    return d < 0 ? -d : d;
}

int64_t ulp_distance_f32(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return -1;
    uint32_t x, y;
    std::memcpy(&x, &a, 4);
    std::memcpy(&y, &b, 4);
    const int64_t d = ordered32(x) - ordered32(y);
    return d < 0 ? -d : d;
}

BlobCompare compare_blob(BlobType dt, bool ulp, const uint8_t * ours, const uint8_t * ref, uint64_t count) {
    BlobCompare c;
    c.ulp = ulp;
    bool ok = true;
    double max_abs = 0.0;
    auto value = [&](const uint8_t * p, uint64_t i) -> double {
        if (dt == BlobType::fp16) {
            uint16_t b;
            std::memcpy(&b, p + 2 * i, 2);
            return f16_to_f32(b);
        }
        float f;
        std::memcpy(&f, p + 4 * i, 4);
        return f;
    };
    for (uint64_t i = 0; i < count; ++i) max_abs = std::max(max_abs, std::fabs(value(ref, i)));
    const double tol = std::ldexp(max_abs, -10);
    const size_t es = blob_type_size(dt);
    for (uint64_t i = 0; i < count; ++i) {
        const double a = value(ours, i);
        const double b = value(ref, i);
        if (!std::isfinite(a) || !std::isfinite(b)) {
            ok = false;
            c.max_ulp = -1;
            if (std::memcmp(ours + es * i, ref + es * i, es) != 0) ++c.differing;
            continue;
        }
        if (std::memcmp(ours + es * i, ref + es * i, es) == 0) continue;
        int64_t d;
        if (dt == BlobType::fp16) {
            uint16_t x, y;
            std::memcpy(&x, ours + 2 * i, 2);
            std::memcpy(&y, ref + 2 * i, 2);
            d = ulp_distance_f16(x, y);
        } else {
            d = ulp_distance_f32(static_cast<float>(a), static_cast<float>(b));
        }
        // The bits differ. An exact blob fails on any difference, including
        // +0 vs -0; only the ulp rule measures +-0 as 0 apart.
        ++c.differing;
        if (c.max_ulp >= 0) c.max_ulp = std::max(c.max_ulp, d);
        if (!ulp || !(d <= 1 || std::fabs(a - b) <= tol)) ok = false;
    }
    c.pass = ok;
    return c;
}

WeightBinCheck verify_weight_bin(const Recipe & r, const uint8_t * ours, size_t ours_size, const uint8_t * ref,
                                 size_t ref_size) {
    WeightBinCheck v;
    std::string err;
    const auto oi = milblob::read_index(ours, ours_size, &err);
    if (!oi) v.errors.push_back("ours: " + err);
    const auto ri = milblob::read_index(ref, ref_size, &err);
    if (!ri) v.errors.push_back("ref: " + err);
    if (ours_size != r.weight_bin_size || ref_size != r.weight_bin_size) {
        v.errors.push_back("size: ours " + std::to_string(ours_size) + ", ref " + std::to_string(ref_size) +
                           ", recipe " + std::to_string(r.weight_bin_size) + " bytes");
    }
    if (oi && ri && (oi->size() != r.blob_count || ri->size() != r.blob_count)) {
        v.errors.push_back("blob count: ours " + std::to_string(oi->size()) + ", ref " + std::to_string(ri->size()) +
                           ", recipe " + std::to_string(r.blob_count));
    }
    // Every byte outside the recipe's payload ranges must be equal: the
    // header, each blob's metadata (dtype, reserved fields) and the padding.
    std::vector<std::pair<uint64_t, uint64_t>> payloads;
    for (const Item & it : r.items) {
        if (it.kind != Item::Kind::blob) continue;
        const uint64_t b = it.offset + milblob::METADATA_BYTES;
        payloads.emplace_back(b, b + it.count * blob_type_size(it.dtype));
    }
    std::sort(payloads.begin(), payloads.end());
    const uint64_t common = std::min<uint64_t>(ours_size, ref_size);
    auto first_diff = [&](uint64_t from, uint64_t to) -> int64_t {
        to = std::min(to, common);
        for (uint64_t i = from; i < to; ++i) {
            if (ours[i] != ref[i]) return static_cast<int64_t>(i);
        }
        return -1;
    };
    uint64_t pos = 0;
    int64_t diff = -1;
    for (const auto & pl : payloads) {
        if (diff < 0 && pl.first > pos) diff = first_diff(pos, pl.first);
        pos = std::max(pos, pl.second);
    }
    if (diff < 0) diff = first_diff(pos, common);
    if (diff >= 0) v.errors.push_back("non-payload byte differs at offset " + std::to_string(diff));

    size_t bi = 0;
    for (const Item & it : r.items) {
        if (it.kind != Item::Kind::blob) continue;
        BlobCheck bc;
        bc.cmp.offset = it.offset;
        bc.cmp.ulp = it.ulp;
        const milblob::Entry * o = oi && bi < oi->size() ? &(*oi)[bi] : nullptr;
        const milblob::Entry * f = ri && bi < ri->size() ? &(*ri)[bi] : nullptr;
        ++bi;
        const uint64_t bytes = it.count * blob_type_size(it.dtype);
        const uint32_t dtype = it.dtype == BlobType::fp16 ? milblob::FLOAT16 : milblob::FLOAT32;
        auto entry_ok = [&](const milblob::Entry * e, const char * who) {
            if (!e) {
                bc.layout = std::string(who) + ": missing";
            } else if (e->offset != it.offset || e->bytes != bytes) {
                bc.layout = std::string(who) + ": offset " + std::to_string(e->offset) + ", " +
                            std::to_string(e->bytes) + " bytes";
            } else if (e->dtype != dtype) {
                bc.layout = std::string(who) + ": MILBlob dtype " + std::to_string(e->dtype) + ", recipe " +
                            std::to_string(dtype);
            }
            return bc.layout.empty();
        };
        if (entry_ok(o, "ours") && entry_ok(f, "ref")) {
            const uint64_t at = o->data_offset;
            bc.cmp = compare_blob(it.dtype, it.ulp, ours + at, ref + at, it.count);
            bc.cmp.offset = it.offset;
        }
        v.blobs.push_back(std::move(bc));
    }
    v.pass = v.errors.empty();
    for (const BlobCheck & b : v.blobs) v.pass = v.pass && b.layout.empty() && b.cmp.pass;
    return v;
}

// ---- Evaluation ----------------------------------------------------------------

namespace {

class EvalError {
public:
    explicit EvalError(Error e) : err(std::move(e)) {}
    Error err;
};

// Product of shape[b, e) (all of it by default). The parser bounds every
// shape, so this throws only for a Recipe that did not come from parse().
size_t count_of(const std::vector<int64_t> & shape, size_t b = 0, size_t e = SIZE_MAX) {
    e = std::min(e, shape.size());
    const std::vector<int64_t> part(shape.begin() + static_cast<std::ptrdiff_t>(std::min(b, e)),
                                    shape.begin() + static_cast<std::ptrdiff_t>(e));
    const std::optional<uint64_t> n = detail::element_count(part);
    if (!n) {
        throw EvalError(Error::deterministic("recipe: a shape has more than " + std::to_string(detail::MAX_ELEMENTS) +
                                             " elements"));
    }
    return static_cast<size_t>(*n);
}

struct Tracker {
    uint64_t live = 0;
    uint64_t peak = 0;
    void alloc(uint64_t n) {
        live += n;
        peak = std::max(peak, live);
    }
    void release(uint64_t n) { live -= n; }
};

// One value: a weak scalar or an f32/f64 array, row-major in numpy order.
struct Buf {
    explicit Buf(Tracker * t) : tr(t) {}
    ~Buf() { tr->release(bytes); }
    Buf(const Buf &) = delete;
    Buf & operator=(const Buf &) = delete;

    void alloc(DType d, std::vector<int64_t> s) {
        dt = d;
        shape = std::move(s);
        const size_t n = count_of(shape);
        if (d == DType::f32) {
            f.assign(n, 0.0f);
            account(n * sizeof(float));
        } else if (d == DType::f64) {
            g.assign(n, 0.0);
            account(n * sizeof(double));
        }
    }
    void account(uint64_t n) {
        tr->alloc(n);
        bytes += n;
    }
    size_t size() const { return dt == DType::f32 ? f.size() : dt == DType::f64 ? g.size() : 1; }

    Tracker * tr;
    DType dt = DType::weak;
    std::vector<int64_t> shape;
    std::vector<float> f;
    std::vector<double> g;
    double weak = 0.0;
    uint64_t bytes = 0;
};
using V = std::shared_ptr<Buf>;

std::vector<size_t> strides_of(const std::vector<int64_t> & shape) {
    std::vector<size_t> s(shape.size(), 1);
    for (size_t i = shape.size(); i-- > 1;) s[i - 1] = count_of(shape, i);
    return s;
}

class Interp {
public:
    // `driver`: the items this pass evaluates in file order (blob for
    // weight.bin, patch for the spec); lets are evaluated on demand.
    Interp(const Recipe & r, TensorSource & src, const ReadObserver & obs, Item::Kind driver = Item::Kind::blob)
        : r_(r), src_(src), obs_(obs), driver_(driver) {
        for (const Item & it : r.items) {
            if (it.kind == Item::Kind::let) lets_[it.name] = &it;
        }
    }

    Tracker tracker;
    uint64_t tensors_read = 0;

    void plan_liveness() {
        // direct(x): the lets an expression names (not through other lets).
        std::map<std::string, std::set<std::string>> direct_let;
        std::vector<std::set<std::string>> direct_blob;
        for (const Item & it : r_.items) {
            std::set<std::string> d;
            names(*it.expr, d);
            if (it.kind == Item::Kind::let) {
                direct_let[it.name] = d;
            } else if (it.kind == driver_) {
                direct_blob.push_back(d);
            }
        }
        // first_need(L): the first blob whose transitive closure includes L.
        std::map<std::string, size_t> first;
        for (size_t b = 0; b < direct_blob.size(); ++b) {
            std::vector<std::string> stack(direct_blob[b].begin(), direct_blob[b].end());
            std::set<std::string> seen;
            while (!stack.empty()) {
                std::string n = stack.back();
                stack.pop_back();
                if (!seen.insert(n).second) continue;
                first.emplace(n, b);
                for (const std::string & m : direct_let[n]) stack.push_back(m);
            }
        }
        // A let is read when a blob names it, and when a let naming it is
        // first evaluated; it is freed after its last read.
        for (size_t b = 0; b < direct_blob.size(); ++b) {
            for (const std::string & n : direct_blob[b]) bump(n, b);
        }
        for (const auto & [l, deps] : direct_let) {
            auto f = first.find(l);
            if (f == first.end()) continue;  // never needed
            for (const std::string & n : deps) bump(n, f->second);
        }
    }

    void free_after(size_t blob) {
        for (const auto & [n, last] : last_use_) {
            if (last == blob) cache_.erase(n);
        }
    }

    V eval(const Expr & e) {
        switch (e.kind) {
            case Expr::Kind::number: {
                V v = std::make_shared<Buf>(&tracker);
                v->dt = DType::weak;
                v->weak = e.number;
                return v;
            }
            case Expr::Kind::name: {
                auto c = cache_.find(e.text);
                if (c != cache_.end()) return c->second;
                V v = eval(*lets_.at(e.text)->expr);
                cache_[e.text] = v;
                return v;
            }
            case Expr::Kind::call: return call(e);
            default: throw EvalError(Error::deterministic("recipe: a string or list is not a value"));
        }
    }

    // The blob's bytes: the top cast applied to its argument.
    std::vector<uint8_t> blob_bytes(const Item & it, Tracker & tr) {
        const Expr & top = *it.expr;
        std::vector<uint8_t> out;
        if (it.dtype == BlobType::fp16) {
            V x = eval(*top.args[0]);
            out.resize(x->f.size() * 2);
            tr.alloc(out.size());
            for (size_t i = 0; i < x->f.size(); ++i) {
                const float v = x->f[i];
                const uint16_t h = f32_to_f16(v);
                if (!std::isfinite(v) || !finite16(h)) {
                    throw EvalError(Error::deterministic("recipe:" + std::to_string(it.line) +
                                                         ": non-finite fp16 weight at element " + std::to_string(i)));
                }
                std::memcpy(out.data() + 2 * i, &h, 2);
            }
        } else {
            V x = call(top);  // f32(...)
            std::vector<float> one;
            const std::vector<float> * src = &x->f;
            if (x->dt == DType::weak) {
                one.push_back(static_cast<float>(x->weak));
                src = &one;
            }
            out.resize(src->size() * 4);
            tr.alloc(out.size());
            for (size_t i = 0; i < src->size(); ++i) {
                if (!std::isfinite((*src)[i])) {
                    throw EvalError(Error::deterministic("recipe:" + std::to_string(it.line) +
                                                         ": non-finite fp32 weight at element " + std::to_string(i)));
                }
            }
            std::memcpy(out.data(), src->data(), out.size());
        }
        return out;
    }

private:
    void bump(const std::string & n, size_t b) {
        auto [it, inserted] = last_use_.emplace(n, b);
        if (!inserted) it->second = std::max(it->second, b);
    }

    static void names(const Expr & e, std::set<std::string> & out) {
        if (e.kind == Expr::Kind::name) out.insert(e.text);
        for (const ExprP & a : e.args) names(*a, out);
    }

    V fresh() { return std::make_shared<Buf>(&tracker); }

    V cast(const V & x, DType to) {
        if (x->dt == to) return x;
        V o = fresh();
        if (x->dt == DType::weak) {
            o->alloc(to, {});
            if (to == DType::f32) o->f[0] = static_cast<float>(x->weak);
            if (to == DType::f64) o->g[0] = x->weak;
            return o;
        }
        o->alloc(to, x->shape);
        if (to == DType::f32) {
            for (size_t i = 0; i < x->g.size(); ++i) o->f[i] = static_cast<float>(x->g[i]);
        } else {
            for (size_t i = 0; i < x->f.size(); ++i) o->g[i] = static_cast<double>(x->f[i]);
        }
        return o;
    }

    template <class T, class F>
    static void broadcast_loop(const T * a, const std::vector<int64_t> & sa, const T * b, const std::vector<int64_t> & sb,
                               T * out, const std::vector<int64_t> & so, F f) {
        const size_t n = count_of(so);
        auto same = [&](const std::vector<int64_t> & s) { return s == so; };
        auto scalar = [](const std::vector<int64_t> & s) {
            for (int64_t d : s) {
                if (d != 1) return false;
            }
            return true;
        };
        if (same(sa) && same(sb)) {
            for (size_t i = 0; i < n; ++i) out[i] = f(a[i], b[i]);
            return;
        }
        if (same(sa) && scalar(sb)) {
            for (size_t i = 0; i < n; ++i) out[i] = f(a[i], b[0]);
            return;
        }
        if (scalar(sa) && same(sb)) {
            for (size_t i = 0; i < n; ++i) out[i] = f(a[0], b[i]);
            return;
        }
        const size_t rank = so.size();
        auto aligned = [&](const std::vector<int64_t> & s) {
            std::vector<size_t> st(rank, 0);
            const std::vector<size_t> own = strides_of(s);
            for (size_t i = 0; i < s.size(); ++i) {
                const size_t d = rank - s.size() + i;
                st[d] = s[i] == 1 ? 0 : own[i];
            }
            return st;
        };
        const std::vector<size_t> ta = aligned(sa), tb = aligned(sb);
        std::vector<int64_t> idx(rank, 0);
        size_t oa = 0, ob = 0;
        for (size_t i = 0; i < n; ++i) {
            out[i] = f(a[oa], b[ob]);
            for (size_t d = rank; d-- > 0;) {
                if (++idx[d] < so[d]) {
                    oa += ta[d];
                    ob += tb[d];
                    break;
                }
                oa -= ta[d] * static_cast<size_t>(idx[d] - 1);
                ob -= tb[d] * static_cast<size_t>(idx[d] - 1);
                idx[d] = 0;
            }
        }
    }

    V binary(const Expr & e, V a, V b) {
        const std::string & op = e.text;
        auto apply = [&](auto x, auto y) {
            using T = decltype(x);
            if (op == "add") return static_cast<T>(x + y);
            if (op == "mul") return static_cast<T>(x * y);
            return static_cast<T>(x / y);
        };
        if (a->dt == DType::weak && b->dt == DType::weak) {
            V o = fresh();
            o->weak = apply(a->weak, b->weak);
            return o;
        }
        const DType dt = (a->dt == DType::f64 || b->dt == DType::f64) ? DType::f64 : DType::f32;
        a = cast(a, dt);
        b = cast(b, dt);
        V o = fresh();
        o->alloc(dt, e.shape);
        if (dt == DType::f32) {
            broadcast_loop<float>(a->f.data(), a->shape, b->f.data(), b->shape, o->f.data(), o->shape,
                                  [&](float x, float y) { return apply(x, y); });
        } else {
            broadcast_loop<double>(a->g.data(), a->shape, b->g.data(), b->shape, o->g.data(), o->shape,
                                   [&](double x, double y) { return apply(x, y); });
        }
        return o;
    }

    // Shape-only ops keep the dtype and move the data when the input is a
    // temporary.
    template <class Fn>
    V relayout(const V & x, const std::vector<int64_t> & shape, Fn fn) {
        V o = fresh();
        o->alloc(x->dt, shape);
        if (x->dt == DType::f32) {
            fn(x->f.data(), o->f.data());
        } else {
            fn(x->g.data(), o->g.data());
        }
        return o;
    }

    template <class T>
    static void permute_loop(const T * in, const std::vector<int64_t> & in_shape, const std::vector<int64_t> & axes,
                             T * out, const std::vector<int64_t> & out_shape) {
        const std::vector<size_t> is = strides_of(in_shape);
        const size_t rank = out_shape.size();
        std::vector<size_t> st(rank);
        for (size_t k = 0; k < rank; ++k) st[k] = is[static_cast<size_t>(axes[k])];
        const size_t n = count_of(out_shape);
        std::vector<int64_t> idx(rank, 0);
        size_t off = 0;
        for (size_t i = 0; i < n; ++i) {
            out[i] = in[off];
            for (size_t d = rank; d-- > 0;) {
                if (++idx[d] < out_shape[d]) {
                    off += st[d];
                    break;
                }
                off -= st[d] * static_cast<size_t>(idx[d] - 1);
                idx[d] = 0;
            }
        }
    }

    static std::vector<int64_t> ints(const Expr & list) {
        std::vector<int64_t> v;
        for (const ExprP & a : list.args) v.push_back(static_cast<int64_t>(a->number));
        return v;
    }

    V read_tensor(const Expr & e) {
        const std::string & name = e.args[0]->text;
        const Source * s = r_.find_source(name);
        const TensorInfo * ti = src_.tensor(name);
        std::array<int64_t, 4> want{{1, 1, 1, 1}};
        for (size_t d = 0; d < s->shape.size(); ++d) want[d] = s->shape[s->shape.size() - 1 - d];
        if (!ti || ti->ne != want || std::find(s->types.begin(), s->types.end(), ti->type) == s->types.end()) {
            throw EvalError(Error::deterministic("recipe:" + std::to_string(e.line) + ": tensor " + name +
                                                 " does not match its source line"));
        }
        V o = fresh();
        o->dt = DType::f32;
        o->shape = s->shape;
        detail::Sha256 h;
        RawTap tap;
        if (obs_) tap = [&h](const uint8_t * p, size_t n) { h.update(p, n); };
        Error err = src_.read_f32(name, o->f, tap);
        if (!err.ok()) throw EvalError(err);
        o->account(o->f.size() * sizeof(float));
        const size_t n = count_of(s->shape);
        if (o->f.size() != n) {
            throw EvalError(Error::deterministic("recipe: tensor " + name + " read " + std::to_string(o->f.size()) +
                                                 " values, its shape has " + std::to_string(n)));
        }
        ++tensors_read;
        if (obs_) {
            err = obs_(name, h.finish());
            if (!err.ok()) throw EvalError(err);
        }
        return o;
    }

    V call(const Expr & e) {
        const std::string & op = e.text;
        if (op == "t") return read_tensor(e);
        if (op == "f32") return cast(eval(*e.args[0]), DType::f32);
        if (op == "f64") return cast(eval(*e.args[0]), DType::f64);
        if (op == "add" || op == "mul" || op == "div") return binary(e, eval(*e.args[0]), eval(*e.args[1]));
        if (op == "recip") {
            V x = eval(*e.args[0]);
            V one = fresh();
            one->weak = 1.0;
            return binary(Expr{Expr::Kind::call, 0.0, "div", {}, e.line, e.dtype, e.shape}, one, x);
        }
        if (op == "clip") {
            V x = eval(*e.args[0]);
            const double lo = e.args[1]->number, hi = e.args[2]->number;
            V o = fresh();
            if (x->dt == DType::weak) {
                if (std::isnan(x->weak)) throw EvalError(Error::deterministic("recipe: NaN in clip()"));
                o->weak = std::min(std::max(x->weak, lo), hi);
                return o;
            }
            o->alloc(x->dt, x->shape);
            auto run = [&](auto * in, auto * out, size_t n) {
                using T = std::remove_pointer_t<decltype(out)>;
                const T l = static_cast<T>(lo), u = static_cast<T>(hi);
                for (size_t i = 0; i < n; ++i) {
                    if (std::isnan(in[i])) {
                        throw EvalError(Error::deterministic("recipe:" + std::to_string(e.line) + ": NaN in clip()"));
                    }
                    out[i] = std::min(std::max(in[i], l), u);
                }
            };
            if (x->dt == DType::f32) {
                run(x->f.data(), o->f.data(), x->f.size());
            } else {
                run(x->g.data(), o->g.data(), x->g.size());
            }
            return o;
        }
        if (op == "reshape") {
            V x = eval(*e.args[0]);
            const size_t n = x->size();
            return relayout(x, e.shape, [n](const auto * in, auto * out) { std::copy(in, in + n, out); });
        }
        if (op == "permute") {
            V x = eval(*e.args[0]);
            const std::vector<int64_t> axes = ints(*e.args[1]);
            return relayout(x, e.shape, [&](const auto * in, auto * out) {
                permute_loop(in, x->shape, axes, out, e.shape);
            });
        }
        if (op == "take") {
            V x = eval(*e.args[0]);
            const size_t axis = static_cast<size_t>(e.args[1]->number);
            const std::vector<int64_t> idx = ints(*e.args[2]);
            const size_t outer = count_of(x->shape, 0, axis);
            const size_t inner = count_of(x->shape, axis + 1);
            const size_t dim = static_cast<size_t>(x->shape[axis]);
            return relayout(x, e.shape, [&](const auto * in, auto * out) {
                size_t w = 0;
                for (size_t o = 0; o < outer; ++o) {
                    for (int64_t i : idx) {
                        if (i >= 0) {
                            std::copy(in + (o * dim + static_cast<size_t>(i)) * inner,
                                      in + (o * dim + static_cast<size_t>(i) + 1) * inner, out + w);
                        }  // index -1 yields zeros (the buffer is zero-filled)
                        w += inner;
                    }
                }
            });
        }
        if (op == "concat") {
            const size_t axis = static_cast<size_t>(e.args[0]->number);
            std::vector<V> parts;
            for (size_t i = 1; i < e.args.size(); ++i) parts.push_back(cast(eval(*e.args[i]), e.dtype));
            V o = fresh();
            o->alloc(e.dtype, e.shape);
            const size_t outer = count_of(e.shape, 0, axis);
            const size_t inner = count_of(e.shape, axis + 1);
            size_t w = 0;
            for (size_t ot = 0; ot < outer; ++ot) {
                for (const V & p : parts) {
                    const size_t blk = static_cast<size_t>(p->shape[axis]) * inner;
                    if (e.dtype == DType::f32) {
                        std::copy(p->f.begin() + ot * blk, p->f.begin() + (ot + 1) * blk, o->f.begin() + w);
                    } else {
                        std::copy(p->g.begin() + ot * blk, p->g.begin() + (ot + 1) * blk, o->g.begin() + w);
                    }
                    w += blk;
                }
            }
            return o;
        }
        if (op == "slice") {
            V x = eval(*e.args[0]);
            const size_t axis = static_cast<size_t>(e.args[1]->number);
            const size_t begin = static_cast<size_t>(e.args[2]->number);
            const size_t step = e.args.size() == 5 ? static_cast<size_t>(e.args[4]->number) : 1;
            const size_t outer = count_of(x->shape, 0, axis);
            const size_t inner = count_of(x->shape, axis + 1);
            const size_t dim = static_cast<size_t>(x->shape[axis]);
            const size_t n = static_cast<size_t>(e.shape[axis]);
            return relayout(x, e.shape, [&](const auto * in, auto * out) {
                size_t w = 0;
                for (size_t o = 0; o < outer; ++o) {
                    for (size_t k = 0; k < n; ++k) {
                        const size_t i = begin + k * step;
                        std::copy(in + (o * dim + i) * inner, in + (o * dim + i + 1) * inner, out + w);
                        w += inner;
                    }
                }
            });
        }
        if (op == "neg") {
            V x = eval(*e.args[0]);
            V o = fresh();
            if (x->dt == DType::weak) {
                o->weak = -x->weak;
                return o;
            }
            o->alloc(x->dt, x->shape);
            for (size_t i = 0; i < x->f.size(); ++i) o->f[i] = -x->f[i];
            for (size_t i = 0; i < x->g.size(); ++i) o->g[i] = -x->g[i];
            return o;
        }
        if (op == "zeros" || op == "ones") {
            V o = fresh();
            o->alloc(DType::f32, e.shape);
            if (op == "ones") std::fill(o->f.begin(), o->f.end(), 1.0f);
            return o;
        }
        if (op == "lit") {
            const Literal * l = r_.find_literal(e.args[0]->text);
            if (l == nullptr || l->data == nullptr) {
                throw EvalError(Error::deterministic("recipe:" + std::to_string(e.line) + ": literal " +
                                                     e.args[0]->text + " has no bytes"));
            }
            V o = fresh();
            o->alloc(DType::f32, l->shape);
            if (l->dtype == BlobType::fp16) {
                for (size_t i = 0; i < o->f.size(); ++i) {
                    uint16_t h;
                    std::memcpy(&h, l->data + 2 * i, 2);
                    o->f[i] = f16_to_f32(h);
                }
            } else {
                std::memcpy(o->f.data(), l->data, o->f.size() * sizeof(float));
            }
            return o;
        }
        throw EvalError(Error::deterministic("recipe: op " + op + "() cannot be evaluated"));
    }

    const Recipe & r_;
    TensorSource & src_;
    const ReadObserver & obs_;
    Item::Kind driver_;
    std::map<std::string, const Item *> lets_;
    std::map<std::string, V> cache_;
    std::map<std::string, size_t> last_use_;
};

}  // namespace

Error evaluate(const Recipe & r, TensorSource & src, const std::function<Error(const BlobOut &)> & sink,
               EvalStats * stats, const ReadObserver & observer) {
    Interp in(r, src, observer);
    in.plan_liveness();
    size_t b = 0;
    Error result;
    try {
        for (const Item & it : r.items) {
            if (it.kind != Item::Kind::blob) continue;
            std::vector<uint8_t> bytes = in.blob_bytes(it, in.tracker);
            BlobOut out;
            out.offset = it.offset;
            out.dtype = it.dtype;
            out.count = it.count;
            out.data = bytes.data();
            out.bytes = bytes.size();
            out.ulp = it.ulp;
            out.line = it.line;
            Error e = sink(out);
            in.tracker.release(bytes.size());
            if (!e.ok()) {
                result = e;
                break;
            }
            in.free_after(b);
            ++b;
        }
    } catch (const EvalError & e) {
        result = e.err;
    } catch (const std::bad_alloc &) {
        result = Error::transient("recipe: out of memory", "qvac.coreml", 12 /* ENOMEM */);
    }
    if (stats) {
        stats->peak_bytes = in.tracker.peak;
        stats->tensors_read = in.tensors_read;
    }
    return result;
}

Error evaluate_patches(const Recipe & r, TensorSource & src, const std::function<Error(const PatchOut &)> & sink,
                       const ReadObserver & observer, EvalStats * stats) {
    Interp in(r, src, observer, Item::Kind::patch);
    in.plan_liveness();
    struct Report {
        Interp & in;
        EvalStats * stats;
        ~Report() {
            if (stats) {
                stats->peak_bytes = in.tracker.peak;
                stats->tensors_read = in.tensors_read;
            }
        }
    } report{in, stats};
    size_t p = 0;
    try {
        for (const Item & it : r.items) {
            if (it.kind != Item::Kind::patch) continue;
            std::vector<uint8_t> bytes = in.blob_bytes(it, in.tracker);
            PatchOut out;
            out.offset = it.offset;
            out.dtype = it.dtype;
            out.count = it.count;
            out.data = bytes.data();
            out.bytes = bytes.size();
            out.line = it.line;
            Error e = sink(out);
            in.tracker.release(bytes.size());
            if (!e.ok()) return e;
            in.free_after(p);
            ++p;
        }
    } catch (const EvalError & e) {
        return e.err;
    } catch (const std::bad_alloc &) {
        return Error::transient("recipe: out of memory", "qvac.coreml", 12 /* ENOMEM */);
    }
    return {};
}

Error patched_spec(const Recipe & r, TensorSource & src, const uint8_t * spec, size_t spec_size,
                   std::vector<uint8_t> & out, const ReadObserver & observer, EvalStats * stats) {
    out.assign(spec, spec + spec_size);
    struct Report {
        EvalStats * stats;
        size_t spec_size;
        ~Report() {
            if (stats) stats->peak_bytes += spec_size;  // the copy is live through the pass
        }
    } report{stats, spec_size};
    return evaluate_patches(
        r, src,
        [&](const PatchOut & p) -> Error {
            if (p.offset > out.size() || p.bytes > out.size() - p.offset) {
                return Error::deterministic("recipe:" + std::to_string(p.line) + ": patch outside the spec");
            }
            std::memcpy(out.data() + p.offset, p.data, p.bytes);
            return {};
        },
        observer, stats);
}

}  // namespace recipe
}  // namespace QVAC_COREML_NS
