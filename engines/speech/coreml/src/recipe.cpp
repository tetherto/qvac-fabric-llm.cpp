// Recipe parser, static checker, templates and template matching.

#include "qvac_coreml/recipe.h"

#include "milblob.h"
#include "sha256.h"
#include "shape.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <sstream>

namespace QVAC_COREML_NS {
namespace recipe {

using detail::Sha256;
using detail::to_hex;

const char * to_string(DType d) {
    switch (d) {
        case DType::weak: return "weak";
        case DType::f32: return "f32";
        case DType::f64: return "f64";
    }
    return "?";
}

const char * to_string(BlobType b) { return b == BlobType::fp16 ? "fp16" : "fp32"; }

size_t blob_type_size(BlobType b) { return b == BlobType::fp16 ? 2 : 4; }

const Source * Recipe::find_source(std::string_view name) const {
    for (const Source & s : sources) {
        if (s.name == name) return &s;
    }
    return nullptr;
}

const Literal * Recipe::find_literal(std::string_view id) const {
    for (const Literal & l : literals) {
        if (l.id == id) return &l;
    }
    return nullptr;
}

size_t Recipe::patch_count() const {
    size_t n = 0;
    for (const Item & it : items) n += it.kind == Item::Kind::patch ? 1 : 0;
    return n;
}

// ---- ggml / gguf type tables -----------------------------------------------

namespace {

struct TypeName {
    uint32_t type;
    const char * name;
};

// enum ggml_type ABI values (ggml.h); gguf_source.cpp static_asserts them.
const TypeName kGgmlTypes[] = {
    {0, "f32"},     {1, "f16"},     {2, "q4_0"},    {3, "q4_1"},    {6, "q5_0"},    {7, "q5_1"},
    {8, "q8_0"},    {9, "q8_1"},    {10, "q2_k"},   {11, "q3_k"},   {12, "q4_k"},   {13, "q5_k"},
    {14, "q6_k"},   {15, "q8_k"},   {20, "iq4_nl"}, {23, "iq4_xs"}, {28, "f64"},    {30, "bf16"},
};

}  // namespace

std::optional<uint32_t> ggml_type_from_name(std::string_view name) {
    for (const TypeName & t : kGgmlTypes) {
        if (name == t.name) return t.type;
    }
    return std::nullopt;
}

const char * ggml_type_name_of(uint32_t type) {
    for (const TypeName & t : kGgmlTypes) {
        if (t.type == type) return t.name;
    }
    return "?";
}

std::optional<uint32_t> gguf_type_of_mtype(std::string_view m, uint32_t * elem) {
    if (elem) *elem = 0;
    if (m == "u32") return GGUF_U32;
    if (m == "i32") return GGUF_I32;
    if (m == "f32") return GGUF_F32;
    if (m == "f64") return GGUF_F64;
    if (m == "bool") return GGUF_BOOL;
    if (m == "str") return GGUF_STRING;
    if (m == "arr_u32" || m == "arr_f32") {
        if (elem) *elem = m == "arr_u32" ? GGUF_U32 : GGUF_F32;
        return GGUF_ARRAY;
    }
    return std::nullopt;
}

// ---- Lexing ------------------------------------------------------------------

namespace {

// The largest value Cursor::uint() accepts (19 decimal digits).
constexpr uint64_t MAX_UINT_LITERAL = 9999999999999999999ull;

struct Line {
    std::string text;
    int line = 0;  // first physical line
};

// Logical lines: `#` comments removed, a newline inside an open `(` or `[` is
// whitespace, blank lines dropped.
std::vector<Line> logical_lines(std::string_view src) {
    std::vector<Line> out;
    std::string cur;
    int line = 1;
    int start = 1;
    int depth = 0;
    bool in_str = false;
    bool in_comment = false;
    auto flush = [&]() {
        size_t b = cur.find_first_not_of(" \t\r");
        if (b != std::string::npos) {
            size_t e = cur.find_last_not_of(" \t\r");
            out.push_back({cur.substr(b, e - b + 1), start});
        }
        cur.clear();
    };
    for (size_t i = 0; i < src.size(); ++i) {
        const char c = src[i];
        if (c == '\n') {
            if (in_str) throw RecipeError(line, "unterminated string");
            in_comment = false;
            ++line;
            if (depth > 0) {
                cur.push_back(' ');
            } else {
                flush();
                start = line;
            }
            continue;
        }
        if (in_comment) continue;
        if (in_str) {
            if (c == '"') in_str = false;
            cur.push_back(c);
            continue;
        }
        if (c == '#') {
            in_comment = true;
            continue;
        }
        if (c == '"') in_str = true;
        if (c == '(' || c == '[') ++depth;
        if (c == ')' || c == ']') {
            if (depth == 0) throw RecipeError(line, std::string("unbalanced '") + c + "'");
            --depth;
        }
        if (cur.empty()) start = line;
        cur.push_back(c);
    }
    if (in_str) throw RecipeError(line, "unterminated string");
    if (depth > 0) throw RecipeError(start, "unclosed '(' or '[' at end of recipe");
    flush();
    return out;
}

bool is_name_start(char c) { return (c >= 'a' && c <= 'z') || c == '_'; }
bool is_name_char(char c) { return is_name_start(c) || (c >= '0' && c <= '9'); }
bool is_digit(char c) { return c >= '0' && c <= '9'; }

class Cursor {
public:
    Cursor(const Line & l) : s_(l.text), line_(l.line) {}

    [[noreturn]] void fail(const std::string & msg) const { throw RecipeError(line_, msg); }
    int line() const { return line_; }

    void ws() {
        while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\r')) ++p_;
    }
    // SP: at least one space or tab.
    void sp() {
        if (p_ >= s_.size() || (s_[p_] != ' ' && s_[p_] != '\t')) fail("expected a space at column " + col());
        ws();
    }
    bool at_end() {
        ws();
        return p_ >= s_.size();
    }
    void end() {
        if (!at_end()) fail("unexpected '" + s_.substr(p_) + "'");
    }
    char peek() {
        ws();
        return p_ < s_.size() ? s_[p_] : '\0';
    }
    bool eat(char c) {
        if (peek() == c) {
            ++p_;
            return true;
        }
        return false;
    }
    void expect(char c) {
        if (!eat(c)) fail(std::string("expected '") + c + "' at column " + col());
    }
    // A run of non-space characters.
    std::string word() {
        ws();
        size_t b = p_;
        while (p_ < s_.size() && s_[p_] != ' ' && s_[p_] != '\t') ++p_;
        if (b == p_) fail("expected a word at column " + col());
        return s_.substr(b, p_ - b);
    }
    std::string name() {
        ws();
        size_t b = p_;
        if (p_ >= s_.size() || !is_name_start(s_[p_])) fail("expected a name at column " + col());
        while (p_ < s_.size() && is_name_char(s_[p_])) ++p_;
        return s_.substr(b, p_ - b);
    }
    std::string string() {
        ws();
        if (p_ >= s_.size() || s_[p_] != '"') fail("expected a string at column " + col());
        size_t e = s_.find('"', p_ + 1);
        if (e == std::string::npos) fail("unterminated string");
        std::string v = s_.substr(p_ + 1, e - p_ - 1);
        p_ = e + 1;
        return v;
    }
    uint64_t uint() {
        ws();
        size_t b = p_;
        while (p_ < s_.size() && is_digit(s_[p_])) ++p_;
        if (b == p_) fail("expected an unsigned integer at column " + col());
        if (p_ - b > 19) fail("integer too large");  // so every value is at most MAX_UINT_LITERAL
        return std::stoull(s_.substr(b, p_ - b));
    }
    bool number_start() {
        ws();
        if (p_ >= s_.size()) return false;
        const char c = s_[p_];
        if (is_digit(c)) return true;
        const char n = p_ + 1 < s_.size() ? s_[p_ + 1] : '\0';
        if (c == '.') return is_digit(n);
        if (c == '-' || c == '+') return is_digit(n) || (n == '.' && p_ + 2 < s_.size() && is_digit(s_[p_ + 2]));
        return false;
    }
    // C strtod decimal syntax, no hex floats, no inf/nan; locale-independent.
    double number() {
        ws();
        size_t b = p_;
        if (p_ < s_.size() && (s_[p_] == '-' || s_[p_] == '+')) ++p_;
        size_t digits = 0;
        while (p_ < s_.size() && is_digit(s_[p_])) ++p_, ++digits;
        if (p_ < s_.size() && s_[p_] == '.') {
            ++p_;
            while (p_ < s_.size() && is_digit(s_[p_])) ++p_, ++digits;
        }
        if (digits == 0) fail("expected a number at column " + col());
        if (p_ < s_.size() && (s_[p_] == 'e' || s_[p_] == 'E')) {
            size_t q = p_ + 1;
            if (q < s_.size() && (s_[q] == '-' || s_[q] == '+')) ++q;
            size_t ed = 0;
            while (q < s_.size() && is_digit(s_[q])) ++q, ++ed;
            if (ed == 0) fail("malformed exponent at column " + col());
            p_ = q;
        }
        if (p_ < s_.size() && (std::isalpha(static_cast<unsigned char>(s_[p_])) || s_[p_] == '.')) {
            fail("malformed number '" + s_.substr(b, p_ - b + 1) + "'");
        }
        std::istringstream in(s_.substr(b, p_ - b));
        in.imbue(std::locale::classic());
        double v = 0.0;
        in >> v;
        if (!in || !std::isfinite(v)) fail("number out of range: " + s_.substr(b, p_ - b));
        return v;
    }
    std::string col() const { return std::to_string(p_ + 1); }

private:
    std::string s_;
    size_t p_ = 0;
    int line_;
};

// ---- Expressions ---------------------------------------------------------------

ExprP parse_expr(Cursor & c, int depth = 0) {
    if (depth > 256) c.fail("expression nested too deeply");
    auto e = std::make_shared<Expr>();
    e->line = c.line();
    const char ch = c.peek();
    if (ch == '"') {
        e->kind = Expr::Kind::string;
        e->text = c.string();
        return e;
    }
    if (ch == '[') {
        c.expect('[');
        e->kind = Expr::Kind::list;
        if (!c.eat(']')) {
            do {
                e->args.push_back(parse_expr(c, depth + 1));
            } while (c.eat(','));
            c.expect(']');
        }
        return e;
    }
    if (c.number_start()) {
        e->kind = Expr::Kind::number;
        e->number = c.number();
        return e;
    }
    const std::string n = c.name();
    if (c.peek() == '(') {
        c.expect('(');
        e->kind = Expr::Kind::call;
        e->text = n;
        if (!c.eat(')')) {
            do {
                e->args.push_back(parse_expr(c, depth + 1));
            } while (c.eat(','));
            c.expect(')');
        }
        return e;
    }
    e->kind = Expr::Kind::name;
    e->text = n;
    return e;
}

// Ops of this generator version. Later generator versions add more.
const std::set<std::string> kOps = {"t",    "f32",   "f64",  "f16",  "add",   "mul", "div",   "recip", "clip",
                                    "reshape", "permute", "take", "concat", "slice", "neg", "zeros", "ones", "lit"};
// Ops that exist in the language but belong to a later generator version.
const std::set<std::string> kLaterOps = {"sub", "sqrt", "full",   "tile",   "exp",
                                         "sin", "cos",  "l2norm", "matmul", "meta_f", "meta_s", "provided"};

std::string shape_str(const std::vector<int64_t> & s) {
    std::string o = "[";
    for (size_t i = 0; i < s.size(); ++i) o += (i ? "," : "") + std::to_string(s[i]);
    return o + "]";
}

std::string too_many_elements(const std::vector<int64_t> & shape) {
    return "shape " + shape_str(shape) + " has more than " + std::to_string(detail::MAX_ELEMENTS) + " elements";
}

// Element count of a shape; throws RecipeError when it exceeds MAX_ELEMENTS.
int64_t elements(const std::vector<int64_t> & shape, int line) {
    const std::optional<uint64_t> n = detail::element_count(shape);
    if (!n) throw RecipeError(line, too_many_elements(shape));
    return static_cast<int64_t>(*n);
}

class Checker {
public:
    explicit Checker(const Recipe & r) : r_(r) {}

    void define(const std::string & name, const ExprP & e, int line) {
        if (lets_.count(name)) throw RecipeError(line, "'" + name + "' is already defined (no redefinition)");
        if (kOps.count(name) || kLaterOps.count(name)) throw RecipeError(line, "'" + name + "' is an op name");
        lets_[name] = e;
    }

    void value(Expr & e) {
        switch (e.kind) {
            case Expr::Kind::number:
                e.dtype = DType::weak;
                e.shape.clear();
                return;
            case Expr::Kind::string: throw RecipeError(e.line, "a string is not a value");
            case Expr::Kind::list: throw RecipeError(e.line, "a list is not a value");
            case Expr::Kind::name: {
                auto it = lets_.find(e.text);
                if (it == lets_.end()) throw RecipeError(e.line, "'" + e.text + "' is used before its let");
                e.dtype = it->second->dtype;
                e.shape = it->second->shape;
                e.ulp = it->second->ulp;
                return;
            }
            case Expr::Kind::call:
                call(e);
                (void) elements(e.shape, e.line);  // every op's result shape is bounded
                return;
        }
    }

    // The top cast of a blob: f16(f32 array) for fp16, f32(x) for fp32.
    void blob_top(Expr & e, BlobType dt) {
        const char * want = dt == BlobType::fp16 ? "f16" : "f32";
        if (e.kind != Expr::Kind::call || e.text != want) {
            throw RecipeError(e.line, std::string("a ") + to_string(dt) + " blob must end in " + want + "(...)");
        }
        if (dt == BlobType::fp32) {
            value(e);
            return;
        }
        arity(e, 1);
        value(*e.args[0]);
        if (e.args[0]->dtype != DType::f32) {
            throw RecipeError(e.line, std::string("f16() takes f32, not ") + to_string(e.args[0]->dtype) +
                                          " (cast explicitly, e.g. f16(f32(x)))");
        }
        e.dtype = DType::f32;  // nominal; the blob dtype carries fp16
        e.shape = e.args[0]->shape;
        e.ulp = e.args[0]->ulp;
    }

private:
    static void arity(const Expr & e, size_t n) {
        if (e.args.size() != n) {
            throw RecipeError(e.line, e.text + "() takes " + std::to_string(n) + " argument(s), got " +
                                          std::to_string(e.args.size()));
        }
    }

    static int64_t integer(const Expr & e, const char * what) {
        if (e.kind != Expr::Kind::number || std::floor(e.number) != e.number || std::fabs(e.number) > 9.0e15) {
            throw RecipeError(e.line, std::string(what) + " must be an integer literal");
        }
        return static_cast<int64_t>(e.number);
    }

    static std::vector<int64_t> int_list(const Expr & e, const char * what) {
        if (e.kind != Expr::Kind::list) throw RecipeError(e.line, std::string(what) + " must be a list");
        std::vector<int64_t> v;
        for (const ExprP & a : e.args) v.push_back(integer(*a, what));
        return v;
    }

    void array(Expr & e, const char * op) {
        value(e);
        if (e.dtype == DType::weak) throw RecipeError(e.line, std::string(op) + "() needs an array, not a literal");
    }

    static DType promote(DType a, DType b) {
        if (a == DType::weak) return b;
        if (b == DType::weak) return a;
        return (a == DType::f64 || b == DType::f64) ? DType::f64 : DType::f32;
    }

    static std::vector<int64_t> broadcast(const std::vector<int64_t> & a, const std::vector<int64_t> & b, int line) {
        const size_t n = std::max(a.size(), b.size());
        std::vector<int64_t> out(n);
        for (size_t i = 0; i < n; ++i) {
            const int64_t x = i < n - a.size() ? 1 : a[i - (n - a.size())];
            const int64_t y = i < n - b.size() ? 1 : b[i - (n - b.size())];
            if (x != y && x != 1 && y != 1) {
                throw RecipeError(line, "shapes " + shape_str(a) + " and " + shape_str(b) + " do not broadcast");
            }
            out[i] = x == 1 ? y : x;
        }
        return out;
    }

    void call(Expr & e) {
        const std::string & op = e.text;
        if (!kOps.count(op)) {
            if (kLaterOps.count(op)) {
                throw RecipeError(e.line, "op " + op + "() is not in generator version " +
                                              std::to_string(GENERATOR_VERSION) + " (a later version adds it)");
            }
            throw RecipeError(e.line, "unknown op " + op + "()");
        }
        if (op == "t") {
            arity(e, 1);
            const Expr & a = *e.args[0];
            if (a.kind != Expr::Kind::string) throw RecipeError(e.line, "t() takes a tensor name string");
            const Source * s = r_.find_source(a.text);
            if (!s) throw RecipeError(e.line, "t(\"" + a.text + "\") has no source line");
            e.dtype = DType::f32;
            e.shape = s->shape;
            return;
        }
        if (op == "f32" || op == "f64") {
            arity(e, 1);
            value(*e.args[0]);
            e.dtype = op == "f32" ? DType::f32 : DType::f64;
            e.shape = e.args[0]->shape;
            e.ulp = e.args[0]->ulp;
            return;
        }
        if (op == "f16") throw RecipeError(e.line, "f16() is only the top cast of an fp16 blob");
        if (op == "add" || op == "mul" || op == "div") {
            arity(e, 2);
            value(*e.args[0]);
            value(*e.args[1]);
            e.dtype = promote(e.args[0]->dtype, e.args[1]->dtype);
            e.shape = broadcast(e.args[0]->shape, e.args[1]->shape, e.line);
            e.ulp = e.args[0]->ulp || e.args[1]->ulp;
            return;
        }
        if (op == "recip") {
            arity(e, 1);
            value(*e.args[0]);
            e.dtype = e.args[0]->dtype;
            e.shape = e.args[0]->shape;
            e.ulp = e.args[0]->ulp;
            return;
        }
        if (op == "clip") {
            arity(e, 3);
            value(*e.args[0]);
            for (int i = 1; i <= 2; ++i) {
                if (e.args[i]->kind != Expr::Kind::number) throw RecipeError(e.line, "clip() bounds must be literals");
            }
            if (e.args[1]->number > e.args[2]->number) throw RecipeError(e.line, "clip() lower bound above upper");
            e.dtype = e.args[0]->dtype;
            e.shape = e.args[0]->shape;
            e.ulp = e.args[0]->ulp;
            return;
        }
        if (op == "reshape") {
            arity(e, 2);
            array(*e.args[0], "reshape");
            std::vector<int64_t> dims = int_list(*e.args[1], "reshape dims");
            if (dims.empty() || dims.size() > 8) throw RecipeError(e.line, "reshape() needs 1 to 8 dims");
            int neg = -1;
            int64_t known = 1;
            for (size_t i = 0; i < dims.size(); ++i) {
                if (dims[i] == -1) {
                    if (neg >= 0) throw RecipeError(e.line, "reshape() allows one -1");
                    neg = static_cast<int>(i);
                } else if (dims[i] <= 0) {
                    throw RecipeError(e.line, "reshape() dims must be positive or -1");
                } else {
                    const std::optional<uint64_t> k = detail::mul_elements(static_cast<uint64_t>(known),
                                                                           static_cast<uint64_t>(dims[i]));
                    if (!k) throw RecipeError(e.line, "reshape() dims " + too_many_elements(dims));
                    known = static_cast<int64_t>(*k);
                }
            }
            const int64_t n = elements(e.args[0]->shape, e.line);
            if (neg >= 0) {
                if (n % known != 0) throw RecipeError(e.line, "reshape() -1 does not divide " + std::to_string(n));
                dims[neg] = n / known;
            }
            if (elements(dims, e.line) != n) {
                throw RecipeError(e.line, "reshape() " + shape_str(e.args[0]->shape) + " -> " + shape_str(dims) +
                                              " changes the element count");
            }
            e.dtype = e.args[0]->dtype;
            e.shape = dims;
            e.ulp = e.args[0]->ulp;
            return;
        }
        if (op == "permute") {
            arity(e, 2);
            array(*e.args[0], "permute");
            const std::vector<int64_t> axes = int_list(*e.args[1], "permute axes");
            const std::vector<int64_t> & in = e.args[0]->shape;
            if (axes.size() != in.size()) throw RecipeError(e.line, "permute() needs one axis per dim");
            std::vector<bool> seen(in.size(), false);
            e.shape.clear();
            for (int64_t a : axes) {
                if (a < 0 || a >= static_cast<int64_t>(in.size()) || seen[a]) {
                    throw RecipeError(e.line, "permute() axes must be a permutation of 0.." +
                                                  std::to_string(in.size() - 1));
                }
                seen[a] = true;
                e.shape.push_back(in[a]);
            }
            e.dtype = e.args[0]->dtype;
            e.ulp = e.args[0]->ulp;
            return;
        }
        if (op == "take") {
            arity(e, 3);
            array(*e.args[0], "take");
            const int64_t axis = integer(*e.args[1], "take axis");
            const std::vector<int64_t> idx = int_list(*e.args[2], "take indices");
            const std::vector<int64_t> & in = e.args[0]->shape;
            if (axis < 0 || axis >= static_cast<int64_t>(in.size())) throw RecipeError(e.line, "take() axis out of range");
            if (idx.empty()) throw RecipeError(e.line, "take() needs at least one index");
            for (int64_t i : idx) {
                if (i < -1 || i >= in[axis]) {
                    throw RecipeError(e.line, "take() index " + std::to_string(i) + " out of range for dim " +
                                                  std::to_string(in[axis]));
                }
            }
            e.shape = in;
            e.shape[axis] = static_cast<int64_t>(idx.size());
            e.dtype = e.args[0]->dtype;
            e.ulp = e.args[0]->ulp;
            return;
        }
        if (op == "concat") {
            if (e.args.size() < 2) throw RecipeError(e.line, "concat() takes an axis and at least one array");
            const int64_t axis = integer(*e.args[0], "concat axis");
            DType dt = DType::weak;
            std::vector<int64_t> shape;
            for (size_t i = 1; i < e.args.size(); ++i) {
                Expr & a = *e.args[i];
                array(a, "concat");
                if (i == 1) {
                    shape = a.shape;
                    if (axis < 0 || axis >= static_cast<int64_t>(shape.size())) {
                        throw RecipeError(e.line, "concat() axis out of range");
                    }
                } else {
                    if (a.shape.size() != shape.size()) throw RecipeError(e.line, "concat() ranks differ");
                    for (size_t d = 0; d < shape.size(); ++d) {
                        if (static_cast<int64_t>(d) != axis && a.shape[d] != shape[d]) {
                            throw RecipeError(e.line, "concat() shapes " + shape_str(shape) + " and " +
                                                          shape_str(a.shape) + " differ off the axis");
                        }
                    }
                    const std::optional<uint64_t> sum = detail::add_elements(
                        static_cast<uint64_t>(shape[axis]), static_cast<uint64_t>(a.shape[axis]));
                    if (!sum) throw RecipeError(e.line, "concat() axis length overflows");
                    shape[axis] = static_cast<int64_t>(*sum);
                }
                dt = promote(dt, a.dtype);
                e.ulp = e.ulp || a.ulp;
            }
            e.dtype = dt;
            e.shape = shape;
            return;
        }
        if (op == "slice") {
            if (e.args.size() != 4 && e.args.size() != 5) {
                throw RecipeError(e.line, "slice() takes an array, an axis, begin, end and an optional step");
            }
            array(*e.args[0], "slice");
            const std::vector<int64_t> & in = e.args[0]->shape;
            const int64_t axis = integer(*e.args[1], "slice axis");
            if (axis < 0 || axis >= static_cast<int64_t>(in.size())) throw RecipeError(e.line, "slice() axis out of range");
            const int64_t begin = integer(*e.args[2], "slice begin");
            const int64_t end = integer(*e.args[3], "slice end");
            const int64_t step = e.args.size() == 5 ? integer(*e.args[4], "slice step") : 1;
            if (begin < 0 || end <= begin || end > in[axis]) {
                throw RecipeError(e.line, "slice() needs 0 <= begin < end <= " + std::to_string(in[axis]) + ", got [" +
                                              std::to_string(begin) + ", " + std::to_string(end) + ")");
            }
            if (step < 1) throw RecipeError(e.line, "slice() step must be at least 1");
            e.shape = in;
            e.shape[axis] = (end - begin + step - 1) / step;
            e.dtype = e.args[0]->dtype;
            e.ulp = e.args[0]->ulp;
            return;
        }
        if (op == "neg") {
            arity(e, 1);
            value(*e.args[0]);
            e.dtype = e.args[0]->dtype;
            e.shape = e.args[0]->shape;
            e.ulp = e.args[0]->ulp;
            return;
        }
        if (op == "zeros" || op == "ones") {
            arity(e, 1);
            const std::vector<int64_t> dims = int_list(*e.args[0], "zeros/ones dims");
            if (dims.empty() || dims.size() > 8) throw RecipeError(e.line, op + "() needs 1 to 8 dims");
            for (int64_t d : dims) {
                if (d <= 0) throw RecipeError(e.line, op + "() dims must be positive");
            }
            (void) elements(dims, e.line);
            e.dtype = DType::f32;
            e.shape = dims;
            return;
        }
        if (op == "lit") {
            arity(e, 1);
            const Expr & a = *e.args[0];
            if (a.kind != Expr::Kind::string) throw RecipeError(e.line, "lit() takes a literal id string");
            const Literal * l = r_.find_literal(a.text);
            if (!l) throw RecipeError(e.line, "lit(\"" + a.text + "\") is not declared in template.json literals");
            e.dtype = DType::f32;
            e.shape = l->shape;
            return;
        }
        throw RecipeError(e.line, "unknown op " + op + "()");  // unreachable: kOps is exhaustive
    }

    const Recipe & r_;
    std::map<std::string, ExprP> lets_;
};

bool is_hex64(const std::string & s) {
    if (s.size() != 64) return false;
    for (char c : s) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

bool is_template_id(const std::string & s) {
    if (s.empty() || s.size() > 128) return false;
    for (char c : s) {
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_')) return false;
    }
    return true;
}

}  // namespace

namespace {

std::string hex_of(const uint8_t * p, size_t n) {
    static const char * digits = "0123456789abcdef";
    std::string out(2 * n, '0');
    for (size_t i = 0; i < n; ++i) {
        out[2 * i] = digits[p[i] >> 4];
        out[2 * i + 1] = digits[p[i] & 15];
    }
    return out;
}

}  // namespace

Recipe parse(std::string_view text, const ParseContext & ctx) {
    const std::vector<Line> lines = logical_lines(text);
    Recipe r;
    {
        std::set<std::string> ids;
        for (const Literal & l : ctx.literals) {
            if (!ids.insert(l.id).second) throw RecipeError(0, "duplicate literal \"" + l.id + "\"");
            if (l.shape.empty()) throw RecipeError(0, "literal \"" + l.id + "\" has no shape");
            for (int64_t d : l.shape) {
                if (d <= 0) throw RecipeError(0, "literal \"" + l.id + "\" dims must be positive");
            }
            if (!detail::element_count(l.shape)) throw RecipeError(0, "literal \"" + l.id + "\" " + too_many_elements(l.shape));
        }
        r.literals = ctx.literals;
    }
    size_t i = 0;
    auto header = [&](const char * key) -> Cursor {
        if (i >= lines.size()) throw RecipeError(0, std::string("missing header line '") + key + "'");
        Cursor c(lines[i]);
        const std::string w = c.word();
        if (w != key) c.fail(std::string("expected header line '") + key + "', got '" + w + "'");
        c.sp();
        ++i;
        return c;
    };
    {
        Cursor c = header("mlrecipe");
        if (c.uint() != 1) c.fail("unsupported recipe version");
        c.end();
    }
    {
        Cursor c = header("template");
        r.template_id = c.word();
        if (!is_template_id(r.template_id)) c.fail("bad template id '" + r.template_id + "'");
        c.end();
    }
    {
        Cursor c = header("spec_sha256");
        r.spec_sha256 = c.word();
        if (!is_hex64(r.spec_sha256)) c.fail("spec_sha256 must be 64 lowercase hex characters");
        c.end();
    }
    {
        Cursor c = header("weight_bin_size");
        r.weight_bin_size = c.uint();
        c.end();
    }
    {
        Cursor c = header("blobs");
        const uint64_t n = c.uint();
        if (n > 1000000) c.fail("too many blobs");
        r.blob_count = static_cast<uint32_t>(n);
        c.end();
    }

    std::set<std::string> source_names;
    std::set<std::string> meta_keys;
    Checker checker(r);
    bool seen_meta = false;
    bool seen_item = false;
    bool ended = false;
    uint64_t running = milblob::HEADER_BYTES;  // where the next blob's metadata must start
    uint64_t data_end = milblob::HEADER_BYTES;
    uint32_t blobs = 0;
    uint64_t patch_end = 0;  // the end of the previous patch in the spec
    bool any_patch = false;
    std::set<uint64_t> used_sites;

    for (; i < lines.size(); ++i) {
        Cursor c(lines[i]);
        if (ended) c.fail("text after 'end'");
        const std::string kw = c.name();
        if (kw == "end") {
            c.end();
            ended = true;
            continue;
        }
        if (kw == "source") {
            if (seen_meta || seen_item) c.fail("source lines come before meta and item lines");
            c.sp();
            Source s;
            s.line = c.line();
            s.name = c.string();
            if (s.name.empty()) c.fail("empty source name");
            if (!source_names.insert(s.name).second) c.fail("duplicate source \"" + s.name + "\"");
            c.sp();
            std::string types = c.word();
            size_t b = 0;
            while (b <= types.size()) {
                size_t e = types.find(',', b);
                if (e == std::string::npos) e = types.size();
                const std::string t = types.substr(b, e - b);
                auto ty = ggml_type_from_name(t);
                if (!ty) c.fail("unknown ggml type '" + t + "'");
                s.types.push_back(*ty);
                b = e + 1;
            }
            c.sp();
            c.expect('[');
            do {
                const uint64_t d = c.uint();
                if (d == 0) c.fail("source dims must be positive");
                if (d > detail::MAX_ELEMENTS) c.fail("source dim " + std::to_string(d) + " is too large");
                s.shape.push_back(static_cast<int64_t>(d));
            } while (c.eat(','));
            c.expect(']');
            c.end();
            if (s.shape.size() > 4) c.fail("a GGUF tensor has at most 4 dims");
            if (!detail::element_count(s.shape)) c.fail("source " + too_many_elements(s.shape));
            r.sources.push_back(std::move(s));
            continue;
        }
        if (kw == "meta") {
            if (seen_item) c.fail("meta lines come before item lines");
            seen_meta = true;
            c.sp();
            Meta m;
            m.line = c.line();
            m.key = c.string();
            if (m.key.empty()) c.fail("empty meta key");
            if (!meta_keys.insert(m.key).second) c.fail("duplicate meta \"" + m.key + "\"");
            c.sp();
            m.mtype = c.word();
            if (!gguf_type_of_mtype(m.mtype)) c.fail("unknown meta type '" + m.mtype + "'");
            c.end();
            r.meta.push_back(std::move(m));
            continue;
        }
        if (kw == "let") {
            seen_item = true;
            c.sp();
            Item it;
            it.kind = Item::Kind::let;
            it.line = c.line();
            it.name = c.name();
            c.expect('=');
            it.expr = parse_expr(c);
            c.end();
            checker.value(*it.expr);
            it.ulp = it.expr->ulp;
            checker.define(it.name, it.expr, it.line);
            r.items.push_back(std::move(it));
            continue;
        }
        if (kw == "blob") {
            seen_item = true;
            c.sp();
            Item it;
            it.kind = Item::Kind::blob;
            it.line = c.line();
            it.offset = c.uint();
            c.sp();
            const std::string dt = c.word();
            if (dt == "fp16") {
                it.dtype = BlobType::fp16;
            } else if (dt == "fp32") {
                it.dtype = BlobType::fp32;
            } else {
                c.fail("blob dtype must be fp16 or fp32, got '" + dt + "'");
            }
            c.sp();
            it.count = c.uint();
            c.sp();
            it.expr = parse_expr(c);
            c.end();
            checker.blob_top(*it.expr, it.dtype);
            it.ulp = it.expr->ulp;
            const int64_t n = elements(it.expr->shape, it.line);
            if (static_cast<uint64_t>(n) != it.count) {
                c.fail("blob declares " + std::to_string(it.count) + " elements, its expression has " +
                       std::to_string(n) + " " + shape_str(it.expr->shape));
            }
            if (it.offset != running) {
                c.fail("blob offset " + std::to_string(it.offset) + " differs from the writer's running offset " +
                       std::to_string(running));
            }
            // No wrap: offset is a literal of at most 19 digits and count equals a
            // bounded element count, so data_end and its 64-byte alignment fit.
            static_assert(MAX_UINT_LITERAL <= std::numeric_limits<uint64_t>::max() - milblob::METADATA_BYTES -
                                                  4 * detail::MAX_ELEMENTS - milblob::ALIGN,
                          "a blob's end offset must fit in uint64_t");
            data_end = it.offset + milblob::METADATA_BYTES + it.count * blob_type_size(it.dtype);
            running = milblob::align_up(data_end);
            ++blobs;
            r.items.push_back(std::move(it));
            continue;
        }
        if (kw == "patch") {
            seen_item = true;
            c.sp();
            Item it;
            it.kind = Item::Kind::patch;
            it.line = c.line();
            it.offset = c.uint();
            c.sp();
            const std::string dt = c.word();
            if (dt == "fp16") {
                it.dtype = BlobType::fp16;
            } else if (dt == "fp32") {
                it.dtype = BlobType::fp32;
            } else if (dt == "str") {
                c.fail("patch str lines are not in generator version " + std::to_string(GENERATOR_VERSION) +
                       " (a later version adds them)");
            } else {
                c.fail("patch dtype must be fp16 or fp32, got '" + dt + "'");
            }
            c.sp();
            it.count = c.uint();
            c.sp();
            it.expr = parse_expr(c);
            c.end();
            checker.blob_top(*it.expr, it.dtype);
            it.ulp = it.expr->ulp;
            const int64_t n = elements(it.expr->shape, it.line);
            if (static_cast<uint64_t>(n) != it.count) {
                c.fail("patch declares " + std::to_string(it.count) + " elements, its expression has " +
                       std::to_string(n) + " " + shape_str(it.expr->shape));
            }
            // count is a bounded element count, offset a literal of at most 19 digits: no wrap.
            const uint64_t len = it.count * blob_type_size(it.dtype);
            if (any_patch && it.offset < patch_end) {
                c.fail("patch at spec offset " + std::to_string(it.offset) +
                       " overlaps or precedes the previous patch (patch lines come in spec offset order)");
            }
            if (ctx.spec != nullptr) {
                if (it.offset > ctx.spec_size || len > ctx.spec_size - it.offset) {
                    c.fail("patch [" + std::to_string(it.offset) + ", " + std::to_string(it.offset + len) +
                           ") lies outside the " + std::to_string(ctx.spec_size) + "-byte spec");
                }
                const PatchSite * site = nullptr;
                for (const PatchSite & p : ctx.patches) {
                    if (p.spec_offset == it.offset) site = &p;
                }
                if (site == nullptr) c.fail("patch at spec offset " + std::to_string(it.offset) + " has no template.json patches entry");
                if (site->len != len) {
                    c.fail("patch at spec offset " + std::to_string(it.offset) + " writes " + std::to_string(len) +
                           " bytes, its template.json entry has len " + std::to_string(site->len));
                }
                if (hex_of(ctx.spec + it.offset, len) != site->sentinel_hex) {
                    c.fail("the spec bytes at offset " + std::to_string(it.offset) + " are not the patch's sentinel");
                }
                used_sites.insert(it.offset);
            }
            any_patch = true;
            patch_end = it.offset + len;
            r.items.push_back(std::move(it));
            continue;
        }
        c.fail("unknown line '" + kw + "'");
    }
    if (!ended) throw RecipeError(0, "missing 'end'");
    if (blobs != r.blob_count) {
        throw RecipeError(0, "header declares " + std::to_string(r.blob_count) + " blobs, the recipe has " +
                                 std::to_string(blobs));
    }
    if (!milblob::valid_total(data_end, r.weight_bin_size)) {
        throw RecipeError(0, "weight_bin_size " + std::to_string(r.weight_bin_size) + " differs from the writer's " +
                                 std::to_string(milblob::align_up(data_end)));
    }
    if (ctx.spec != nullptr) {
        for (const PatchSite & p : ctx.patches) {
            if (!used_sites.count(p.spec_offset)) {
                throw RecipeError(0, "template.json patch at spec offset " + std::to_string(p.spec_offset) +
                                         " has no patch line");
            }
        }
    }
    return r;
}

// ---- Templates -----------------------------------------------------------------

const std::vector<std::string> & Template::required_files() {
    static const std::vector<std::string> names = {"Manifest.json", "model.mlmodel", "recipe.txt", "template.json"};
    return names;
}

namespace {

std::shared_ptr<std::vector<uint8_t>> read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return nullptr;
    auto v = std::make_shared<std::vector<uint8_t>>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return v;
}

}  // namespace

std::shared_ptr<const Template> Template::from_dir(const std::string & dir) {
    std::shared_ptr<Template> t(new Template());
    std::vector<std::string> names = required_files();
    names.push_back("literals.bin");
    for (const std::string & n : names) {
        auto bytes = read_file(dir + "/" + n);
        if (!bytes) {
            if (n == "literals.bin") continue;
            throw std::runtime_error("template " + dir + ": missing " + n);
        }
        TemplateFile f;
        f.name = n;
        f.data = bytes->data();
        f.size = bytes->size();
        t->owned_.push_back(bytes);
        t->files_.push_back(f);
    }
    t->init(true);
    return t;
}

std::shared_ptr<const Template> Template::from_files(std::vector<TemplateFile> files) {
    std::shared_ptr<Template> t(new Template());
    t->files_ = std::move(files);
    t->init(true);
    return t;
}

std::shared_ptr<const Template> Template::from_embedded(std::vector<TemplateFile> files) {
    std::shared_ptr<Template> t(new Template());
    t->files_ = std::move(files);
    t->init(false);
    return t;
}

const TemplateFile * Template::file(std::string_view name) const {
    for (const TemplateFile & f : files_) {
        if (f.name == name) return &f;
    }
    return nullptr;
}

std::string Template::sha256_hex() const { return to_hex(sha256_); }

void Template::init(bool hash_files) {
    if (hash_files) {
        for (TemplateFile & f : files_) f.sha256 = detail::sha256(f.data, f.size);
    }
    for (const std::string & n : required_files()) {
        if (!file(n)) throw std::runtime_error("template: missing " + n);
    }
    std::sort(files_.begin(), files_.end(), [](const TemplateFile & a, const TemplateFile & b) { return a.name < b.name; });
    for (size_t i = 1; i < files_.size(); ++i) {
        if (files_[i].name == files_[i - 1].name) throw std::runtime_error("template: duplicate file " + files_[i].name);
    }
    // template_sha256 = H("qvac-coreml-template\0" || for f in sorted(files): str(name) || sha256)
    Sha256 h;
    static const char tag[] = "qvac-coreml-template";
    h.update(tag, sizeof(tag));
    for (const TemplateFile & f : files_) {
        h.str(f.name);
        h.update(f.sha256.data(), f.sha256.size());
    }
    sha256_ = h.finish();

    const TemplateFile * jf = file("template.json");
    const nlohmann::json j =
        nlohmann::json::parse(std::string(reinterpret_cast<const char *>(jf->data), jf->size), nullptr, false);
    if (!j.is_object()) throw std::runtime_error("template.json: not a JSON object");
    if (j.value("schema", 0) != 1) throw std::runtime_error("template.json: schema must be 1");
    auto str = [&](const char * k) {
        auto it = j.find(k);
        return it != j.end() && it->is_string() ? it->get<std::string>() : std::string();
    };
    info_.id = str("id");
    info_.family = str("family");
    info_.stage = str("stage");
    info_.export_kind = str("export_kind");
    info_.default_compute_units = str("default_compute_units");
    info_.weight_bin_size = j.value("weight_bin_size", uint64_t{0});
    info_.generation_peak_bytes = j.value("generation_peak_bytes", uint64_t{0});
    info_.stateful = j.value("stateful", false);
    info_.shareable = j.value("shareable", true);
    if (auto it = j.find("functions"); it != j.end() && it->is_array()) {
        for (const auto & f : *it) {
            if (f.is_object() && f.contains("name") && f["name"].is_string()) info_.functions.push_back(f["name"]);
        }
    }
    if (info_.functions.empty()) info_.functions.push_back("main");
    if (auto it = j.find("match"); it != j.end() && it->is_object()) {
        if (auto a = it->find("architecture"); a != it->end() && a->is_string()) info_.match_architecture = *a;
        if (auto m = it->find("metadata"); m != it->end()) {
            if (!m->is_object()) throw std::runtime_error("template.json: match.metadata must be an object");
            info_.match_metadata_json = m->dump();
        }
    }
    if (auto it = j.find("provided"); it != j.end() && !(it->is_array() && it->empty())) {
        throw std::runtime_error("template.json: provided is not supported by generator version " +
                                 std::to_string(GENERATOR_VERSION));
    }
    const TemplateFile * spec = file("model.mlmodel");
    ParseContext ctx;
    ctx.spec = spec->data;
    ctx.spec_size = spec->size;
    if (auto it = j.find("patches"); it != j.end()) {
        if (!it->is_array()) throw std::runtime_error("template.json: patches must be an array");
        for (const auto & p : *it) {
            if (!p.is_object() || !p.contains("spec_offset") || !p["spec_offset"].is_number_unsigned() ||
                !p.contains("len") || !p["len"].is_number_unsigned() || !p.contains("sentinel_hex") ||
                !p["sentinel_hex"].is_string()) {
                throw std::runtime_error("template.json: a patches entry needs spec_offset, len and sentinel_hex");
            }
            PatchSite site;
            site.spec_offset = p["spec_offset"].get<uint64_t>();
            site.len = p["len"].get<uint64_t>();
            site.sentinel_hex = p["sentinel_hex"].get<std::string>();
            if (site.len == 0 || site.sentinel_hex.size() != 2 * site.len) {
                throw std::runtime_error("template.json: patch at spec offset " + std::to_string(site.spec_offset) +
                                         " needs a sentinel_hex of 2 x len characters");
            }
            ctx.patches.push_back(std::move(site));
        }
    }
    if (auto it = j.find("literals"); it != j.end()) {
        if (!it->is_array()) throw std::runtime_error("template.json: literals must be an array");
        const TemplateFile * lf = file("literals.bin");
        for (const auto & l : *it) {
            if (!l.is_object() || !l.contains("id") || !l["id"].is_string() || !l.contains("dtype") ||
                !l["dtype"].is_string() || !l.contains("shape") || !l["shape"].is_array() || !l.contains("offset") ||
                !l["offset"].is_number_unsigned()) {
                throw std::runtime_error("template.json: a literals entry needs id, dtype, shape and offset");
            }
            Literal lit;
            lit.id = l["id"].get<std::string>();
            const std::string dt = l["dtype"].get<std::string>();
            if (dt == "fp16") {
                lit.dtype = BlobType::fp16;
            } else if (dt == "fp32") {
                lit.dtype = BlobType::fp32;
            } else {
                throw std::runtime_error("template.json: literal " + lit.id + " dtype must be fp16 or fp32");
            }
            for (const auto & d : l["shape"]) {
                if (!d.is_number_unsigned() || d.get<uint64_t>() == 0 || d.get<uint64_t>() > detail::MAX_ELEMENTS) {
                    throw std::runtime_error("template.json: literal " + lit.id + " dims must be positive integers");
                }
                lit.shape.push_back(static_cast<int64_t>(d.get<uint64_t>()));
            }
            lit.offset = l["offset"].get<uint64_t>();
            const std::optional<uint64_t> n = detail::element_count(lit.shape);
            if (lit.shape.empty() || !n) throw std::runtime_error("template.json: literal " + lit.id + " has a bad shape");
            if (lf == nullptr) throw std::runtime_error("template.json: literal " + lit.id + " needs literals.bin");
            const uint64_t bytes = *n * blob_type_size(lit.dtype);
            if (lit.offset > lf->size || bytes > lf->size - lit.offset) {
                throw std::runtime_error("template.json: literal " + lit.id + " lies outside literals.bin");
            }
            lit.data = lf->data + lit.offset;
            ctx.literals.push_back(std::move(lit));
        }
    }
    const TemplateFile * rf = file("recipe.txt");
    recipe_ = parse(std::string_view(reinterpret_cast<const char *>(rf->data), rf->size), ctx);
    if (info_.id != recipe_.template_id) {
        throw std::runtime_error("template.json id '" + info_.id + "' differs from recipe template '" +
                                 recipe_.template_id + "'");
    }
    if (info_.weight_bin_size != recipe_.weight_bin_size) {
        throw std::runtime_error("template.json weight_bin_size differs from the recipe's");
    }
    if (to_hex(spec->sha256) != recipe_.spec_sha256) {
        throw std::runtime_error("model.mlmodel sha256 differs from the recipe's spec_sha256");
    }
}

// ---- Matching ----------------------------------------------------------------

MatchResult match(const Template & t, const TensorSource & src) {
    MatchResult m;
    const TemplateInfo & info = t.info();
    if (!info.match_architecture.empty()) {
        const std::string a = src.meta_json("general.architecture");
        const nlohmann::json v = nlohmann::json::parse(a.empty() ? "null" : a, nullptr, false);
        if (!v.is_string() || v.get<std::string>() != info.match_architecture) {
            m.reason = "general.architecture is " + (a.empty() ? std::string("absent") : a) + ", the template needs \"" +
                       info.match_architecture + "\"";
            return m;
        }
    }
    if (!info.match_metadata_json.empty()) {
        const nlohmann::json want = nlohmann::json::parse(info.match_metadata_json);
        for (auto it = want.begin(); it != want.end(); ++it) {
            const std::string got = src.meta_json(it.key());
            if (got.empty()) {
                m.reason = "metadata " + it.key() + " is absent";
                return m;
            }
            const nlohmann::json g = nlohmann::json::parse(got, nullptr, false);
            if (g != it.value()) {
                m.reason = "metadata " + it.key() + " is " + got + ", the template needs " + it.value().dump();
                return m;
            }
        }
    }
    for (const Source & s : t.recipe().sources) {
        const TensorInfo * ti = src.tensor(s.name);
        if (!ti) {
            m.reason = "tensor " + s.name + " is absent";
            return m;
        }
        if (std::find(s.types.begin(), s.types.end(), ti->type) == s.types.end()) {
            m.reason = "tensor " + s.name + " has type " + ggml_type_name_of(ti->type) + ", not an allowed type";
            return m;
        }
        std::array<int64_t, 4> want{{1, 1, 1, 1}};
        for (size_t d = 0; d < s.shape.size(); ++d) want[d] = s.shape[s.shape.size() - 1 - d];
        if (want != ti->ne) {
            m.reason = "tensor " + s.name + " has ne [" + std::to_string(ti->ne[0]) + "," + std::to_string(ti->ne[1]) +
                       "," + std::to_string(ti->ne[2]) + "," + std::to_string(ti->ne[3]) + "], the recipe needs " +
                       shape_str(s.shape) + " (numpy order)";
            return m;
        }
    }
    for (const Meta & me : t.recipe().meta) {
        uint32_t ty = 0, elem = 0, want_elem = 0;
        std::vector<uint8_t> bytes;
        if (!src.meta_canonical(me.key, ty, bytes, &elem)) {
            m.reason = "metadata " + me.key + " is absent";
            return m;
        }
        const uint32_t want = *gguf_type_of_mtype(me.mtype, &want_elem);
        if (ty != want || (want == GGUF_ARRAY && elem != want_elem)) {
            m.reason = "metadata " + me.key + " is not " + me.mtype;
            return m;
        }
    }
    m.ok = true;
    return m;
}

}  // namespace recipe
}  // namespace QVAC_COREML_NS
