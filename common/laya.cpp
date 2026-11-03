#include "laya.h"

#include "common.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// The sequence layout, option budget, calibration and answer decoding follow laya/common.py and
// laya/agent.py. The model graph is src/models/laya.cpp.

using json = nlohmann::ordered_json;

static const char * QTYPE_NAMES[3] = { "choice", "score", "noul" };

//
// Python compatible formatting
//

// repr(float)
static std::string py_float(double v) {
    if (std::isnan(v)) {
        return "NaN";
    }
    if (std::isinf(v)) {
        return v > 0 ? "Infinity" : "-Infinity";
    }

    // the shortest digits that round-trip (17 significant digits always do), independent of the
    // global locale; floating-point std::to_chars is not available on every deployment target.
    // A candidate that overflows (2e+308 for DBL_MAX) fails the parse instead of comparing equal.
    std::string sci;
    for (int prec = 0; prec <= 16; ++prec) {
        std::ostringstream os;
        os.imbue(std::locale::classic());
        os << std::scientific << std::setprecision(prec) << v;
        sci = os.str();

        std::istringstream is(sci);
        is.imbue(std::locale::classic());
        double back = 0.0;
        if ((is >> back) && back == v) {
            break;
        }
    }

    const size_t epos = sci.find('e');
    const int    exp  = std::stoi(sci.substr(epos + 1));

    std::string mant = sci.substr(0, epos);
    const bool neg = mant[0] == '-';
    if (neg) {
        mant = mant.substr(1);
    }
    std::string digits;
    for (char c : mant) {
        if (c != '.') {
            digits += c;
        }
    }

    std::string out;
    if (exp < -4 || exp >= 16) {
        out = digits.substr(0, 1);
        if (digits.size() > 1) {
            out += "." + digits.substr(1);
        }
        char ebuf[16];
        snprintf(ebuf, sizeof(ebuf), "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
        out += ebuf;
    } else if (exp < 0) {
        out = "0." + std::string(-exp - 1, '0') + digits;
    } else {
        const size_t n_int = exp + 1;
        if (digits.size() <= n_int) {
            out = digits + std::string(n_int - digits.size(), '0') + ".0";
        } else {
            out = digits.substr(0, n_int) + "." + digits.substr(n_int);
        }
    }

    return neg ? "-" + out : out;
}

// json.dumps(s, ensure_ascii=False) for a string
static std::string py_json_str(const std::string & s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    return out + "\"";
}

// json.dumps(v, ensure_ascii=False) with the default separators (", ", ": ")
static std::string py_dumps(const json & v) {
    switch (v.type()) {
        case json::value_t::null:            return "null";
        case json::value_t::boolean:         return v.get<bool>() ? "true" : "false";
        case json::value_t::number_integer:  return std::to_string(v.get<int64_t>());
        case json::value_t::number_unsigned: return std::to_string(v.get<uint64_t>());
        case json::value_t::number_float:    return py_float(v.get<double>());
        case json::value_t::string:          return py_json_str(v.get<std::string>());
        case json::value_t::array:
            {
                std::string out = "[";
                for (size_t i = 0; i < v.size(); ++i) {
                    out += (i ? ", " : "") + py_dumps(v[i]);
                }
                return out + "]";
            }
        case json::value_t::object:
            {
                std::string out = "{";
                bool first = true;
                for (const auto & [key, val] : v.items()) {
                    out += (first ? "" : ", ") + py_json_str(key) + ": " + py_dumps(val);
                    first = false;
                }
                return out + "}";
            }
        default:
            throw std::runtime_error("unsupported JSON value");
    }
}

// str(v) of a scalar label
static std::string py_str(const json & v) {
    switch (v.type()) {
        case json::value_t::string:       return v.get<std::string>();
        case json::value_t::boolean:      return v.get<bool>() ? "True" : "False";
        case json::value_t::null:         return "None";
        case json::value_t::number_float:
            {
                const double d = v.get<double>();
                if (std::isnan(d)) return "nan";
                if (std::isinf(d)) return d > 0 ? "inf" : "-inf";
                return py_float(d);
            }
        default:                          return py_dumps(v);
    }
}

// the key json.dumps writes for a dict key
static std::string py_json_key(const json & v) {
    switch (v.type()) {
        case json::value_t::string:       return v.get<std::string>();
        case json::value_t::null:         return "null";
        case json::value_t::boolean:      return v.get<bool>() ? "true" : "false";
        default:                          return py_dumps(v);
    }
}

// labels that Python treats as one dict key (1 == 1.0 == True) collide
static std::string py_key_identity(const json & v) {
    if (v.is_boolean()) {
        return v.get<bool>() ? "n:1" : "n:0";
    }
    if (v.is_number_integer() || v.is_number_unsigned()) {
        return "n:" + v.dump();
    }
    if (v.is_number_float()) {
        const double d = v.get<double>();
        if (std::isfinite(d) && d == std::floor(d) && std::fabs(d) < 9.2e18) {
            return "n:" + std::to_string((int64_t) d);
        }
        return "n:" + py_float(d);
    }
    return "s:" + v.get<std::string>();
}

static std::string replace_all(std::string s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return s;
    }
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

//
// tokenizer
//

struct laya_tokenizer {
    const llama_vocab * vocab;

    llama_token cls;
    llama_token sep;
    llama_token mask;

    std::string mask_text;

    // mmBERT: byte-fallback BPE behind a Metaspace pre-tokenizer (prepend "always", split on spaces)
    bool metaspace;

    // added tokens by first byte, longest first
    std::vector<std::vector<std::pair<std::string, llama_token>>> added;

    explicit laya_tokenizer(const llama_vocab * vocab) : vocab(vocab) {
        cls       = llama_vocab_bos(vocab);
        sep       = llama_vocab_sep(vocab);
        mask      = llama_vocab_mask(vocab);
        if (cls == LLAMA_TOKEN_NULL || sep == LLAMA_TOKEN_NULL || mask == LLAMA_TOKEN_NULL) {
            throw std::invalid_argument("the model vocabulary has no [CLS], [SEP] or [MASK] token");
        }
        mask_text = llama_vocab_get_text(vocab, mask);
        metaspace = llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_SPM;

        if (metaspace) {
            added.resize(256);
            for (llama_token id = 0; id < llama_vocab_n_tokens(vocab); ++id) {
                if (!(llama_vocab_get_attr(vocab, id) & (LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_CONTROL))) {
                    continue;
                }
                const std::string text = llama_vocab_get_text(vocab, id);
                if (!text.empty()) {
                    added[(unsigned char) text[0]].emplace_back(text, id);
                }
            }
            for (auto & list : added) {
                std::stable_sort(list.begin(), list.end(), [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });
            }
        }
    }

    void tokenize_raw(const std::string & text, bool parse_special, std::vector<llama_token> & out) const {
        const auto tokens = common_tokenize(vocab, text, /*add_special*/ false, parse_special);
        out.insert(out.end(), tokens.begin(), tokens.end());
    }

    // Metaspace: prepend a space unless there is one, then start a new piece at every space
    void tokenize_metaspace(const std::string & text, std::vector<llama_token> & out) const {
        if (text.empty()) {
            return;
        }
        // spaces become U+2581 in the normalizer, so a U+2581 in the text acts as a space too
        const std::string t = replace_all(text, "\xe2\x96\x81", " ");
        const std::string s = t[0] == ' ' ? t : " " + t;
        size_t start = 0;
        for (size_t i = 1; i <= s.size(); ++i) {
            if (i == s.size() || s[i] == ' ') {
                tokenize_raw(s.substr(start, i - start), false, out);
                start = i;
            }
        }
    }

    // tok(text, add_special_tokens=False)["input_ids"]
    std::vector<llama_token> tokenize(const std::string & text, size_t max_tokens = SIZE_MAX) const {
        std::vector<llama_token> out;
        if (!metaspace) {
            // special tokens in the text are matched, as the HF tokenizer does
            tokenize_raw(text, true, out);
        } else {
            // the HF tokenizer splits out the added tokens (leftmost-longest) before the
            // pre-tokenizer, so every segment between them gets its own space prefix
            std::string segment;
            for (size_t i = 0; i < text.size(); ) {
                const std::pair<std::string, llama_token> * match = nullptr;
                for (const auto & cand : added[(unsigned char) text[i]]) {
                    if (text.compare(i, cand.first.size(), cand.first) == 0) {
                        match = &cand;
                        break;
                    }
                }
                if (match) {
                    tokenize_metaspace(segment, out);
                    segment.clear();
                    out.push_back(match->second);
                    i += match->first.size();
                } else {
                    segment += text[i++];
                }
            }
            tokenize_metaspace(segment, out);
        }
        if (out.size() > max_tokens) {
            out.resize(max_tokens);
        }
        return out;
    }
};

//
// questions
//

struct laya_question {
    std::string id;
    int         type; // index into QTYPE_NAMES
    std::string ins;

    // choice: (label, description) per option; a label keeps its JSON type and is the answer key
    std::vector<std::pair<json, json>> choices;

    // score: the level descriptions
    json levels = json::array();

    // noul: the descriptions keyed "false" / "true", and the labels shown for them
    json        noul_crit  = json::object();
    std::string noul_false = "false";
    std::string noul_true  = "true";

    // option_order[s] is the option shown in slot s; empty for the canonical order
    std::vector<int> option_order;

    size_t n_options() const {
        return type == 0 ? choices.size() : type == 1 ? levels.size() : 2;
    }
};

static std::string render_criterion(const json & v) {
    return v.is_string() ? v.get<std::string>() : py_dumps(v);
}

static bool is_empty_criterion(const json & v) {
    return v.is_null() || (v.is_string() && v.get<std::string>().empty());
}

[[noreturn]] static void question_error(const std::string & id, const std::string & msg) {
    throw std::invalid_argument("question '" + id + "': " + msg);
}

static bool is_blank(const std::string & s) {
    return s.find_first_not_of(" \t\n\r\f\v") == std::string::npos;
}

// Agent._check_question + Agent._to_internal
static laya_question parse_question(const std::string & id, const json & q) {
    if (is_blank(id)) {
        throw std::invalid_argument("question id must be a non-empty string, got " + py_json_str(id));
    }
    if (!q.is_object()) {
        question_error(id, "definition must be an object");
    }

    laya_question res;
    res.id = id;

    const json type = q.value("type", json());
    res.type = -1;
    for (int t = 0; t < 3 && type.is_string(); ++t) {
        if (type.get<std::string>() == QTYPE_NAMES[t]) {
            res.type = t;
        }
    }
    if (res.type < 0) {
        question_error(id, "unknown type " + py_dumps(type) + "; use one of choice, noul, score");
    }

    if (!q.contains("instructions")) {
        question_error(id, "no 'instructions'; add the text the model should answer");
    }
    const json & ins = q["instructions"];
    if (ins.is_null()) {
        question_error(id, "'instructions' must not be null; add the text the model should answer");
    }
    if ((ins.is_string() && is_blank(ins.get<std::string>())) || ((ins.is_array() || ins.is_object()) && ins.empty())) {
        question_error(id, "'instructions' must not be empty; add the text the model should answer");
    }
    res.ins = ins.is_string() ? ins.get<std::string>() : py_dumps(ins);

    const json crit = q.value("criteria", json());
    if (q.contains("labels") && res.type != 2) {
        question_error(id, "'labels' is only supported for noul questions");
    }

    switch (res.type) {
        case 0:
            {
                if (crit.is_array()) {
                    std::set<std::string> seen;
                    for (const auto & label : crit) {
                        if (label.is_null() || label.is_array() || label.is_object()) {
                            question_error(id, "a choice label must be a string, number or bool");
                        }
                        if (!seen.insert(py_key_identity(label)).second) {
                            question_error(id, "choice label " + py_dumps(label) + " repeats another label (1, 1.0 and true are one key)");
                        }
                        res.choices.emplace_back(label, nullptr);
                    }
                } else if (crit.is_object()) {
                    for (const auto & [key, val] : crit.items()) {
                        res.choices.emplace_back(key, val);
                    }
                } else {
                    question_error(id, "a choice question takes 'criteria' as an object of label -> description, or a list of labels");
                }
                if (res.choices.empty()) {
                    question_error(id, "a choice question needs at least one criterion");
                }
            } break;
        case 1:
            {
                if (!crit.is_array() || crit.empty()) {
                    question_error(id, "a score question takes 'criteria' as a non-empty list of level descriptions");
                }
                for (const auto & c : crit) {
                    if (c.is_null()) {
                        question_error(id, "score levels cannot be null");
                    }
                }
                res.levels = crit;
            } break;
        case 2:
            {
                if (crit.is_object()) {
                    for (const auto & [key, val] : crit.items()) {
                        std::string k = key;
                        std::transform(k.begin(), k.end(), k.begin(), [](unsigned char c) { return (char) std::tolower(c); });
                        if (k != "true" && k != "false") {
                            question_error(id, "a noul question takes 'criteria' keyed only 'true'/'false'");
                        }
                        res.noul_crit[k] = val;
                    }
                } else if (!crit.is_null()) {
                    question_error(id, "a noul question takes 'criteria' as an object with optional 'true'/'false' descriptions");
                }
                if (q.contains("labels")) {
                    const json & labels = q["labels"];
                    if (!labels.is_object() || labels.size() != 2 || !labels.contains("false") || !labels.contains("true") ||
                        !labels["false"].is_string() || !labels["true"].is_string()) {
                        question_error(id, "noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
                    }
                    auto strip = [](std::string s) {
                        const auto b = s.find_first_not_of(" \t\n\r\f\v");
                        const auto e = s.find_last_not_of(" \t\n\r\f\v");
                        return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
                    };
                    res.noul_false = strip(labels["false"].get<std::string>());
                    res.noul_true  = strip(labels["true"].get<std::string>());
                    if (res.noul_false.empty() || res.noul_true.empty() || res.noul_false == res.noul_true) {
                        question_error(id, "noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
                    }
                }
            } break;
    }

    if (q.contains("option_order")) {
        // slot s shows option order[s]: anything but a permutation would drop or repeat an option
        const json & order = q["option_order"];
        const size_t n = res.n_options();
        bool ok = order.is_array() && order.size() == n;
        std::vector<int> seen(n, 0);
        for (size_t s = 0; ok && s < n; ++s) {
            ok = order[s].is_number_integer() && order[s].get<int64_t>() >= 0 && order[s].get<int64_t>() < (int64_t) n && !seen[order[s].get<int64_t>()]++;
            if (ok) {
                res.option_order.push_back(order[s].get<int>());
            }
        }
        if (!ok) {
            question_error(id, "'option_order' must be a permutation of range(" + std::to_string(n) + ") -- one slot per option, each option once -- got " + py_dumps(order));
        }
    }

    return res;
}

// render_options
static std::vector<std::string> render_options(const laya_question & q) {
    std::vector<std::string> opts;
    switch (q.type) {
        case 0:
            {
                for (const auto & [label, desc] : q.choices) {
                    opts.push_back(is_empty_criterion(desc) ? py_str(label) : py_str(label) + ": " + render_criterion(desc));
                }
            } break;
        case 1:
            {
                for (size_t i = 0; i < q.levels.size(); ++i) {
                    opts.push_back("level " + std::to_string(i) + ": " + render_criterion(q.levels[i]));
                }
            } break;
        case 2:
            {
                const json f = q.noul_crit.value("false", json());
                const json t = q.noul_crit.value("true",  json());
                opts.push_back(q.noul_false + ": " + (is_empty_criterion(f) ? "no, the statement does not hold" : render_criterion(f)));
                opts.push_back(q.noul_true  + ": " + (is_empty_criterion(t) ? "yes, the statement holds"        : render_criterion(t)));
            } break;
    }
    return opts;
}

//
// sequences
//

// the part of a sequence that depends only on the question: [CLS] head [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]
struct laya_head {
    std::vector<llama_token> ids;
    std::vector<int32_t>     markers;

    size_t n_options          = 0;
    size_t n_options_distinct = 0;
    int    tokens_per_option  = -1; // -1: not capped
};

struct laya_item {
    std::vector<llama_token> ids;
    std::vector<int32_t>     markers;

    size_t state_tokens_dropped = 0;
};

// build_sequence, up to the state
static laya_head build_head(const laya_tokenizer & tok, const laya_question & q, int head_max_len) {
    const auto opts = render_options(q);

    std::vector<llama_token> head_ids = tok.tokenize(std::string(QTYPE_NAMES[q.type]) + " question: " + replace_all(q.ins, tok.mask_text, " "));

    // slot s shows option option_order[s]
    std::vector<int> order = q.option_order;
    if (order.empty()) {
        for (size_t i = 0; i < opts.size(); ++i) {
            order.push_back(i);
        }
    }

    std::vector<std::vector<llama_token>> opt_ids;
    int n_opt_tokens = 0;
    for (int i : order) {
        std::vector<llama_token> o = { tok.mask };
        const auto opt_tokens = tok.tokenize(" " + replace_all(opts[i], tok.mask_text, " "), 48);
        o.insert(o.end(), opt_tokens.begin(), opt_tokens.end());
        n_opt_tokens += o.size();
        opt_ids.push_back(std::move(o));
    }

    laya_head head;

    int opt_budget = head_max_len - n_opt_tokens;
    if (opt_budget < 16) {
        const int per = std::max(4, (head_max_len - 16) / std::max<int>(1, opt_ids.size()));
        head.tokens_per_option = per;
        n_opt_tokens = 0;
        for (auto & o : opt_ids) {
            if ((int) o.size() > per) {
                o.resize(per);
            }
            n_opt_tokens += o.size();
        }
        opt_budget = head_max_len - n_opt_tokens;
    }
    const int n_head = std::max(8, opt_budget);
    if ((int) head_ids.size() > n_head) {
        head_ids.resize(n_head);
    }

    auto & ids = head.ids;
    ids.push_back(tok.cls);
    ids.insert(ids.end(), head_ids.begin(), head_ids.end());
    ids.push_back(tok.sep);
    for (const auto & o : opt_ids) {
        head.markers.push_back(ids.size());
        ids.insert(ids.end(), o.begin(), o.end());
    }
    ids.push_back(tok.sep);

    head.n_options          = opt_ids.size();
    head.n_options_distinct = std::set<std::vector<llama_token>>(opt_ids.begin(), opt_ids.end()).size();

    return head;
}

// build_sequence: the question head, then as much of the state as fits, then [SEP], cut at max_len
static laya_item build_sequence(const laya_head & head, const std::vector<llama_token> & state_ids, llama_token sep,
        int max_len, bool truncate_left) {
    laya_item item;
    item.ids     = head.ids;
    item.markers = head.markers;

    auto & ids = item.ids;

    const size_t room = std::max<int>(0, max_len - (int) ids.size() - 1);
    const size_t n_st = std::min(room, state_ids.size());
    item.state_tokens_dropped = state_ids.size() - n_st;
    if (truncate_left) {
        ids.insert(ids.end(), state_ids.end() - n_st, state_ids.end());
    } else {
        ids.insert(ids.end(), state_ids.begin(), state_ids.begin() + n_st);
    }
    ids.push_back(sep);

    if ((int) ids.size() > max_len) {
        ids.resize(max_len);
    }
    item.markers.erase(std::remove_if(item.markers.begin(), item.markers.end(), [&](int32_t m) { return m >= max_len; }), item.markers.end());

    return item;
}

//
// calibration and decoding
//

struct laya_config {
    int max_len      = 512;
    int head_max_len = 192;

    float temperature[3] = { 1.0f, 1.0f, 1.0f };
    std::map<std::string, float> temperature_by_options;
};

// clamp_temperature
static float clamp_temperature(const json & t) {
    if (!t.is_number()) {
        return 1.0f;
    }
    const double v = t.get<double>();
    if (!std::isfinite(v)) {
        return 1.0f;
    }
    return std::min(5.0, std::max(0.5, v));
}

static laya_config load_config(const llama_model * model) {
    laya_config cfg;

    const char * key = "laya.decision.config";
    const int32_t n = llama_model_meta_val_str(model, key, nullptr, 0);
    if (n < 0) {
        throw std::invalid_argument("the model has no laya.decision.config; is it a Laya checkpoint?");
    }
    std::string buf(n + 1, '\0');
    llama_model_meta_val_str(model, key, buf.data(), buf.size());
    buf.resize(n);

    const json j = json::parse(buf);

    cfg.max_len      = j.value("max_len", 512);
    cfg.head_max_len = j.value("head_max_len", 192);
    if (j.contains("temperature")) {
        if (!j["temperature"].is_array() || j["temperature"].size() != 3) {
            throw std::invalid_argument("laya.decision.config: temperature must be a list of 3 floats");
        }
        for (int i = 0; i < 3; ++i) {
            cfg.temperature[i] = clamp_temperature(j["temperature"][i]);
        }
    }
    const json by_options = j.value("temperature_by_options", json::object());
    for (const auto & [bucket, t] : by_options.items()) {
        cfg.temperature_by_options[bucket] = clamp_temperature(t);
    }

    return cfg;
}

// temp_bucket
static std::string temp_bucket(int qtype, size_t k) {
    const char * size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
    return std::string(QTYPE_NAMES[qtype]) + ":" + size;
}

static double round4(double v) {
    return std::round(v * 1e4) / 1e4;
}

// confidence_from_probs
static double confidence_from_probs(const std::vector<double> & p) {
    const size_t k = p.size();
    if (k < 2) {
        return 1.0;
    }
    double ent = 0.0;
    for (double v : p) {
        ent -= v * std::log(std::min(1.0, std::max(1e-12, v)));
    }
    return std::min(1.0, std::max(0.0, 1.0 - ent / std::log((double) k)));
}

// Agent._decode_answers for one question
static json decode_answer(const laya_config & cfg, const laya_question & q, const float * out, int n_act, size_t k) {
    const auto it = cfg.temperature_by_options.find(temp_bucket(q.type, k));
    const double t_scale = it != cfg.temperature_by_options.end() ? it->second : cfg.temperature[q.type];

    const float * logits = out + n_act;

    std::vector<double> p(k);
    double zmax = -INFINITY;
    for (size_t i = 0; i < k; ++i) {
        p[i] = logits[i] / t_scale;
        zmax = std::max(zmax, p[i]);
    }
    double sum = 0.0;
    for (size_t i = 0; i < k; ++i) {
        p[i] = std::exp(p[i] - zmax);
        sum += p[i];
    }
    for (size_t i = 0; i < k; ++i) {
        p[i] /= sum;
    }

    // the row comes back in slot order, everything below indexes by option (unpermute_probs)
    if (q.option_order.size() == k) {
        std::vector<double> canonical(k);
        for (size_t s = 0; s < k; ++s) {
            canonical[q.option_order[s]] = p[s];
        }
        p = canonical;
    }

    size_t argmax = 0;
    for (size_t i = 0; i < k; ++i) {
        if (p[i] > p[argmax]) {
            argmax = i;
        }
    }

    // softmax over the act logits
    double act_max = -INFINITY;
    for (int i = 0; i < n_act; ++i) {
        act_max = std::max<double>(act_max, out[i]);
    }
    double act_sum = 0.0;
    for (int i = 0; i < n_act; ++i) {
        act_sum += std::exp(out[i] - act_max);
    }
    const double act_probability = std::exp(out[0] - act_max) / act_sum;

    json ans;
    ans["type"] = QTYPE_NAMES[q.type];

    const double answer_confidence = k < 1 ? 1.0 : std::min(1.0, std::max(0.0, p[argmax]));

    switch (q.type) {
        case 0:
            {
                ans["choice"] = q.choices[argmax].first;
                // labels that are equal as JSON keys (1 and "1") share one entry, as json.dumps + loads would leave them
                json probs = json::object();
                for (size_t i = 0; i < k; ++i) {
                    probs[py_json_key(q.choices[i].first)] = round4(p[i]);
                }
                ans["probabilities"] = probs;
                ans["confidence"] = round4(confidence_from_probs(p));
            } break;
        case 1:
            {
                double score = 0.0;
                for (size_t i = 0; i < k; ++i) {
                    score += i * p[i];
                }
                ans["score"] = round4(score);
                json legend = json::object();
                json probs  = json::object();
                for (size_t i = 0; i < q.levels.size(); ++i) {
                    legend[std::to_string(i)] = render_criterion(q.levels[i]);
                }
                for (size_t i = 0; i < k; ++i) {
                    probs[std::to_string(i)] = round4(p[i]);
                }
                ans["legend"] = legend;
                ans["probabilities"] = probs;
                ans["confidence"] = round4(confidence_from_probs(p));
            } break;
        case 2:
            {
                ans["noul"] = round4(p[1]);
                ans["confidence"] = round4(std::max(p[1], 1.0 - p[1]));
            } break;
    }

    ans["answer_confidence"] = round4(answer_confidence);
    ans["action"] = { { "act_probability", round4(act_probability) } };

    return ans;
}

//
// API
//

void common_laya_context_params(llama_context_params & cparams, uint32_t n_batch) {
    cparams.embeddings   = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_RANK;
    cparams.n_ctx        = n_batch; // no KV cache: the context only has to hold one batch
    cparams.n_batch      = n_batch;
    cparams.n_ubatch     = n_batch; // the decision head needs every sequence in one ubatch
    cparams.n_seq_max    = std::min<uint32_t>(llama_max_parallel_sequences(), n_batch); // at most one sequence per token
    cparams.kv_unified   = true;
}

struct common_laya {
    llama_context * ctx;

    laya_tokenizer tok;
    laya_config    cfg;

    int    n_act;
    size_t n_max_options;
};

void common_laya_deleter::operator()(common_laya * laya) {
    delete laya;
}

common_laya_ptr common_laya_init(llama_context * ctx) {
    const llama_model * model = llama_get_model(ctx);

    char arch[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    if (std::string(arch) != "laya") {
        throw std::invalid_argument(std::string("expected a laya model, got '") + arch + "'");
    }
    if (llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_RANK) {
        throw std::invalid_argument("the context must use LLAMA_POOLING_TYPE_RANK, see common_laya_context_params");
    }

    int n_act = 0;
    {
        char buf[32] = {};
        llama_model_meta_val_str(model, "laya.decision.act_count", buf, sizeof(buf));
        n_act = std::atoi(buf);
    }
    const int n_cls_out = llama_model_n_cls_out(model);
    if (n_act < 1 || n_cls_out <= n_act) {
        throw std::invalid_argument("the model has " + std::to_string(n_cls_out) + " outputs per sequence for " + std::to_string(n_act) + " act outputs");
    }

    return common_laya_ptr(new common_laya {
        /*.ctx           =*/ ctx,
        /*.tok           =*/ laya_tokenizer(llama_model_get_vocab(model)),
        /*.cfg           =*/ load_config(model),
        /*.n_act         =*/ n_act,
        /*.n_max_options =*/ (size_t) (n_cls_out - n_act),
    });
}

struct laya_row {
    size_t    state;
    size_t    question;
    size_t    state_tokens;
    laya_item item;
};

static common_laya_result laya_predict(const common_laya & laya, const json & request) {
    llama_context * ctx = laya.ctx;

    const laya_tokenizer & tok = laya.tok;
    const laya_config    & cfg = laya.cfg;

    const int    n_act         = laya.n_act;
    const size_t n_max_options = laya.n_max_options;
    const int    n_cls_out     = n_act + (int) n_max_options;

    const int n_batch = std::min(llama_n_batch(ctx), llama_n_ubatch(ctx));

    if (!request.is_object()) {
        throw std::invalid_argument("the request must be an object");
    }

    const int max_len      = request.value("max_len",      cfg.max_len);
    const int head_max_len = request.value("head_max_len", cfg.head_max_len);
    if (max_len <= 0 || head_max_len <= 0) {
        throw std::invalid_argument("max_len and head_max_len must be positive, got " + std::to_string(max_len) + " and " + std::to_string(head_max_len));
    }

    const bool batch_request = request.contains("states");
    const json states = batch_request ? request["states"] : json::array({ request.value("state", json()) });
    if (!states.is_array()) {
        throw std::invalid_argument("'states' must be a list");
    }
    for (const auto & st : states) {
        if (st.is_null()) {
            throw std::invalid_argument("state must not be null; pass a string, object or list");
        }
    }

    const json qdefs = request.value("questions", json::object());
    if (!qdefs.is_object()) {
        throw std::invalid_argument("questions must be an object of question id -> definition");
    }

    // everything up to the state depends only on the question
    std::vector<laya_question> questions;
    std::vector<laya_head>     heads;
    for (const auto & [id, qdef] : qdefs.items()) {
        questions.push_back(parse_question(id, qdef));
        heads.push_back(build_head(tok, questions.back(), head_max_len));
        if (heads.back().n_options > n_max_options) {
            throw std::invalid_argument("question '" + id + "' has more than " + std::to_string(n_max_options) + " options");
        }
    }

    // Agent._encode_state
    std::vector<laya_row> rows;
    for (size_t s = 0; s < states.size(); ++s) {
        const json & st = states[s];
        const std::string text = st.is_string() ? st.get<std::string>() : py_dumps(st);
        // conversation lists are serialized newest-last: keep the newest turns
        const bool truncate_left = st.is_array();

        const auto state_ids = tok.tokenize(replace_all(text, tok.mask_text, " "));

        for (size_t qi = 0; qi < questions.size(); ++qi) {
            laya_item item = build_sequence(heads[qi], state_ids, tok.sep, max_len, truncate_left);
            if (item.markers.size() != heads[qi].n_options) {
                throw std::invalid_argument("question '" + questions[qi].id + "' options exceed head_max_len=" + std::to_string(head_max_len));
            }
            if ((int) item.ids.size() > n_batch) {
                throw std::invalid_argument("a sequence of " + std::to_string(item.ids.size()) + " tokens does not fit the batch size " + std::to_string(n_batch));
            }
            rows.push_back({ s, qi, state_ids.size(), std::move(item) });
        }
    }

    common_laya_result res;
    res.sequences.resize(rows.size());

    // forward passes, packing as many sequences as fit in one batch
    const int n_seq_max = llama_n_seq_max(ctx);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    const int64_t t_start_us = ggml_time_us();

    for (size_t r0 = 0; r0 < rows.size(); ) {
        common_batch_clear(batch);

        size_t r1 = r0;
        while (r1 < rows.size() && (int) (r1 - r0) < n_seq_max && batch.n_tokens + (int) rows[r1].item.ids.size() <= n_batch) {
            const auto & ids = rows[r1].item.ids;
            for (size_t i = 0; i < ids.size(); ++i) {
                common_batch_add(batch, ids[i], i, { (llama_seq_id) (r1 - r0) }, true);
            }
            ++r1;
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("llama_decode failed");
        }
        res.n_tokens += batch.n_tokens;
        res.n_passes++;

        for (size_t r = r0; r < r1; ++r) {
            const float * out = llama_get_embeddings_seq(ctx, r - r0);
            if (out == nullptr) {
                llama_batch_free(batch);
                throw std::runtime_error("no decision output for a sequence");
            }

            auto & seq = res.sequences[r];
            seq.state    = rows[r].state;
            seq.question = questions[rows[r].question].id;
            seq.tokens   = rows[r].item.ids;
            seq.markers  = rows[r].item.markers;
            seq.act.assign(out, out + n_act);
            seq.logits.assign(out + n_act, out + n_act + seq.markers.size());
        }

        r0 = r1;
    }

    llama_batch_free(batch);

    res.t_ms = (ggml_time_us() - t_start_us) / 1000.0;

    // Agent._decode_answers
    std::vector<float> out(n_cls_out);

    json results = json::array();
    for (size_t s = 0; s < states.size(); ++s) {
        json answers   = json::object();
        json collapsed = json::object();
        json truncated_questions = json::array();
        size_t n_tokens     = 0;
        size_t state_tokens = 0;
        size_t dropped      = 0;
        for (size_t r = 0; r < rows.size(); ++r) {
            if (rows[r].state != s) {
                continue;
            }
            const auto & q    = questions[rows[r].question];
            const auto & item = rows[r].item;
            const auto & seq  = res.sequences[r];

            std::copy(seq.act.begin(),    seq.act.end(),    out.begin());
            std::copy(seq.logits.begin(), seq.logits.end(), out.begin() + n_act);

            answers[q.id] = decode_answer(cfg, q, out.data(), n_act, item.markers.size());
            n_tokens += item.ids.size();

            // the questions share one state but not one head budget: report the worst case
            state_tokens = rows[r].state_tokens;
            dropped      = std::max(dropped, item.state_tokens_dropped);
            if (item.state_tokens_dropped > 0) {
                truncated_questions.push_back(q.id);
            }

            const auto & head = heads[rows[r].question];
            if (head.n_options_distinct < head.n_options) {
                collapsed[q.id] = {
                    { "total",             head.n_options },
                    { "distinct",          head.n_options_distinct },
                    { "tokens_per_option", head.tokens_per_option < 0 ? json() : json(head.tokens_per_option) },
                };
            }
        }

        json usage = {
            { "input_tokens",         n_tokens },
            { "output_tokens",        0 },
            { "state_tokens",         state_tokens },
            { "state_tokens_dropped", dropped },
            { "truncated",            dropped > 0 },
            { "truncated_questions",  truncated_questions },
        };
        if (!collapsed.empty()) {
            usage["options"] = collapsed;
        }

        results.push_back({ { "model", "laya-rl-agent" }, { "answers", answers }, { "usage", usage } });
    }

    res.response = batch_request ? results : results[0];

    return res;
}

common_laya_result common_laya_predict(const common_laya * laya, const json & request) {
    try {
        return laya_predict(*laya, request);
    } catch (const json::exception & e) {
        // a value of the wrong type in the request
        throw std::invalid_argument(e.what());
    }
}

void common_laya_warmup(const common_laya * laya) {
    common_laya_predict(laya, {
        { "state",     "warmup" },
        { "questions", { { "warmup", { { "type", "noul" }, { "instructions", "warmup" } } } } },
    });
}
