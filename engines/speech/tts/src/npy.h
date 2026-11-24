#pragma once
// Minimal NumPy .npy reader (v1.0 little-endian float32/int32 only - enough for our tests).

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct npy_array {
    std::vector<int64_t> shape;
    std::string          dtype;   // e.g. "<f4", "<i4"
    std::vector<uint8_t> data;    // raw bytes

    size_t n_elements() const {
        size_t n = 1;
        for (auto d : shape) n *= (size_t)d;
        return n;
    }
};

inline void npy_read_exact(std::istream & stream, char * data, size_t size,
                           const char * field, const std::string & path) {
    if (size != 0 && !stream.read(data, size)) {
        throw std::runtime_error(std::string("npy_load: truncated ") + field + " in " + path);
    }
}

inline npy_array npy_load(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy_load: cannot open " + path);

    char magic[6];
    npy_read_exact(f, magic, sizeof(magic), "magic", path);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("npy_load: bad magic in " + path);

    uint8_t v_major, v_minor;
    npy_read_exact(f, (char*)&v_major, sizeof(v_major), "version", path);
    npy_read_exact(f, (char*)&v_minor, sizeof(v_minor), "version", path);

    uint32_t header_len;
    if (v_major == 1) {
        uint16_t h;
        npy_read_exact(f, (char*)&h, sizeof(h), "header length", path);
        header_len = h;
    } else {
        npy_read_exact(f, (char*)&header_len, sizeof(header_len), "header length", path);
    }

    std::string header(header_len, 0);
    npy_read_exact(f, header.data(), header.size(), "header", path);

    npy_array a;

    auto find_value = [&](const std::string & key) -> std::string {
        auto p = header.find("'" + key + "'");
        if (p == std::string::npos) throw std::runtime_error("npy_load: missing key " + key);
        p = header.find(":", p) + 1;
        while (p < header.size() && std::isspace((unsigned char)header[p])) ++p;
        if (header[p] == '\'') {
            ++p;
            auto e = header.find('\'', p);
            return header.substr(p, e - p);
        }
        if (header[p] == '(') {
            ++p;
            auto e = header.find(')', p);
            return header.substr(p, e - p);
        }
        auto e = header.find_first_of(",}", p);
        return header.substr(p, e - p);
    };

    a.dtype = find_value("descr");

    // Check fortran_order — we don't support it; warn loudly if present.
    auto fortran_pos = header.find("fortran_order");
    if (fortran_pos != std::string::npos) {
        auto colon = header.find(":", fortran_pos);
        if (colon != std::string::npos) {
            auto start = colon + 1;
            while (start < header.size() && std::isspace((unsigned char)header[start])) ++start;
            if (header.substr(start, 4) == "True") {
                throw std::runtime_error("npy_load: fortran_order=True not supported for " + path
                    + " — re-save with np.ascontiguousarray(...)");
            }
        }
    }

    std::string shape_str = find_value("shape");
    std::vector<int64_t> shape;
    size_t i = 0;
    while (i < shape_str.size()) {
        while (i < shape_str.size() && !std::isdigit((unsigned char)shape_str[i])) ++i;
        if (i >= shape_str.size()) break;
        size_t j = i;
        while (j < shape_str.size() && std::isdigit((unsigned char)shape_str[j])) ++j;
        shape.push_back(std::stoll(shape_str.substr(i, j - i)));
        i = j;
    }
    a.shape = shape;

    size_t n_elem = a.n_elements();
    size_t elem_size = 4; // default float32/int32
    if (a.dtype == "<f4" || a.dtype == "<i4" || a.dtype == "<u4") elem_size = 4;
    else if (a.dtype == "<f8" || a.dtype == "<i8") elem_size = 8;
    else if (a.dtype == "|b1") elem_size = 1;
    else if (a.dtype == "<f2" || a.dtype == "<i2" || a.dtype == "<u2") elem_size = 2;
    else if (a.dtype == "|i1" || a.dtype == "|u1") elem_size = 1;

    a.data.resize(n_elem * elem_size);
    npy_read_exact(f, (char*)a.data.data(), a.data.size(), "payload", path);
    return a;
}

inline const float * npy_as_f32(const npy_array & a) {
    if (a.dtype != "<f4") throw std::runtime_error("npy_as_f32: dtype " + a.dtype);
    return reinterpret_cast<const float*>(a.data.data());
}

inline const int32_t * npy_as_i32(const npy_array & a) {
    if (a.dtype != "<i4") throw std::runtime_error("npy_as_i32: dtype " + a.dtype);
    return reinterpret_cast<const int32_t*>(a.data.data());
}

// Minimal NumPy .npy writer (v1.0 little-endian).  Handles the types our C++
// voice-conditioning tensors actually use: float32 and int32 with arbitrary
// shape.  No Fortran order, no object dtype.  Output is byte-identical to
// what `np.save(path, np.ascontiguousarray(arr))` produces for these dtypes.
inline void npy_save(const std::string & path,
                     const std::string & dtype,           // "<f4" or "<i4"
                     const std::vector<int64_t> & shape,
                     const void * data,
                     size_t elem_size)
{
    // Build the header dict.
    std::string shape_str;
    if (shape.empty()) {
        shape_str = "()";
    } else {
        shape_str = "(";
        for (size_t i = 0; i < shape.size(); ++i) {
            shape_str += std::to_string(shape[i]);
            if (shape.size() == 1 || i + 1 < shape.size()) shape_str += ",";
            if (i + 1 < shape.size()) shape_str += " ";
        }
        shape_str += ")";
    }
    std::string header = "{'descr': '" + dtype + "', 'fortran_order': False, 'shape': "
                       + shape_str + ", }";
    // Pad header so (10 bytes magic+version+hdr_len) + header.size() + 1 ('\n')
    // is a multiple of 64 bytes (NumPy padding rule).
    const size_t preamble = 10;
    size_t total = preamble + header.size() + 1;
    while (total % 64 != 0) { header += ' '; ++total; }
    header += '\n';

    if (header.size() > 0xFFFF)
        throw std::runtime_error("npy_save: header too large for v1.0 format");

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy_save: cannot open " + path);
    f.write("\x93NUMPY", 6);
    uint8_t v_major = 1, v_minor = 0;
    f.write((const char*)&v_major, 1);
    f.write((const char*)&v_minor, 1);
    uint16_t hlen = (uint16_t)header.size();
    f.write((const char*)&hlen, 2);
    f.write(header.data(), header.size());

    size_t n_elem = 1;
    for (auto d : shape) n_elem *= (size_t)d;
    f.write((const char*)data, n_elem * elem_size);
}

inline void npy_save_f32(const std::string & path,
                         const std::vector<int64_t> & shape,
                         const float * data) {
    npy_save(path, "<f4", shape, data, 4);
}

inline void npy_save_i32(const std::string & path,
                         const std::vector<int64_t> & shape,
                         const int32_t * data) {
    npy_save(path, "<i4", shape, data, 4);
}

struct compare_stats {
    double max_abs_err = 0;
    double mean_abs_err = 0;
    double rms_err = 0;
    size_t n = 0;
    double expected_abs_max = 0;
    double rel_err = 0;   // max_abs_err / max(|expected|)
    // Every comparison below is a `>` against a running maximum, and `>` is
    // false for NaN. Counting non-finite values is what keeps a NaN result
    // from reading as a zero error and clearing every tolerance.
    size_t non_finite = 0;
};

inline compare_stats compare_f32(const float * got, const float * expected, size_t n) {
    compare_stats s;
    double sum_abs = 0, sum_sq = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(expected[i])) {
            ++s.non_finite;
            continue;
        }
        double d = std::fabs((double)got[i] - (double)expected[i]);
        sum_abs += d;
        sum_sq += d * d;
        if (d > s.max_abs_err) s.max_abs_err = d;
        double e = std::fabs((double)expected[i]);
        if (e > s.expected_abs_max) s.expected_abs_max = e;
    }
    s.n = n;
    const size_t finite = n - s.non_finite;
    s.mean_abs_err = finite ? sum_abs / (double) finite : 0.0;
    s.rms_err = finite ? std::sqrt(sum_sq / (double) finite) : 0.0;
    s.rel_err = s.expected_abs_max > 0 ? (s.max_abs_err / s.expected_abs_max) : s.max_abs_err;
    return s;
}

// Single gate for every parity check: a tolerance only means something once
// the comparison is known to have looked at real numbers.
inline bool compare_within(const compare_stats & s, double tolerance) {
    return s.non_finite == 0 && s.n > 0 && s.rel_err < tolerance;
}

inline void print_compare(const char * name, const compare_stats & s) {
    fprintf(stderr, "  [%s] n=%zu  max_abs=%.3e  mean_abs=%.3e  rms=%.3e  max|ref|=%.3e  rel=%.3e  non_finite=%zu\n",
        name, s.n, s.max_abs_err, s.mean_abs_err, s.rms_err, s.expected_abs_max, s.rel_err, s.non_finite);
}
