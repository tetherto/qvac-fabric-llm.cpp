#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <system_error>
#include <vector>

namespace tts_cpp::acestep {

inline constexpr int32_t   STAGE_DUMP_RANK          = 2;
inline constexpr size_t    STAGE_DUMP_HEADER_FIELDS = 3;
inline constexpr uintmax_t STAGE_DUMP_HEADER_BYTES  = STAGE_DUMP_HEADER_FIELDS * sizeof(int32_t);

struct StageDumpHeader {
    int32_t rank = 0;
    int32_t d0   = 0;
    int32_t d1   = 0;
};

inline bool stage_dump_header_valid(const StageDumpHeader & h) {
    return h.rank == STAGE_DUMP_RANK && h.d0 > 0 && h.d1 > 0;
}

inline uintmax_t stage_dump_payload_bytes(const StageDumpHeader & h) {
    return (uintmax_t) h.d0 * (uintmax_t) h.d1 * sizeof(float);
}

inline bool stage_dump_payload_present(const char * path, const StageDumpHeader & h) {
    std::error_code ec;
    const uintmax_t file_bytes = std::filesystem::file_size(path, ec);
    return !ec && file_bytes >= STAGE_DUMP_HEADER_BYTES &&
           file_bytes - STAGE_DUMP_HEADER_BYTES >= stage_dump_payload_bytes(h);
}

inline bool read_stage_dump_header(FILE * f, StageDumpHeader & h) {
    int32_t fields[STAGE_DUMP_HEADER_FIELDS];
    if (fread(fields, sizeof(int32_t), STAGE_DUMP_HEADER_FIELDS, f) != STAGE_DUMP_HEADER_FIELDS) return false;
    h.rank = fields[0];
    h.d0   = fields[1];
    h.d1   = fields[2];
    return true;
}

inline bool read_stage_dump(const char * tag, const char * path, std::vector<float> & out, int * d0, int * d1) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[%s] cannot open %s\n", tag, path);
        return false;
    }
    StageDumpHeader h;
    if (!read_stage_dump_header(f, h) || !stage_dump_header_valid(h)) {
        fprintf(stderr, "[%s] %s is not a rank-2 stage dump\n", tag, path);
        fclose(f);
        return false;
    }
    if (!stage_dump_payload_present(path, h)) {
        fprintf(stderr, "[%s] %s is truncated: header promises %d x %d floats\n", tag, path, h.d0, h.d1);
        fclose(f);
        return false;
    }
    const size_t n = (size_t) h.d0 * (size_t) h.d1;
    out.resize(n);
    const bool ok = fread(out.data(), sizeof(float), n, f) == n;
    fclose(f);
    if (!ok) {
        fprintf(stderr, "[%s] %s is truncated\n", tag, path);
        return false;
    }
    *d0 = h.d0;
    *d1 = h.d1;
    return true;
}

inline bool write_stage_dump(const char * tag, const char * path, const std::vector<float> & v, int d0, int d1) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[%s] cannot write %s\n", tag, path);
        return false;
    }
    const int32_t hdr[STAGE_DUMP_HEADER_FIELDS] = { STAGE_DUMP_RANK, d0, d1 };
    fwrite(hdr, sizeof(int32_t), STAGE_DUMP_HEADER_FIELDS, f);
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
    return true;
}

}  // namespace tts_cpp::acestep
