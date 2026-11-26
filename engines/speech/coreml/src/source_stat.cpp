// Stat helpers (no ggml): the (size, mtime, inode, device) identity of a source file.

#include "gguf_source.h"

#include <sys/stat.h>

namespace QVAC_COREML_NS {

namespace {

void fill(const struct stat & st, SourceStat & out) {
    out.size = static_cast<uint64_t>(st.st_size);
#if defined(__APPLE__)
    out.mtime_ns = static_cast<int64_t>(st.st_mtimespec.tv_sec) * 1000000000LL + st.st_mtimespec.tv_nsec;
#else
    out.mtime_ns = static_cast<int64_t>(st.st_mtim.tv_sec) * 1000000000LL + st.st_mtim.tv_nsec;
#endif
    out.ino = static_cast<uint64_t>(st.st_ino);
    out.dev = static_cast<uint64_t>(st.st_dev);
}

}  // namespace

bool stat_path(const std::string & path, SourceStat & out) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) return false;
    out.path = path;
    fill(st, out);
    return true;
}

bool stat_fd(int fd, SourceStat & out) {
    struct stat st{};
    if (::fstat(fd, &st) != 0) return false;
    fill(st, out);
    return true;
}

bool same_identity(const SourceStat & a, const SourceStat & b) {
    return a.size == b.size && a.mtime_ns == b.mtime_ns && a.ino == b.ino && a.dev == b.dev;
}

}  // namespace QVAC_COREML_NS
