#include "templates.h"

#include "qvac_coreml/options.h"

#include <algorithm>
#include <array>
#include <map>
#include <mutex>
#include <stdexcept>
#include <sys/stat.h>

namespace QVAC_COREML_NS {
namespace templates {

namespace {

bool parse_hex32(const char * hex, std::array<uint8_t, 32> & out) {
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < 32; ++i) {
        const int hi = nib(hex[2 * i]);
        const int lo = hi < 0 ? -1 : nib(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = static_cast<uint8_t>(hi << 4 | lo);
    }
    return hex[64] == '\0';
}

}  // namespace

std::vector<std::string> embedded_ids() {
    size_t n = 0;
    const EmbeddedFile * f = embedded_files(n);
    std::vector<std::string> ids;
    for (size_t i = 0; i < n; ++i) {
        if (std::find(ids.begin(), ids.end(), f[i].id) == ids.end()) ids.emplace_back(f[i].id);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

std::shared_ptr<const recipe::Template> embedded(const std::string & id) {
    // Leaked on purpose, like the manager: prepare jobs on the runtime's
    // threads hold the templates through raw pointers and may still run
    // while the process exits, after static destructors.
    static auto * mu = new std::mutex();
    static auto * parsed = new std::map<std::string, std::shared_ptr<const recipe::Template>>();
    std::lock_guard<std::mutex> lk(*mu);
    if (auto it = parsed->find(id); it != parsed->end()) return it->second;

    size_t n = 0;
    const EmbeddedFile * f = embedded_files(n);
    std::vector<recipe::TemplateFile> files;
    for (size_t i = 0; i < n; ++i) {
        if (id != f[i].id) continue;
        recipe::TemplateFile tf;
        tf.name = f[i].name;
        tf.data = f[i].begin;
        tf.size = static_cast<size_t>(f[i].end - f[i].begin);
        if (!parse_hex32(f[i].sha256_hex, tf.sha256)) {
            throw std::runtime_error("embedded template " + id + ": bad sha256 for " + tf.name);
        }
        files.push_back(tf);
    }
    if (files.empty()) return nullptr;
    std::shared_ptr<const recipe::Template> t = recipe::Template::from_embedded(std::move(files));
    if (t->id() != id) throw std::runtime_error("embedded template " + id + ": template.json id is " + t->id());
    parsed->emplace(id, t);
    return t;
}

std::shared_ptr<const recipe::Template> find(const std::string & id, const std::string & template_dir) {
    if (template_dir_override_compiled() && !template_dir.empty()) {
        const std::string dir = template_dir + "/" + id;
        struct stat st {};
        if (::stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return nullptr;
        return recipe::Template::from_dir(dir);
    }
    return embedded(id);
}

}  // namespace templates
}  // namespace QVAC_COREML_NS
