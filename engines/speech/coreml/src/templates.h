#pragma once

// The templates a library generates from.
//
// On Apple builds with <ENGINE>_COREML_GENERATE, qvac_coreml_add_to(...
// TEMPLATES <dirs>) embeds every template file uncompressed in __TEXT,__const
// (one .incbin assembler stub per file) and generates the table that
// embedded_files() returns: {template id, file name, begin, end, sha256}, the
// sha256 computed by CMake. Builds without templates get an empty table.
// Linux test builds embed nothing and read template directories from the
// source tree (recipe::Template::from_dir).

#include "qvac_coreml/recipe.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifndef QVAC_COREML_NS
#error "QVAC_COREML_NS must be defined"
#endif

namespace QVAC_COREML_NS {
namespace templates {

struct EmbeddedFile {
    const char * id;          // template id (the template directory's name)
    const char * name;        // e.g. model.mlmodel
    const uint8_t * begin;
    const uint8_t * end;
    const char * sha256_hex;  // file(SHA256) at configure time
};

// The generated table (qvac_coreml.cmake). `count` receives its length.
const EmbeddedFile * embedded_files(size_t & count);

// The embedded template ids, sorted.
std::vector<std::string> embedded_ids();

// The embedded template `id`, parsed once per process and kept until the
// process ends (exit-time static destructors included); nullptr when this
// library does not embed it. Throws std::runtime_error / recipe::RecipeError
// when the embedded files do not form a valid template (a build defect).
std::shared_ptr<const recipe::Template> embedded(const std::string & id);

// The template to generate `id` with. With the development override compiled
// in (template_dir_override_compiled()) and a non-empty `template_dir`, it is
// read from <template_dir>/<id> on every call (nullptr when that directory
// does not exist); otherwise it is embedded(id).
std::shared_ptr<const recipe::Template> find(const std::string & id, const std::string & template_dir = {});

}  // namespace templates
}  // namespace QVAC_COREML_NS
