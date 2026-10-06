#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr const char * KERNEL_EXTENSION = ".cl";

std::string strip_comments(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        if (text.compare(i, 2, "//") == 0) {
            while (i < text.size() && text[i] != '\n') {
                i++;
            }
        } else if (text.compare(i, 2, "/*") == 0) {
            const size_t end = text.find("*/", i + 2);
            i = end == std::string::npos ? text.size() : end + 2;
            out.push_back(' ');
        } else {
            out.push_back(text[i]);
            i++;
        }
    }
    return out;
}

bool is_preprocessor_line(const std::string & line) {
    const size_t first = line.find_first_not_of(" \t");
    return first != std::string::npos && line[first] == '#';
}

std::string strip_preprocessor(const std::string & text) {
    std::istringstream in(text);
    std::string out;
    std::string line;
    bool continuation = false;
    while (std::getline(in, line)) {
        const bool directive = continuation || is_preprocessor_line(line);
        continuation = directive && !line.empty() && line.back() == '\\';
        out += directive ? std::string() : line;
        out.push_back('\n');
    }
    return out;
}

std::string strip_string_literals(const std::string & text) {
    return std::regex_replace(text, std::regex("\"([^\"\\\\]|\\\\.)*\""), "\"\"");
}

std::string normalize(const std::string & text) {
    return strip_string_literals(strip_preprocessor(strip_comments(text)));
}

bool is_identifier_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

std::string identifier_before(const std::string & text, size_t pos) {
    size_t end = pos;
    while (end > 0 && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        end--;
    }
    size_t begin = end;
    while (begin > 0 && is_identifier_char(text[begin - 1])) {
        begin--;
    }
    return text.substr(begin, end - begin);
}

size_t matching_paren(const std::string & text, size_t open) {
    int depth = 0;
    for (size_t i = open; i < text.size(); i++) {
        if (text[i] == '(') {
            depth++;
        } else if (text[i] == ')') {
            depth--;
            if (depth == 0) {
                return i;
            }
        }
    }
    return std::string::npos;
}

size_t skip_attributes(const std::string & text, size_t pos) {
    static const std::string attribute = "__attribute__";
    while (true) {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
            pos++;
        }
        if (text.compare(pos, attribute.size(), attribute) != 0) {
            return pos;
        }
        const size_t open = text.find('(', pos);
        const size_t close = open == std::string::npos ? std::string::npos : matching_paren(text, open);
        if (close == std::string::npos) {
            return pos;
        }
        pos = close + 1;
    }
}

bool is_definition_at(const std::string & text, size_t open, std::string & name) {
    name = identifier_before(text, open);
    if (name.empty() || std::isdigit(static_cast<unsigned char>(name[0]))) {
        return false;
    }
    const size_t close = matching_paren(text, open);
    if (close == std::string::npos) {
        return false;
    }
    const size_t body = skip_attributes(text, close + 1);
    return body < text.size() && text[body] == '{';
}

std::vector<std::string> function_definitions(const std::string & text) {
    std::vector<std::string> names;
    int brace_depth = 0;
    for (size_t i = 0; i < text.size(); i++) {
        const char c = text[i];
        if (c == '{') {
            brace_depth++;
        } else if (c == '}') {
            brace_depth--;
        } else if (c == '(' && brace_depth == 0) {
            std::string name;
            if (is_definition_at(text, i, name)) {
                names.push_back(name);
            }
            i = matching_paren(text, i);
            if (i == std::string::npos) {
                break;
            }
        }
    }
    return names;
}

std::vector<std::string> typedef_names(const std::string & text) {
    static const std::regex pattern("typedef\\s+(?:struct|union)\\b[^{;]*\\{[^}]*\\}\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*;");
    std::vector<std::string> names;
    for (std::sregex_iterator it(text.begin(), text.end(), pattern), end; it != end; ++it) {
        names.push_back((*it)[1]);
    }
    return names;
}

std::vector<std::string> duplicates(const std::vector<std::string> & names) {
    std::map<std::string, int> counts;
    for (const auto & name : names) {
        counts[name]++;
    }
    std::vector<std::string> out;
    for (const auto & entry : counts) {
        if (entry.second > 1) {
            out.push_back(entry.first);
        }
    }
    return out;
}

std::vector<std::string> duplicate_definitions(const std::string & source) {
    const std::string text = normalize(source);
    std::vector<std::string> names = function_definitions(text);
    const std::vector<std::string> typedefs = typedef_names(text);
    names.insert(names.end(), typedefs.begin(), typedefs.end());
    return duplicates(names);
}

std::string read_file(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::vector<std::filesystem::path> kernel_files(const std::filesystem::path & dir) {
    std::vector<std::filesystem::path> files;
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == KERNEL_EXTENSION) {
            files.push_back(entry.path());
        }
    }
    return files;
}

int report_duplicates(const std::filesystem::path & path, const std::vector<std::string> & names) {
    for (const auto & name : names) {
        fprintf(stderr, "%s: '%s' is defined more than once\n", path.string().c_str(), name.c_str());
    }
    return (int) names.size();
}

int check_kernel_dir(const std::filesystem::path & dir) {
    int failures = 0;
    const std::vector<std::filesystem::path> files = kernel_files(dir);
    for (const auto & path : files) {
        failures += report_duplicates(path, duplicate_definitions(read_file(path)));
    }
    printf("checked %zu kernel sources in %s\n", files.size(), dir.string().c_str());
    return failures;
}

bool detector_flags_redefinition() {
    const std::string source =
        "kernel void kernel_a(global float * x) { x[0] = 1.0f; }\n"
        "typedef struct { half d; char qs[32]; } block_q8_0;\n"
        "kernel void kernel_a(global char * q, global half * d) { q[0] = 0; }\n";
    const std::vector<std::string> found = duplicate_definitions(source);
    return found.size() == 1 && found[0] == "kernel_a";
}

bool detector_accepts_distinct_definitions() {
    const std::string source =
        "#define DUP(x) kernel void x(void) { } \\\n"
        "    kernel void x(void) { }\n"
        "// kernel void kernel_a(void) { }\n"
        "/* kernel void kernel_a(void) { } */\n"
        "inline float helper(float v) { if (v > 0.0f) { return v; } return -v; }\n"
        "REQD_SUBGROUP_SIZE_64 kernel void kernel_a(global float * x) __attribute__((reqd_work_group_size(64, 1, 1))) {\n"
        "    for (int i = 0; i < 4; i++) { x[i] = helper(x[i]); }\n"
        "}\n"
        "typedef struct { half d; char qs[32]; } block_q8_0;\n"
        "kernel void kernel_b(global block_q8_0 * b) { }\n";
    return duplicate_definitions(source).empty();
}

} // namespace

int main(int argc, char ** argv) {
    if (!detector_flags_redefinition() || !detector_accepts_distinct_definitions()) {
        fprintf(stderr, "duplicate-definition detector self-check failed\n");
        return 1;
    }
    if (argc < 2) {
        fprintf(stderr, "usage: %s <kernels-dir>\n", argv[0]);
        return 1;
    }
    return check_kernel_dir(argv[1]) == 0 ? 0 : 1;
}
