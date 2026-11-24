#include "mini_json.h"

#include <cstdio>
#include <utility>

namespace tts_cpp::acestep::music_cli {

std::string read_text_file(const char *path) {
  FILE *file = fopen(path, "rb");
  if (!file)
    return {};
  fseek(file, 0, SEEK_END);
  const long length = ftell(file);
  fseek(file, 0, SEEK_SET);
  if (length < 0) {
    fclose(file);
    return {};
  }
  std::string text((size_t)length, '\0');
  const size_t read = fread(text.data(), 1, (size_t)length, file);
  fclose(file);
  text.resize(read);
  return text;
}

bool json_field(const std::string &json, const char *key, std::string &value) {
  const std::string needle = std::string("\"") + key + "\"";
  size_t cursor = json.find(needle);
  if (cursor == std::string::npos)
    return false;
  cursor = json.find(':', cursor + needle.size());
  if (cursor == std::string::npos)
    return false;
  ++cursor;
  while (cursor < json.size() &&
         (json[cursor] == ' ' || json[cursor] == '\t' || json[cursor] == '\n' ||
          json[cursor] == '\r')) {
    ++cursor;
  }
  if (cursor >= json.size())
    return false;
  if (json[cursor] == '"') {
    size_t end = ++cursor;
    std::string parsed;
    while (end < json.size() && json[end] != '"') {
      if (json[end] == '\\' && end + 1 < json.size()) {
        switch (json[end + 1]) {
        case '"':
          parsed += '"';
          break;
        case '\\':
          parsed += '\\';
          break;
        case '/':
          parsed += '/';
          break;
        case 'b':
          parsed += '\b';
          break;
        case 'f':
          parsed += '\f';
          break;
        case 'n':
          parsed += '\n';
          break;
        case 'r':
          parsed += '\r';
          break;
        case 't':
          parsed += '\t';
          break;
        default:
          parsed += json[end + 1];
          break;
        }
        end += 2;
        continue;
      }
      parsed += json[end++];
    }
    if (end >= json.size())
      return false;
    value = std::move(parsed);
  } else {
    size_t end = cursor;
    while (end < json.size() && json[end] != ',' && json[end] != '}' &&
           json[end] != '\n') {
      ++end;
    }
    value = json.substr(cursor, end - cursor);
    while (!value.empty() && (value.back() == ' ' || value.back() == '\r' ||
                              value.back() == '\t')) {
      value.pop_back();
    }
  }
  return true;
}

std::vector<std::string> json_array_objects(const std::string &json,
                                            const char *key) {
  std::vector<std::string> objects;
  const std::string needle = std::string("\"") + key + "\"";
  size_t cursor = json.find(needle);
  if (cursor == std::string::npos)
    return objects;
  cursor = json.find('[', cursor + needle.size());
  if (cursor == std::string::npos)
    return objects;

  bool in_string = false;
  bool escaped = false;
  int object_depth = 0;
  size_t object_start = std::string::npos;
  for (++cursor; cursor < json.size(); ++cursor) {
    const char character = json[cursor];
    if (in_string) {
      if (escaped)
        escaped = false;
      else if (character == '\\')
        escaped = true;
      else if (character == '"')
        in_string = false;
      continue;
    }
    if (character == '"') {
      in_string = true;
    } else if (character == '{') {
      if (object_depth++ == 0)
        object_start = cursor;
    } else if (character == '}') {
      if (object_depth <= 0)
        return {};
      if (--object_depth == 0 && object_start != std::string::npos) {
        objects.push_back(json.substr(object_start, cursor - object_start + 1));
        object_start = std::string::npos;
      }
    } else if (character == ']' && object_depth == 0) {
      break;
    }
  }
  if (in_string || object_depth != 0)
    return {};
  return objects;
}

} // namespace tts_cpp::acestep::music_cli
