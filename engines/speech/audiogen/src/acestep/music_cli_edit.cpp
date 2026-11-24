#include "music_cli_edit.h"

#include "audio_edit.h"
#include "mini_json.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <utility>

namespace tts_cpp::acestep::music_cli {
namespace {

constexpr char kEditPlanOption[] = "--edit-plan";
constexpr char kRepaintStartOption[] = "--repaint-start";
constexpr char kRepaintEndOption[] = "--repaint-end";
constexpr char kRepaintModeOption[] = "--repaint-mode";
constexpr char kRepaintStrengthOption[] = "--repaint-strength";
constexpr char kFlowSourceCaptionOption[] = "--flow-source-caption";
constexpr char kFlowSourceLyricsOption[] = "--flow-source-lyrics";
constexpr char kFlowMinimumOption[] = "--flow-n-min";
constexpr char kFlowMaximumOption[] = "--flow-n-max";
constexpr char kFlowAverageOption[] = "--flow-n-avg";
constexpr char kTextToMusicTask[] = "text2music";
constexpr char kOperationsField[] = "operations";

bool option_present(int argc, char **argv, const char *key) {
  for (int index = 1; index < argc; ++index) {
    if (strcmp(argv[index], key) == 0)
      return true;
  }
  return false;
}

bool read_option(int argc, char **argv, const char *key, const char *&value,
                 std::string &error) {
  value = nullptr;
  for (int index = 1; index < argc; ++index) {
    if (strcmp(argv[index], key) != 0)
      continue;
    if (index + 1 >= argc || strncmp(argv[index + 1], "--", 2) == 0) {
      error = std::string(key) + " requires a value";
      return false;
    }
    value = argv[index + 1];
    return true;
  }
  return true;
}

bool parse_float(const char *text, const char *option, float &value,
                 std::string &error) {
  char *end = nullptr;
  errno = 0;
  const float parsed = strtof(text, &end);
  if (errno == ERANGE || end == text || *end != '\0' ||
      !std::isfinite(parsed)) {
    error = std::string(option) + " must be a finite number";
    return false;
  }
  value = parsed;
  return true;
}

bool parse_integer(const char *text, const char *option, int &value,
                   std::string &error) {
  char *end = nullptr;
  errno = 0;
  const long parsed = strtol(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0' || parsed < INT_MIN ||
      parsed > INT_MAX) {
    error = std::string(option) + " must be an integer";
    return false;
  }
  value = (int)parsed;
  return true;
}

bool parse_json_float(const std::string &text, const char *field, float &value,
                      std::string &error) {
  return parse_float(text.c_str(), field, value, error);
}

bool parse_json_integer(const std::string &text, const char *field, int &value,
                        std::string &error) {
  return parse_integer(text.c_str(), field, value, error);
}

bool parse_repaint_mode(const std::string &value, RepaintMode &mode) {
  if (value == "conservative")
    mode = RepaintMode::Conservative;
  else if (value == "balanced")
    mode = RepaintMode::Balanced;
  else if (value == "aggressive")
    mode = RepaintMode::Aggressive;
  else
    return false;
  return true;
}

bool parse_repaint_operation(const std::string &operation,
                             std::vector<AudioEditParams> &plan,
                             std::string &error) {
  RepaintParams repaint;
  std::string value;
  if (json_field(operation, "start", value) &&
      !parse_json_float(value, "repaint start", repaint.start_seconds, error)) {
    return false;
  }
  if (json_field(operation, "end", value) &&
      !parse_json_float(value, "repaint end", repaint.end_seconds, error)) {
    return false;
  }
  if (json_field(operation, "strength", value) &&
      !parse_json_float(value, "repaint strength", repaint.strength, error)) {
    return false;
  }
  if (json_field(operation, "caption", value))
    repaint.caption = value;
  if (json_field(operation, "lyrics", value))
    repaint.lyrics = value;
  if (json_field(operation, "mode", value) &&
      !parse_repaint_mode(value, repaint.mode)) {
    error = "repaint mode must be conservative|balanced|aggressive";
    return false;
  }
  plan.emplace_back(std::move(repaint));
  return true;
}

bool parse_flow_operation(const std::string &operation,
                          std::vector<AudioEditParams> &plan,
                          std::string &error) {
  FlowEditParams flow;
  std::string value;
  if (json_field(operation, "source_caption", value)) {
    flow.source_caption = value;
  }
  if (json_field(operation, "source_lyrics", value)) {
    flow.source_lyrics = value;
  }
  if (json_field(operation, "target_caption", value)) {
    flow.target_caption = value;
  }
  if (json_field(operation, "target_lyrics", value)) {
    flow.target_lyrics = value;
  }
  if (json_field(operation, "n_min", value) &&
      !parse_json_float(value, "flow-edit n_min", flow.n_min, error)) {
    return false;
  }
  if (json_field(operation, "n_max", value) &&
      !parse_json_float(value, "flow-edit n_max", flow.n_max, error)) {
    return false;
  }
  if (json_field(operation, "n_avg", value) &&
      !parse_json_integer(value, "flow-edit n_avg", flow.n_avg, error)) {
    return false;
  }
  error = validate_flow_edit_params(flow);
  if (!error.empty())
    return false;
  plan.emplace_back(std::move(flow));
  return true;
}

bool parse_edit_operation(const std::string &operation,
                          std::vector<AudioEditParams> &plan,
                          std::string &error) {
  std::string type;
  if (!json_field(operation, "type", type)) {
    error = "edit plan operation is missing type";
    return false;
  }
  if (type == "repaint") {
    return parse_repaint_operation(operation, plan, error);
  }
  if (type == "flow-edit") {
    return parse_flow_operation(operation, plan, error);
  }
  error = "unsupported edit plan operation type '" + type + "'";
  return false;
}

bool has_standalone_repaint_flags(int argc, char **argv) {
  return option_present(argc, argv, kRepaintStartOption) ||
         option_present(argc, argv, kRepaintEndOption) ||
         option_present(argc, argv, kRepaintModeOption) ||
         option_present(argc, argv, kRepaintStrengthOption);
}

bool has_standalone_flow_flags(int argc, char **argv) {
  return option_present(argc, argv, kFlowSourceCaptionOption) ||
         option_present(argc, argv, kFlowSourceLyricsOption) ||
         option_present(argc, argv, kFlowMinimumOption) ||
         option_present(argc, argv, kFlowMaximumOption) ||
         option_present(argc, argv, kFlowAverageOption);
}

bool parse_standalone_repaint(int argc, char **argv,
                              std::vector<AudioEditParams> &plan,
                              std::string &error) {
  if (!has_standalone_repaint_flags(argc, argv))
    return true;
  RepaintParams repaint;
  const char *value = nullptr;
  if (!read_option(argc, argv, kRepaintStartOption, value, error))
    return false;
  if (value &&
      !parse_float(value, kRepaintStartOption, repaint.start_seconds, error)) {
    return false;
  }
  if (!read_option(argc, argv, kRepaintEndOption, value, error))
    return false;
  if (value &&
      !parse_float(value, kRepaintEndOption, repaint.end_seconds, error)) {
    return false;
  }
  if (!read_option(argc, argv, kRepaintStrengthOption, value, error)) {
    return false;
  }
  if (value &&
      !parse_float(value, kRepaintStrengthOption, repaint.strength, error)) {
    return false;
  }
  if (!read_option(argc, argv, kRepaintModeOption, value, error))
    return false;
  if (value && !parse_repaint_mode(value, repaint.mode)) {
    error = "--repaint-mode must be conservative|balanced|aggressive";
    return false;
  }
  plan.emplace_back(std::move(repaint));
  return true;
}

bool parse_standalone_flow(int argc, char **argv, const GenerateParams &params,
                           std::vector<AudioEditParams> &plan,
                           std::string &error) {
  if (!has_standalone_flow_flags(argc, argv))
    return true;
  const char *value = nullptr;
  if (!read_option(argc, argv, kFlowSourceCaptionOption, value, error)) {
    return false;
  }
  if (!value) {
    error = "--flow-source-caption is required when using standalone flow-edit "
            "flags";
    return false;
  }
  FlowEditParams flow;
  flow.source_caption = value;
  if (!read_option(argc, argv, kFlowSourceLyricsOption, value, error)) {
    return false;
  }
  if (value)
    flow.source_lyrics = value;
  flow.target_caption = params.caption;
  flow.target_lyrics = params.lyrics;
  if (!read_option(argc, argv, kFlowMinimumOption, value, error))
    return false;
  if (value && !parse_float(value, kFlowMinimumOption, flow.n_min, error)) {
    return false;
  }
  if (!read_option(argc, argv, kFlowMaximumOption, value, error))
    return false;
  if (value && !parse_float(value, kFlowMaximumOption, flow.n_max, error)) {
    return false;
  }
  if (!read_option(argc, argv, kFlowAverageOption, value, error))
    return false;
  if (value && !parse_integer(value, kFlowAverageOption, flow.n_avg, error)) {
    return false;
  }
  error = validate_flow_edit_params(flow);
  if (!error.empty())
    return false;
  plan.emplace_back(std::move(flow));
  return true;
}

} // namespace

bool parse_edit_plan_json(const std::string &json,
                          std::vector<AudioEditParams> &plan,
                          std::string &error) {
  const std::vector<std::string> operations =
      json_array_objects(json, kOperationsField);
  if (operations.empty()) {
    error = "edit plan requires a non-empty operations array";
    return false;
  }
  std::vector<AudioEditParams> parsed;
  parsed.reserve(operations.size());
  for (const std::string &operation : operations) {
    if (!parse_edit_operation(operation, parsed, error))
      return false;
  }
  plan.insert(plan.end(), std::make_move_iterator(parsed.begin()),
              std::make_move_iterator(parsed.end()));
  return true;
}

bool parse_edit_flags(int argc, char **argv, GenerateParams &params,
                      std::string &error) {
  const bool standalone_repaint = has_standalone_repaint_flags(argc, argv);
  const bool standalone_flow = has_standalone_flow_flags(argc, argv);
  const bool has_edit_plan = option_present(argc, argv, kEditPlanOption);
  const char *edit_plan_path = nullptr;
  if (!read_option(argc, argv, kEditPlanOption, edit_plan_path, error)) {
    return false;
  }
  if (has_edit_plan) {
    if (standalone_repaint || standalone_flow) {
      error = "--edit-plan cannot be combined with standalone edit flags";
      return false;
    }
    const std::string json = read_text_file(edit_plan_path);
    if (json.empty()) {
      error = "edit plan failed: cannot read edit plan";
      return false;
    }
    if (!parse_edit_plan_json(json, params.edit_plan, error)) {
      error = "edit plan failed: " + error;
      return false;
    }
    return true;
  }

  std::vector<AudioEditParams> parsed;
  if (!parse_standalone_repaint(argc, argv, parsed, error) ||
      !parse_standalone_flow(argc, argv, params, parsed, error)) {
    return false;
  }
  params.edit_plan.insert(params.edit_plan.end(),
                          std::make_move_iterator(parsed.begin()),
                          std::make_move_iterator(parsed.end()));
  return true;
}

bool validate_edit_configuration(const GenerateParams &params,
                                 std::string &error) {
  if (params.edit_plan.empty())
    return true;
  if (params.source_audio.empty()) {
    error = "editing requires --src-audio <48-kHz PCM16 WAV>";
    return false;
  }
  if (params.task_type != kTextToMusicTask) {
    error = "editing cannot be combined with --task " + params.task_type;
    return false;
  }
  return true;
}

} // namespace tts_cpp::acestep::music_cli
