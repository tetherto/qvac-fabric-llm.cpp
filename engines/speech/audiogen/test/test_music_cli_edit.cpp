#include "music_cli_edit.h"

#include <cstdio>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    ++checks;                                                                  \
    if (!(condition)) {                                                        \
      ++failures;                                                              \
      std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__,             \
                   #condition);                                                \
    }                                                                          \
  } while (0)

struct Arguments {
  explicit Arguments(std::vector<std::string> values)
      : storage(std::move(values)) {
    argv.reserve(storage.size());
    for (std::string &value : storage)
      argv.push_back(value.data());
  }

  int argc() const { return (int)argv.size(); }

  std::vector<std::string> storage;
  std::vector<char *> argv;
};

bool parse_arguments(std::vector<std::string> values,
                     tts_cpp::acestep::GenerateParams &params,
                     std::string &error) {
  Arguments arguments(std::move(values));
  error.clear();
  return tts_cpp::acestep::music_cli::parse_edit_flags(
      arguments.argc(), arguments.argv.data(), params, error);
}

void test_json_operation_order() {
  using namespace tts_cpp::acestep;
  const std::string json =
      R"({"operations":[)"
      R"({"type":"flow-edit","source_caption":"source one","target_caption":"target one"},)"
      R"({"type":"repaint","start":1.5,"end":3.0,"mode":"aggressive"},)"
      R"({"type":"flow-edit","source_caption":"source two","target_caption":"target two","n_avg":2})"
      R"(]})";
  std::vector<AudioEditParams> plan;
  std::string error;
  CHECK(music_cli::parse_edit_plan_json(json, plan, error));
  CHECK(error.empty());
  CHECK(plan.size() == 3);
  CHECK(std::holds_alternative<FlowEditParams>(plan[0]));
  CHECK(std::holds_alternative<RepaintParams>(plan[1]));
  CHECK(std::holds_alternative<FlowEditParams>(plan[2]));
  CHECK(std::get<FlowEditParams>(plan[0]).source_caption == "source one");
  CHECK(std::get<RepaintParams>(plan[1]).mode == RepaintMode::Aggressive);
  CHECK(std::get<FlowEditParams>(plan[2]).n_avg == 2);
}

void check_invalid_plan(const std::string &json, const char *expected) {
  using namespace tts_cpp::acestep;
  std::vector<AudioEditParams> plan;
  std::string error;
  CHECK(!music_cli::parse_edit_plan_json(json, plan, error));
  CHECK(error.find(expected) != std::string::npos);
  CHECK(plan.empty());
}

void test_invalid_json_plans() {
  check_invalid_plan("{}", "non-empty operations");
  check_invalid_plan(R"({"operations":[]})", "non-empty operations");
  check_invalid_plan(R"({"operations":[{}]})", "missing type");
  check_invalid_plan(R"({"operations":[{"type":"unknown"}]})", "unsupported");
  check_invalid_plan(R"({"operations":[{"type":"repaint","mode":"fast"}]})",
                     "conservative|balanced|aggressive");
  check_invalid_plan(
      R"({"operations":[{"type":"flow-edit","target_caption":"target"}]})",
      "source_caption is required");
  check_invalid_plan(
      R"({"operations":[{"type":"flow-edit","source_caption":"source"}]})",
      "target_caption is required");
  check_invalid_plan(
      R"({"operations":[{"type":"flow-edit","source_caption":"source","target_caption":"target","n_min":0.8,"n_max":0.2}]})",
      "0 <= n_min");
  check_invalid_plan(
      R"({"operations":[{"type":"flow-edit","source_caption":"source","target_caption":"target","n_avg":"many"}]})",
      "must be an integer");
}

void test_every_flow_flag_conflicts_with_edit_plan() {
  const std::vector<std::pair<std::string, std::string>> flow_flags = {
      {"--flow-source-caption", "source"},
      {"--flow-source-lyrics", "lyrics"},
      {"--flow-n-min", "0.1"},
      {"--flow-n-max", "0.9"},
      {"--flow-n-avg", "2"},
  };
  for (const auto &option : flow_flags) {
    tts_cpp::acestep::GenerateParams params;
    std::string error;
    CHECK(!parse_arguments({"music-cli", "--edit-plan", "unused.json",
                            option.first, option.second},
                           params, error));
    CHECK(error.find("cannot be combined") != std::string::npos);
    CHECK(params.edit_plan.empty());
  }
}

void test_orphan_and_missing_flow_values() {
  {
    tts_cpp::acestep::GenerateParams params;
    std::string error;
    CHECK(
        !parse_arguments({"music-cli", "--flow-n-min", "0.2"}, params, error));
    CHECK(error.find("--flow-source-caption is required") != std::string::npos);
  }
  {
    tts_cpp::acestep::GenerateParams params;
    std::string error;
    CHECK(!parse_arguments({"music-cli", "--flow-source-caption"}, params,
                           error));
    CHECK(error.find("requires a value") != std::string::npos);
  }
  {
    tts_cpp::acestep::GenerateParams params;
    params.caption = "target";
    std::string error;
    CHECK(!parse_arguments(
        {"music-cli", "--flow-source-caption", "source", "--flow-n-max"},
        params, error));
    CHECK(error.find("--flow-n-max requires a value") != std::string::npos);
  }
  {
    tts_cpp::acestep::GenerateParams params;
    params.caption = "target";
    std::string error;
    CHECK(!parse_arguments(
        {"music-cli", "--flow-source-caption", "--flow-n-min", "0.2"}, params,
        error));
    CHECK(error.find("--flow-source-caption requires a value") !=
          std::string::npos);
  }
}

void test_standalone_order_and_values() {
  using namespace tts_cpp::acestep;
  GenerateParams params;
  params.caption = "target caption";
  params.lyrics = "target lyrics";
  std::string error;
  CHECK(parse_arguments(
      {
          "music-cli",
          "--flow-source-caption",
          "source caption",
          "--flow-source-lyrics",
          "source lyrics",
          "--flow-n-min",
          "0.2",
          "--flow-n-max",
          "0.8",
          "--flow-n-avg",
          "3",
          "--repaint-start",
          "1.0",
          "--repaint-end",
          "2.0",
          "--repaint-mode",
          "conservative",
      },
      params, error));
  CHECK(error.empty());
  CHECK(params.edit_plan.size() == 2);
  CHECK(std::holds_alternative<RepaintParams>(params.edit_plan[0]));
  CHECK(std::holds_alternative<FlowEditParams>(params.edit_plan[1]));
  const RepaintParams &repaint = std::get<RepaintParams>(params.edit_plan[0]);
  const FlowEditParams &flow = std::get<FlowEditParams>(params.edit_plan[1]);
  CHECK(repaint.mode == RepaintMode::Conservative);
  CHECK(repaint.start_seconds == 1.0f);
  CHECK(repaint.end_seconds == 2.0f);
  CHECK(flow.source_caption == "source caption");
  CHECK(flow.source_lyrics == "source lyrics");
  CHECK(flow.target_caption == "target caption");
  CHECK(flow.target_lyrics == "target lyrics");
  CHECK(flow.n_min == 0.2f);
  CHECK(flow.n_max == 0.8f);
  CHECK(flow.n_avg == 3);
}

void test_edit_configuration() {
  using namespace tts_cpp::acestep;
  GenerateParams params;
  params.edit_plan.emplace_back(RepaintParams{});
  std::string error;
  CHECK(!music_cli::validate_edit_configuration(params, error));
  CHECK(error.find("--src-audio") != std::string::npos);

  params.source_audio = {0.0f, 0.0f};
  params.task_type = "cover-nofsq";
  error.clear();
  CHECK(!music_cli::validate_edit_configuration(params, error));
  CHECK(error.find("--task cover-nofsq") != std::string::npos);

  params.task_type = "text2music";
  error.clear();
  CHECK(music_cli::validate_edit_configuration(params, error));
  CHECK(error.empty());
}

} // namespace

int main() {
  test_json_operation_order();
  test_invalid_json_plans();
  test_every_flow_flag_conflicts_with_edit_plan();
  test_orphan_and_missing_flow_values();
  test_standalone_order_and_values();
  test_edit_configuration();

  std::fprintf(stderr, "[test-music-cli-edit] %d/%d checks passed\n",
               checks - failures, checks);
  return failures == 0 ? 0 : 1;
}
