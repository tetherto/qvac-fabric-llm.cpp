#include "finetune.h"

#include "chat.h"
#include "common.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

ggml_opt_dataset_t common_opt_sft_dataset_init(
        struct llama_context * ctx,
        const std::string    & json_content,
        int64_t                stride,
        const std::string    & chat_template_path) {
    using json = nlohmann::json;

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    common_chat_templates_ptr chat_templates;
    std::string chat_template_source;
    if (!chat_template_path.empty()) {
        std::ifstream tmpl_file(chat_template_path);
        if (!tmpl_file.is_open()) {
            LOG_ERR("Warning: Failed to open chat template file: %s\n", chat_template_path.c_str());
        } else {
            chat_template_source.assign(std::istreambuf_iterator<char>(tmpl_file), std::istreambuf_iterator<char>());
            tmpl_file.close();
        }
    }

    try {
        chat_templates = common_chat_templates_init(llama_get_model(ctx), chat_template_source);
        if (chat_template_source.empty()) {
            LOG_INF("Using model's built-in chat template\n");
        } else {
            LOG_INF("Using custom chat template from: %s\n", chat_template_path.c_str());
        }
    } catch (const std::exception & e) {
        if (!chat_template_path.empty()) {
            LOG_ERR("Warning: Failed to parse chat template '%s': %s\n", chat_template_path.c_str(), e.what());
        } else {
            LOG_ERR("Warning: Failed to initialize chat template: %s\n", e.what());
        }
    }

    std::vector<json> conversations;
    std::istringstream content_stream(json_content);

    std::string line;
    while (std::getline(content_stream, line)) {
        if (line.empty() || line[0] == '#') continue;
        try {
            json conv = json::parse(line);
            if (conv.contains("messages") && conv["messages"].is_array()) {
                conversations.push_back(conv);
            }
        } catch (const json::exception & e) {
            LOG_DBG("Warning: Failed to parse JSON line: %s\n", e.what());
        }
    }

    if (conversations.empty()) {
        LOG_ERR("Error: No valid conversations found\n");
        return nullptr;
    }
    LOG_INF("Loaded %zu conversations\n", conversations.size());

    const int64_t ne_datapoint = llama_n_ctx(ctx);
    if (stride <= 0) stride = ne_datapoint;
    if (stride >  ne_datapoint) stride = ne_datapoint;

    std::vector<std::vector<llama_token>> all_tokenized_data;
    std::vector<std::vector<int32_t>>     all_assistant_masks;

    auto token_count_prefix = [&](const std::string & render, size_t char_count) -> size_t {
        std::string prefix = render.substr(0, char_count);
        auto t = common_tokenize(ctx, prefix, /*add_special=*/false, /*parse_special=*/true);
        return t.size();
    };

    // TODO: The masking logic currently relies on string searching for specific role tags.
    // While chat templates render appropriately, we must parse the rendered string to find
    // the boundaries of the "assistant" role to calculate loss gracefully.
    // Currently, this only supports ChatML (Qwen, etc.) and Gemma formats.
    // A more reliable, model-agnostic method would require common_chat_templates_apply to
    // return token role spans directly, which is not yet supported in common/chat.cpp.
    const std::string START_SYS = "<|im_start|>system\n";
    const std::string START_USR = "<|im_start|>user\n";
    const std::string START_AST = "<|im_start|>assistant\n";
    const std::string END_TAG   = "<|im_end|>";
    const std::string NL        = "\n";

    for (size_t i = 0; i < conversations.size(); ++i) {
        const auto & messages = conversations[i]["messages"];
        if (!messages.is_array() || messages.empty()) continue;

        std::string render;

        if (chat_templates) {
            std::vector<common_chat_msg> chat_msgs;
            chat_msgs.reserve(messages.size());
            for (const auto & msg : messages) {
                if (!msg.contains("role") || !msg.contains("content")) {
                    continue;
                }
                common_chat_msg chat_msg;
                chat_msg.role    = msg["role"].get<std::string>();
                chat_msg.content = msg["content"].get<std::string>();
                chat_msgs.push_back(std::move(chat_msg));
            }

            if (!chat_msgs.empty()) {
                common_chat_templates_inputs inputs;
                inputs.messages = std::move(chat_msgs);
                inputs.add_generation_prompt = false;
                inputs.use_jinja = true;

                inputs.enable_thinking = std::any_of(
                    inputs.messages.begin(), inputs.messages.end(),
                    [](const common_chat_msg & m) { return !m.reasoning_content.empty(); });
                try {
                    render = common_chat_templates_apply(chat_templates.get(), inputs).prompt;

                    size_t last_im_end = render.rfind("<|im_end|>");
                    if (last_im_end != std::string::npos) {
                        size_t end_pos = last_im_end + 10; // length of "<|im_end|>"
                        // Remove any trailing whitespace/newlines after the final <|im_end|>
                        while (end_pos < render.size() && (render[end_pos] == '\n' || render[end_pos] == '\r' || render[end_pos] == ' ')) {
                            end_pos++;
                        }
                        if (end_pos < render.size()) {
                            render = render.substr(0, last_im_end + 10); // Keep only up to </im_end>
                        }
                    }
                } catch (const std::exception & e) {
                    LOG_WRN("Warning: chat template rendering failed for conversation %zu: %s. Falling back to default ChatML rendering.\n",
                           i, e.what());
                }
            }
        }

        if (render.empty()) {
            render.reserve(4096);
            for (const auto & msg : messages) {
                if (!msg.contains("role") || !msg.contains("content")) continue;
                const std::string role    = msg["role"].get<std::string>();
                const std::string content = msg["content"].get<std::string>();

                if (role == "system") {
                    render += START_SYS; render += content; render += END_TAG + NL;
                } else if (role == "user") {
                    render += START_USR; render += content; render += END_TAG + NL;
                } else if (role == "assistant") {
                    render += START_AST; render += content; render += END_TAG + NL;
                }
            }
        }

        if (render.empty()) {
            continue;
        }

        struct Span { size_t lo, hi; };
        std::vector<Span> assistant_spans;

        {
            const bool is_gemma3 = render.find("<start_of_turn>model\n") != std::string::npos;
            const bool is_gemma4 = render.find("<|turn>model\n") != std::string::npos;

            const std::string start_tag = is_gemma4 ? "<|turn>model\n" : is_gemma3 ? "<start_of_turn>model\n" : START_AST;
            const std::string end_tag   = is_gemma4 ? "<turn|>"        : is_gemma3 ? "<end_of_turn>"          : END_TAG;

            size_t from = 0;
            while (true) {
                size_t open = render.find(start_tag, from);
                if (open == std::string::npos) break;

                // Skip past the model-turn header -- supervise content only, not the role header
                size_t lo = open + start_tag.size();
                size_t close = render.find(end_tag, lo);
                if (close == std::string::npos) {
                    assistant_spans.push_back({lo, render.size()});
                    break;
                }

                size_t hi = close + end_tag.size();
                assistant_spans.push_back({lo, std::min(hi, render.size())});

                from = hi;
            }
        }

        if (assistant_spans.empty()) {
            LOG_WRN("Conversation %zu has no assistant spans\n", i);
            continue;
        }

        auto tokens_full = common_tokenize(ctx, render, /*add_special=*/false, /*parse_special=*/true);
        if (tokens_full.empty()) continue;

        std::vector<int32_t> assistant_mask(tokens_full.size(), 0);
        size_t assistant_token_count = 0;

        for (const auto & sp : assistant_spans) {
            size_t t_lo = token_count_prefix(render, sp.lo);
            size_t t_hi = token_count_prefix(render, sp.hi);
            if (t_lo > tokens_full.size()) t_lo = tokens_full.size();
            if (t_hi > tokens_full.size()) t_hi = tokens_full.size();


            for (size_t t = t_lo; t < t_hi; ++t) {
                assistant_mask[t] = 1;
                ++assistant_token_count;
            }
        }

        if (assistant_token_count == 0) {
            LOG_WRN("Warning: Conversation %zu has zero assistant tokens after masking\n", i);
            continue;
        }

        all_tokenized_data.push_back(tokens_full);
        all_assistant_masks.push_back(assistant_mask);
    }

    if (all_tokenized_data.empty()) {
        LOG_ERR("ERROR: No valid training samples generated after processing %zu conversations\n", conversations.size());
        return nullptr;
    }

    llama_token pad_token = llama_vocab_pad(vocab);
    if (pad_token == LLAMA_TOKEN_NULL) {
        pad_token = llama_vocab_eos(vocab);
    }

    // count the samples that fit before allocating the dataset
    int64_t ndata = 0;
    for (size_t i = 0; i < all_tokenized_data.size(); ++i) {
        if ((int64_t) all_tokenized_data[i].size() > ne_datapoint) {
            LOG_WRN("Skipping conversation %zu: too long (%zu tokens > %lld)\n", i, all_tokenized_data[i].size(), (long long) ne_datapoint);
            continue;
        }
        ndata++;
    }

    ggml_opt_dataset_t result = ggml_opt_dataset_init_with_masks(
        GGML_TYPE_I32, GGML_TYPE_I32, GGML_TYPE_I32,
        /*ne_datapoint=*/ne_datapoint, /*ne_label=*/ne_datapoint, /*ne_mask=*/ne_datapoint,
        /*ndata=*/ndata, /*ndata_shard=*/1);

    if (result == nullptr) {
        return nullptr;
    }

    int32_t * data   = (int32_t *) ggml_opt_dataset_data(result)->data;
    int32_t * labels = (int32_t *) ggml_opt_dataset_labels(result)->data;
    int32_t * masks  = (int32_t *) ggml_opt_dataset_masks(result)->data;

    int64_t idata = 0;
    for (size_t i = 0; i < all_tokenized_data.size(); ++i) {
        const auto & conv_tokens = all_tokenized_data[i];
        const auto & conv_mask   = all_assistant_masks[i];

        if ((int64_t) conv_tokens.size() > ne_datapoint) {
            continue;
        }

        int32_t * data_row   = data   + idata * ne_datapoint;
        int32_t * labels_row = labels + idata * ne_datapoint;
        int32_t * masks_row  = masks  + idata * ne_datapoint;

        const int64_t n_tokens = conv_tokens.size();

        // inputs, padded to the full context
        for (int64_t t = 0; t < ne_datapoint; ++t) {
            data_row[t] = t < n_tokens ? conv_tokens[t] : pad_token;
        }

        // labels: set actual next tokens for ALL positions (masked cross-entropy needs real tokens)
        for (int64_t t = 0; t < ne_datapoint - 1; ++t) {
            labels_row[t] = data_row[t + 1];
        }
        labels_row[ne_datapoint - 1] = data_row[ne_datapoint - 1]; // last token predicts itself (will be masked)

        // masks: indicate which preds should be trained on (shifted by 1 from conv_mask)
        // Since we predict token[t+1] from token[t], we train when token[t+1] is assistant;
        // padding tokens are never trained on
        for (int64_t t = 0; t < ne_datapoint - 1; ++t) {
            masks_row[t] = (t + 1 < n_tokens && conv_mask[t + 1] == 1) ? 1 : 0;
        }
        masks_row[ne_datapoint - 1] = 0;

        idata++;
    }

    return result;
}
