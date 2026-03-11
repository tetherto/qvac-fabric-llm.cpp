#import "FinetuneBridge.h"

#import <llama/llama.h>
#import <llama/ggml-opt.h>
#import <llama/ggml-backend.h>
#import <llama/ggml.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

#if defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
constexpr bool kIsAppleMobileGPU = true;
#else
constexpr bool kIsAppleMobileGPU = false;
#endif

struct Hms {
    int64_t hours;
    int64_t minutes;
    int64_t seconds;
};

Hms microseconds_to_hms(int64_t microseconds) {
    if (microseconds < 0) {
        microseconds = 0;
    }

    int64_t total_seconds = microseconds / 1000000;
    Hms result{};
    result.hours = total_seconds / 3600;
    total_seconds -= result.hours * 3600;
    result.minutes = total_seconds / 60;
    total_seconds -= result.minutes * 60;
    result.seconds = total_seconds;
    return result;
}

struct Logger {
    llama_swift_finetune_log_callback callback;
    void * user;

    void log(const std::string & message) const {
        if (callback) {
            callback(message.c_str(), user);
        }
    }

    void logf(const char * fmt, ...) const {
        if (!callback) {
            return;
        }

        va_list args;
        va_start(args, fmt);
        std::array<char, 1024> stack_buffer;
        int written = vsnprintf(stack_buffer.data(), stack_buffer.size(), fmt, args);
        va_end(args);

        if (written < 0) {
            return;
        }

        if (written < static_cast<int>(stack_buffer.size())) {
            callback(stack_buffer.data(), user);
            return;
        }

        std::vector<char> heap_buffer(static_cast<size_t>(written + 1));
        va_start(args, fmt);
        vsnprintf(heap_buffer.data(), heap_buffer.size(), fmt, args);
        va_end(args);

        callback(heap_buffer.data(), user);
    }
};

struct checkpoint_metadata {
    int32_t epoch;
    int32_t lora_rank;
    float lora_alpha;
    uint32_t target_modules;
};

struct checkpoint_callback_data {
    llama_context* ctx;
    llama_adapter_lora* adapter;
    int32_t checkpoint_save_steps;
    std::string checkpoint_save_dir;
    int64_t global_step;
    int64_t initial_step;
    int32_t current_epoch;
    int32_t lora_rank;
    float lora_alpha;
    uint32_t target_modules;
    const Logger* logger;
};

static checkpoint_callback_data* g_checkpoint_data = nullptr;

static std::string get_checkpoint_filename(const std::string& checkpoint_dir, int64_t step) {
    std::ostringstream oss;
    oss << checkpoint_dir << "/checkpoint_step_" << std::setfill('0') << std::setw(8) << step;
    return oss.str();
}

static std::string find_latest_checkpoint(const std::string& checkpoint_dir) {
    if (!std::filesystem::exists(checkpoint_dir)) {
        return "";
    }

    std::string latest_checkpoint;
    int64_t latest_step = -1;

    for (const auto& entry : std::filesystem::directory_iterator(checkpoint_dir)) {
        if (entry.is_directory()) {
            std::string dirname = entry.path().filename().string();
            if (dirname.find("checkpoint_step_") == 0 && dirname.size() >= 16) {
                std::string step_str = dirname.substr(16, 8);
                try {
                    int64_t step = std::stoll(step_str);
                    if (step > latest_step) {
                        latest_step = step;
                        latest_checkpoint = entry.path().string();
                    }
                } catch (const std::exception&) {
                    continue;
                }
            }
        }
    }

    return latest_checkpoint;
}

static bool save_checkpoint(
        llama_context* ctx,
        llama_adapter_lora* adapter,
        const checkpoint_metadata& meta,
        const std::string& checkpoint_dir,
        const Logger& logger) {

    try {
        std::filesystem::create_directories(checkpoint_dir);
    } catch (const std::exception& e) {
        logger.logf("Failed to create checkpoint directory: %s\n", e.what());
        return false;
    }

    std::string adapter_path = checkpoint_dir + "/model.gguf";
    if (!llama_lora_save_adapter(adapter, adapter_path.c_str(), llama_get_model(ctx))) {
        logger.logf("Failed to save LoRA adapter to %s\n", adapter_path.c_str());
        return false;
    }

    std::string optimizer_path = checkpoint_dir + "/optimizer.gguf";
    if (!llama_opt_save_state(ctx, optimizer_path.c_str())) {
        logger.logf("Failed to save optimizer state to %s\n", optimizer_path.c_str());
        return false;
    }

    std::string meta_path = checkpoint_dir + "/metadata.txt";
    std::ofstream meta_file(meta_path);
    if (meta_file.is_open()) {
        meta_file << "epoch=" << meta.epoch << "\n";
        meta_file << "lora_rank=" << meta.lora_rank << "\n";
        meta_file << "lora_alpha=" << meta.lora_alpha << "\n";
        meta_file << "target_modules=" << meta.target_modules << "\n";
        meta_file.close();
    } else {
        logger.logf("Failed to save checkpoint metadata to %s\n", meta_path.c_str());
        return false;
    }

    logger.logf("Checkpoint saved to %s\n", checkpoint_dir.c_str());
    return true;
}

static bool load_checkpoint_metadata(const std::string& checkpoint_dir, checkpoint_metadata& metadata, const Logger& logger) {
    std::string meta_path = checkpoint_dir + "/metadata.txt";

    if (std::filesystem::exists(meta_path)) {
        std::ifstream meta_file(meta_path);
        if (meta_file.is_open()) {
            std::string line;
            while (std::getline(meta_file, line)) {
                size_t eq_pos = line.find('=');
                if (eq_pos != std::string::npos) {
                    std::string key = line.substr(0, eq_pos);
                    std::string value = line.substr(eq_pos + 1);

                    if (key == "epoch") {
                        metadata.epoch = std::stoi(value);
                    } else if (key == "lora_rank") {
                        metadata.lora_rank = std::stoi(value);
                    } else if (key == "lora_alpha") {
                        metadata.lora_alpha = std::stof(value);
                    } else if (key == "target_modules") {
                        metadata.target_modules = std::stoul(value);
                    }
                }
            }
            meta_file.close();
            logger.logf("Checkpoint metadata loaded successfully\n");
            return true;
        } else {
            logger.logf("Failed to open checkpoint metadata file\n");
            return false;
        }
    } else {
        logger.logf("Checkpoint metadata file not found: %s\n", meta_path.c_str());
        return false;
    }
}

std::optional<std::string> read_file(const char * path, std::string & error_message) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        error_message = "Unable to open dataset file";
        return std::nullopt;
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    if (stream.fail() && !stream.eof()) {
        error_message = "Failed reading dataset file";
        return std::nullopt;
    }

    return buffer.str();
}

bool tokenize_dataset(const llama_model * model, const std::string & text, std::vector<llama_token> & tokens, std::string & error_message) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) {
        error_message = "Model vocabulary unavailable";
        return false;
    }

    const int32_t text_len = static_cast<int32_t>(text.size());
    int32_t capacity = text_len + 8;
    capacity = std::max<int32_t>(capacity, 8);

    tokens.resize(capacity);
    int32_t n_tokens = llama_tokenize(vocab, text.c_str(), text_len, tokens.data(), static_cast<int32_t>(tokens.size()), /*add_bos=*/true, /*parse_special=*/false);
    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        error_message = "Dataset too large to tokenize";
        return false;
    }
    if (n_tokens < 0) {
        tokens.resize(static_cast<size_t>(-n_tokens));
        n_tokens = llama_tokenize(vocab, text.c_str(), text_len, tokens.data(), static_cast<int32_t>(tokens.size()), /*add_bos=*/true, /*parse_special=*/false);
    } else {
        tokens.resize(static_cast<size_t>(n_tokens));
    }

    if (n_tokens <= 0) {
        error_message = "Dataset produced no tokens";
        return false;
    }

    return true;
}

int32_t default_thread_count() {
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        return 4;
    }
    if (hw <= 2) {
        return 1;
    }
    return static_cast<int32_t>(hw - 1);
}

struct EpochLoggerState {
    const Logger * logger = nullptr;
    int32_t epoch_index = 0;
    int32_t epoch_total = 0;
    int64_t last_logged_train = -1;
    int64_t last_logged_eval = -1;
};

static thread_local EpochLoggerState g_epoch_logger_state;

struct EpochLoggerScope {
    EpochLoggerScope(const Logger & logger, int32_t epoch_index, int32_t epoch_total) {
        g_epoch_logger_state.logger = &logger;
        g_epoch_logger_state.epoch_index = epoch_index;
        g_epoch_logger_state.epoch_total = epoch_total;
        g_epoch_logger_state.last_logged_train = -1;
        g_epoch_logger_state.last_logged_eval = -1;
    }

    ~EpochLoggerScope() {
        g_epoch_logger_state.logger = nullptr;
    }
};

static void finetune_epoch_progress_callback(
        bool train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t,
        ggml_opt_result_t result,
        int64_t ibatch,
        int64_t ibatch_max,
        int64_t t_start_us) {

    EpochLoggerState & state = g_epoch_logger_state;
    if (!state.logger || ibatch_max <= 0) {
        return;
    }

    int64_t & last_logged = train ? state.last_logged_train : state.last_logged_eval;
    if (ibatch <= last_logged) {
        return;
    }
    last_logged = ibatch;

    const int64_t total_batches = std::max<int64_t>(ibatch_max, int64_t(1));
    const int64_t clamped_batch = std::clamp<int64_t>(ibatch, int64_t(1), total_batches);

    const ggml_tensor * input_tensor = ggml_opt_inputs(opt_ctx);
    const int64_t batch_size = input_tensor ? input_tensor->ne[1] : 0;
    const int64_t data_processed = batch_size > 0 ? clamped_batch * batch_size : clamped_batch;
    const int64_t data_total = batch_size > 0 ? total_batches * batch_size : total_batches;

    double loss = 0.0;
    double loss_unc = 0.0;
    ggml_opt_result_loss(result, &loss, &loss_unc);
    if (!std::isfinite(loss)) {
        loss = 0.0;
    }
    if (!std::isfinite(loss_unc)) {
        loss_unc = 0.0;
    }

    double accuracy = 0.0;
    double accuracy_unc = 0.0;
    ggml_opt_result_accuracy(result, &accuracy, &accuracy_unc);
    if (!std::isfinite(accuracy)) {
        accuracy = 0.0;
    }
    if (!std::isfinite(accuracy_unc)) {
        accuracy_unc = 0.0;
    }

    int64_t elapsed_us = 0;
    if (t_start_us > 0) {
        const int64_t now_us = ggml_time_us();
        if (now_us >= t_start_us) {
            elapsed_us = now_us - t_start_us;
        }
    }

    int64_t eta_us = 0;
    if (clamped_batch > 0 && total_batches > clamped_batch) {
        eta_us = (elapsed_us * (total_batches - clamped_batch)) / clamped_batch;
    }

    const Hms elapsed_hms = microseconds_to_hms(elapsed_us);
    const Hms eta_hms = microseconds_to_hms(eta_us);

    const double progress = (static_cast<double>(clamped_batch) * 100.0) / static_cast<double>(total_batches);

    state.logger->logf(
        "    [epoch %d/%d][%s] step %lld/%lld data=%lld/%lld loss=%.6f±%.6f acc=%.2f±%.2f%% "
        "t=%02lld:%02lld:%02lld ETA=%02lld:%02lld:%02lld (%.1f%%)\n",
        state.epoch_index + 1,
        state.epoch_total,
        train ? "train" : "eval",
        static_cast<long long>(clamped_batch),
        static_cast<long long>(total_batches),
        static_cast<long long>(data_processed),
        static_cast<long long>(data_total),
        loss,
        loss_unc,
        100.0 * accuracy,
        100.0 * accuracy_unc,
        static_cast<long long>(elapsed_hms.hours),
        static_cast<long long>(elapsed_hms.minutes),
        static_cast<long long>(elapsed_hms.seconds),
        static_cast<long long>(eta_hms.hours),
        static_cast<long long>(eta_hms.minutes),
        static_cast<long long>(eta_hms.seconds),
        progress);
}

static void checkpoint_progress_callback(
        bool train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t result,
        int64_t ibatch,
        int64_t ibatch_max,
        int64_t t_start_us) {

    finetune_epoch_progress_callback(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);

    if (!train) {
        return;
    }

    checkpoint_callback_data* cb_data = g_checkpoint_data;
    if (!cb_data) {
        return;
    }

    if (cb_data->checkpoint_save_steps <= 0) {
        return;
    }

    cb_data->global_step++;

    if (cb_data->global_step % cb_data->checkpoint_save_steps == 0) {
        if (!cb_data->ctx) {
            if (cb_data->logger) {
                cb_data->logger->logf("ERROR: Context is null in checkpoint callback!\n");
            }
            return;
        }

        if (!cb_data->adapter) {
            if (cb_data->logger) {
                cb_data->logger->logf("ERROR: LoRA adapter is null in checkpoint callback!\n");
            }
            return;
        }

        checkpoint_metadata meta = {
            /*epoch          =*/ cb_data->current_epoch,
            /*lora_rank      =*/ cb_data->lora_rank,
            /*lora_alpha     =*/ cb_data->lora_alpha,
            /*target_modules =*/ cb_data->target_modules,
        };

        std::string checkpoint_path = get_checkpoint_filename(cb_data->checkpoint_save_dir, cb_data->global_step);

        if (cb_data->logger) {
            cb_data->logger->logf("Saving checkpoint at step %lld to %s\n",
                (long long)cb_data->global_step, checkpoint_path.c_str());
        }

        if (!save_checkpoint(cb_data->ctx, cb_data->adapter, meta, checkpoint_path, *cb_data->logger)) {
            if (cb_data->logger) {
                cb_data->logger->logf("ERROR: Failed to save checkpoint at step %lld\n",
                    (long long)cb_data->global_step);
            }
        }
    }
}

} // namespace

extern "C" enum llama_swift_finetune_error llama_swift_run_lora_finetune(
    const char * model_path,
    const char * dataset_path,
    const char * output_adapter_path,
    const struct llama_swift_finetune_options * options,
    llama_swift_finetune_log_callback logger_callback,
    void * user_data) {

    Logger logger{logger_callback, user_data};

    if (!model_path || !dataset_path || !output_adapter_path) {
        logger.log("Invalid arguments supplied to finetune request\n");
        return LLAMA_SWIFT_FINETUNE_ERROR_INVALID_ARGUMENT;
    }

    llama_swift_finetune_options opts{};
    if (options) {
        opts = *options;
    }

    if (opts.n_ctx <= 0) {
        opts.n_ctx = 256;
    }
    if (opts.n_threads <= 0) {
        opts.n_threads = default_thread_count();
    }
    if (opts.n_batch <= 0) {
        opts.n_batch = 256;
    }
    if (opts.n_ubatch <= 0) {
        opts.n_ubatch = opts.n_batch;
    }
    if (opts.epochs <= 0) {
        opts.epochs = 2;
    }
    if (opts.lora_rank <= 0) {
        opts.lora_rank = 8;
    }
    if (opts.lora_alpha <= 0.0f) {
        opts.lora_alpha = 16.0f;
    }
    if (opts.learning_rate <= 0.0f) {
        opts.learning_rate = 1e-5f;
    }
    if (opts.val_split < 0.0f) {
        opts.val_split = 0.0f;
    }
    if (opts.val_split >= 1.0f) {
        opts.val_split = 0.0f;
    }
    if (opts.target_modules == 0) {
        opts.target_modules = LLAMA_LORA_TARGET_ATTN_Q |
                              LLAMA_LORA_TARGET_ATTN_K |
                              LLAMA_LORA_TARGET_ATTN_V |
                              LLAMA_LORA_TARGET_ATTN_O;
    }
    if (opts.n_gpu_layers < 0) {
        const char * env_ngl = std::getenv("LLAMA_SWIFT_FINETUNE_NGL");
        if (env_ngl && *env_ngl) {
            char * endptr = nullptr;
            long parsed = std::strtol(env_ngl, &endptr, 10);
            if (endptr != env_ngl && parsed >= 0 && parsed <= std::numeric_limits<int32_t>::max()) {
                opts.n_gpu_layers = static_cast<int32_t>(parsed);
                logger.logf("Environment override: using n_gpu_layers=%d\n", opts.n_gpu_layers);
            } else {
                logger.logf("Ignoring invalid LLAMA_SWIFT_FINETUNE_NGL value '%s'\n", env_ngl);
                opts.n_gpu_layers = 999;
            }
        } else {
            opts.n_gpu_layers = 999;
        }
    }

    // opts.n_gpu_layers = 0;

    const bool gpu_available = opts.n_gpu_layers > 0 && llama_supports_gpu_offload();
    if (gpu_available) {
        const int32_t max_gpu_batch = kIsAppleMobileGPU ? 64 : 256;
        const int32_t max_gpu_ubatch = kIsAppleMobileGPU ? 32 : 128;

        if (opts.n_batch > max_gpu_batch) {
            logger.logf("Clamping batch size from %d to %d for on-device GPU stability\n", opts.n_batch, max_gpu_batch);
            opts.n_batch = max_gpu_batch;
        }

        if (opts.n_ubatch > max_gpu_ubatch) {
            logger.logf("Clamping micro-batch size from %d to %d for on-device GPU stability\n", opts.n_ubatch, max_gpu_ubatch);
            opts.n_ubatch = max_gpu_ubatch;
        }

        if (opts.n_ubatch > opts.n_batch) {
            logger.logf("Adjusting micro-batch size to not exceed batch size (%d -> %d)\n", opts.n_ubatch, opts.n_batch);
            opts.n_ubatch = opts.n_batch;
        }
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = opts.n_gpu_layers;
    model_params.use_mmap = false;
    model_params.use_mlock = false;

    logger.logf("Loading model from %s\n", model_path);
    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        logger.log("Failed to load model for finetuning\n");
        return LLAMA_SWIFT_FINETUNE_ERROR_MODEL_LOAD;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = opts.n_ctx;
    ctx_params.n_batch = opts.n_batch;
    ctx_params.n_ubatch = opts.n_ubatch;
    ctx_params.n_threads = opts.n_threads;
    ctx_params.n_threads_batch = opts.n_threads;
    ctx_params.flash_attn_type = opts.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    if (opts.seed != 0) {
        logger.log("Warning: custom RNG seed requested but not supported by bundled llama.framework.\n");
    }

    logger.logf("Creating training context (n_ctx=%d, n_threads=%d, batch=%d, ubatch=%d, ngl=%d)\n",
                ctx_params.n_ctx, ctx_params.n_threads, ctx_params.n_batch, ctx_params.n_ubatch, opts.n_gpu_layers);

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        logger.log("Failed to create llama context for finetuning\n");
        llama_model_free(model);
        return LLAMA_SWIFT_FINETUNE_ERROR_CONTEXT_CREATE;
    }

    std::string error_message;
    auto dataset_text = read_file(dataset_path, error_message);
    if (!dataset_text.has_value()) {
        logger.logf("%s: %s\n", error_message.c_str(), dataset_path);
        llama_free(ctx);
        llama_model_free(model);
        return LLAMA_SWIFT_FINETUNE_ERROR_DATASET;
    }

    std::vector<llama_token> tokens;
    if (!tokenize_dataset(model, dataset_text.value(), tokens, error_message)) {
        logger.logf("Tokenization failed: %s\n", error_message.c_str());
        llama_free(ctx);
        llama_model_free(model);
        return LLAMA_SWIFT_FINETUNE_ERROR_DATASET;
    }

    const int64_t ne_datapoint = llama_n_ctx(ctx);
    const int64_t stride = std::max<int64_t>(1, ne_datapoint / 2);
    if (tokens.size() < static_cast<size_t>(ne_datapoint + 1)) {
        logger.log("Dataset is too short for the selected context size\n");
        llama_free(ctx);
        llama_model_free(model);
        return LLAMA_SWIFT_FINETUNE_ERROR_DATASET;
    }

    const int64_t ndata = (static_cast<int64_t>(tokens.size()) - ne_datapoint - 1) / stride;
    if (ndata <= 0) {
        logger.log("Tokenized dataset produced no training samples\n");
        llama_free(ctx);
        llama_model_free(model);
        return LLAMA_SWIFT_FINETUNE_ERROR_DATASET;
    }

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
        GGML_TYPE_I32,
        GGML_TYPE_I32,
        ne_datapoint,
        ne_datapoint,
        ndata,
        /*ndata_shard=*/1);

    llama_token * data_ptr = reinterpret_cast<llama_token *>(ggml_opt_dataset_data(dataset)->data);
    llama_token * labels_ptr = reinterpret_cast<llama_token *>(ggml_opt_dataset_labels(dataset)->data);

    for (int64_t i = 0; i < ndata; ++i) {
        std::memcpy(data_ptr + i * ne_datapoint, tokens.data() + i * stride, static_cast<size_t>(ne_datapoint) * sizeof(llama_token));
        std::memcpy(labels_ptr + i * ne_datapoint, tokens.data() + i * stride + 1, static_cast<size_t>(ne_datapoint) * sizeof(llama_token));
    }

    logger.logf("Prepared dataset with %lld sequences (stride=%lld)\n", static_cast<long long>(ndata), static_cast<long long>(stride));

    llama_lora_training_params lora_params{
        /*target_modules =*/ opts.target_modules,
        /*rank           =*/ opts.lora_rank,
        /*alpha          =*/ opts.lora_alpha,
        /*dropout        =*/ 0.0f,
        /*init_std       =*/ 0.02f,
    };

    llama_adapter_lora * adapter = llama_lora_training_init(ctx, model, &lora_params);
    if (!adapter) {
        logger.log("Failed to initialize LoRA training\n");
        ggml_opt_dataset_free(dataset);
        llama_free(ctx);
        llama_model_free(model);
        return LLAMA_SWIFT_FINETUNE_ERROR_TRAINING_INIT;
    }

    bool checkpoint_loaded = false;
    int32_t start_epoch = 0;
    int64_t start_step = 0;
    std::string checkpoint_dir = opts.checkpoint_save_dir ? opts.checkpoint_save_dir : "./checkpoints";
    std::string resume_from_checkpoint;

    if (opts.auto_resume) {
        logger.logf("Auto-resume enabled, searching for latest checkpoint in %s\n", checkpoint_dir.c_str());
        resume_from_checkpoint = find_latest_checkpoint(checkpoint_dir);
        if (!resume_from_checkpoint.empty()) {
            logger.logf("Found latest checkpoint: %s\n", resume_from_checkpoint.c_str());
        } else {
            logger.logf("No existing checkpoint found, starting from scratch\n");
        }
    } else if (opts.resume_from_checkpoint && opts.resume_from_checkpoint[0] != '\0') {
        resume_from_checkpoint = opts.resume_from_checkpoint;
        logger.logf("Resuming from specified checkpoint: %s\n", resume_from_checkpoint.c_str());
    }

    if (!resume_from_checkpoint.empty()) {
        checkpoint_metadata checkpoint_meta{};
        if (load_checkpoint_metadata(resume_from_checkpoint, checkpoint_meta, logger)) {
            if (checkpoint_meta.lora_rank != opts.lora_rank) {
                logger.logf("Warning: Checkpoint LoRA rank (%d) doesn't match current rank (%d)\n",
                    checkpoint_meta.lora_rank, opts.lora_rank);
            }
            if (checkpoint_meta.lora_alpha != opts.lora_alpha) {
                logger.logf("Warning: Checkpoint LoRA alpha (%.3f) doesn't match current alpha (%.3f)\n",
                    checkpoint_meta.lora_alpha, opts.lora_alpha);
            }
            if (checkpoint_meta.target_modules != opts.target_modules) {
                logger.logf("Warning: Checkpoint target_modules doesn't match current target_modules\n");
            }

            start_epoch = checkpoint_meta.epoch;
            checkpoint_loaded = true;
        } else {
            logger.logf("Failed to load checkpoint metadata, starting from scratch\n");
        }
    }

    std::string optimizer_checkpoint_path;
    if (checkpoint_loaded && !resume_from_checkpoint.empty()) {
        optimizer_checkpoint_path = resume_from_checkpoint + "/optimizer.gguf";
    }

    ggml_opt_optimizer_params optimizer_params = ggml_opt_get_default_optimizer_params(nullptr);
    optimizer_params.adamw.alpha = opts.learning_rate;

    llama_opt_params opt_params{
        /*n_ctx_train            =*/ 0,
        /*param_filter           =*/ llama_opt_param_filter_lora,
        /*param_filter_ud        =*/ nullptr,
        /*get_opt_pars           =*/ ggml_opt_get_constant_optimizer_params,
        /*get_opt_pars_ud        =*/ &optimizer_params,
        /*optimizer_type         =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
        /*checkpoint_path        =*/ checkpoint_loaded ? optimizer_checkpoint_path.c_str() : nullptr,
        /*load_optimizer_state   =*/ checkpoint_loaded,
        /*assistant_loss_only    =*/ false,
    };

    llama_opt_init(ctx, model, opt_params);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    const int64_t total_samples = ggml_opt_dataset_ndata(dataset);
    int64_t idata_split = total_samples;
    if (opts.val_split > 0.0f && opts.val_split < 1.0f) {
        const double train_fraction = std::clamp(1.0 - static_cast<double>(opts.val_split), 0.0, 1.0);
        int64_t proposed = static_cast<int64_t>(std::llround(train_fraction * static_cast<double>(total_samples)));
        proposed = std::clamp<int64_t>(proposed, 1, std::max<int64_t>(total_samples - 1, 1));
        idata_split = proposed;
    }

    checkpoint_callback_data cb_data{};
    cb_data.ctx = ctx;
    cb_data.adapter = adapter;
    cb_data.checkpoint_save_steps = opts.checkpoint_save_steps;
    cb_data.checkpoint_save_dir = checkpoint_dir;
    cb_data.global_step = start_step;
    cb_data.initial_step = start_step;
    cb_data.current_epoch = start_epoch;
    cb_data.lora_rank = opts.lora_rank;
    cb_data.lora_alpha = opts.lora_alpha;
    cb_data.target_modules = opts.target_modules;
    cb_data.logger = &logger;

    g_checkpoint_data = &cb_data;

    ggml_opt_epoch_callback train_callback = opts.checkpoint_save_steps > 0
        ? checkpoint_progress_callback
        : finetune_epoch_progress_callback;

    ggml_opt_epoch_callback eval_callback = (idata_split < total_samples)
        ? finetune_epoch_progress_callback
        : nullptr;

    if (opts.checkpoint_save_steps > 0) {
        logger.logf("Checkpointing enabled, saving every %d steps to %s\n",
            opts.checkpoint_save_steps, checkpoint_dir.c_str());
    } else {
        logger.logf("Checkpointing disabled\n");
    }

    logger.logf("Starting LoRA finetuning for %d epoch(s)\n", opts.epochs);

    for (int32_t epoch = start_epoch; epoch < opts.epochs; ++epoch) {
        logger.logf("Epoch %d/%d\n", epoch + 1, opts.epochs);
        EpochLoggerScope epoch_scope(logger, epoch, opts.epochs);

        cb_data.current_epoch = epoch;

        int64_t resume_batch = 0;

        llama_opt_epoch_resume(ctx,
                        dataset,
                        result_train,
                        result_eval,
                        idata_split,
                        train_callback,
                        eval_callback,
                        resume_batch);

        double train_loss = 0.0;
        double train_unc = 0.0;
        ggml_opt_result_loss(result_train, &train_loss, &train_unc);
        ggml_opt_result_reset(result_train);

        if (idata_split < total_samples) {
            double val_loss = 0.0;
            double val_unc = 0.0;
            ggml_opt_result_loss(result_eval, &val_loss, &val_unc);
            ggml_opt_result_reset(result_eval);
            logger.logf("  train loss: %.6f  val loss: %.6f\n", train_loss, val_loss);
        } else {
            logger.logf("  train loss: %.6f\n", train_loss);
        }
    }

    g_checkpoint_data = nullptr;

    std::filesystem::path output_path(output_adapter_path);
    if (output_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(output_path.parent_path(), ec);
        if (ec) {
            logger.logf("Failed to create output directory: %s (%d)\n", ec.message().c_str(), static_cast<int>(ec.value()));
            ggml_opt_result_free(result_train);
            ggml_opt_result_free(result_eval);
            ggml_opt_dataset_free(dataset);
            llama_adapter_lora_free(adapter);
            llama_free(ctx);
            llama_model_free(model);
            return LLAMA_SWIFT_FINETUNE_ERROR_SAVE;
        }
    }

    logger.logf("Saving LoRA adapter to %s\n", output_adapter_path);
    if (!llama_lora_save_adapter(adapter, output_adapter_path, model)) {
        logger.log("Failed to save LoRA adapter\n");
        ggml_opt_result_free(result_train);
        ggml_opt_result_free(result_eval);
        ggml_opt_dataset_free(dataset);
        llama_adapter_lora_free(adapter);
        llama_free(ctx);
        llama_model_free(model);
        return LLAMA_SWIFT_FINETUNE_ERROR_SAVE;
    }

    std::error_code size_ec;
    const auto file_size = std::filesystem::file_size(output_path, size_ec);
    if (!size_ec) {
        logger.logf("Saved adapter size: %.2f MB\n", static_cast<double>(file_size) / (1024.0 * 1024.0));
    }

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);
    ggml_opt_dataset_free(dataset);
    llama_adapter_lora_free(adapter);
    llama_free(ctx);
    llama_model_free(model);

    logger.log("LoRA finetuning completed successfully\n");
    return LLAMA_SWIFT_FINETUNE_OK;
}
