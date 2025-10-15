#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cstring>
#include <vector>
#include <fstream>
#include <filesystem>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif


struct checkpoint_callback_data;
static checkpoint_callback_data* g_checkpoint_data = nullptr;

static uint32_t parse_lora_modules(const std::string& modules_str) {
    if (modules_str.empty()) {
        return LLAMA_LORA_TARGET_ATTN_Q | LLAMA_LORA_TARGET_ATTN_K | LLAMA_LORA_TARGET_ATTN_V | LLAMA_LORA_TARGET_ATTN_O;
    }

    static const std::map<std::string, uint32_t> module_map = {
        {"attn_q",    LLAMA_LORA_TARGET_ATTN_Q},
        {"attn_k",    LLAMA_LORA_TARGET_ATTN_K},
        {"attn_v",    LLAMA_LORA_TARGET_ATTN_V},
        {"attn_o",    LLAMA_LORA_TARGET_ATTN_O},
        {"ffn_gate",  LLAMA_LORA_TARGET_FFN_GATE},
        {"ffn_up",    LLAMA_LORA_TARGET_FFN_UP},
        {"ffn_down",  LLAMA_LORA_TARGET_FFN_DOWN},
        {"output",    LLAMA_LORA_TARGET_OUTPUT},
        {"all",       LLAMA_LORA_TARGET_ALL}
    };

    uint32_t target_modules = 0;
    std::stringstream ss(modules_str);
    std::string module;

    while (std::getline(ss, module, ',')) {
        module.erase(0, module.find_first_not_of(" \t"));
        module.erase(module.find_last_not_of(" \t") + 1);

        auto it = module_map.find(module);
        if (it != module_map.end()) {
            target_modules |= it->second;
            LOG_INF("Added target module: %s\n", module.c_str());
        } else {
            LOG_ERR("Unknown LoRA target module: %s\n", module.c_str());
            LOG_ERR("Available modules: attn_q, attn_k, attn_v, attn_o, ffn_gate, ffn_up, ffn_down, output, all\n");
            return 0;
        }
    }

    return target_modules;
}

static bool training_supports_out_prod_f16(const common_params & params) {
    std::vector<ggml_backend_dev_t> devices;

    if (!params.devices.empty()) {
        devices.assign(params.devices.begin(), params.devices.end());
    } else {
        ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (gpu) {
            devices.push_back(gpu);
        }
    }

    if (devices.empty()) {
        return true;
    }

    constexpr int64_t ne0 = 4;
    constexpr int64_t ne1 = 3;
    constexpr int64_t k   = 2;

    struct ggml_tensor src0 = {};
    struct ggml_tensor src1 = {};
    struct ggml_tensor dst  = {};

    src0.type = GGML_TYPE_F16;
    src1.type = GGML_TYPE_F32;
    dst.type  = GGML_TYPE_F32;

    src0.ne[0] = ne0; src0.ne[1] = k;   src0.ne[2] = 1; src0.ne[3] = 1;
    src1.ne[0] = ne1; src1.ne[1] = k;   src1.ne[2] = 1; src1.ne[3] = 1;
    dst.ne [0] = ne0; dst.ne [1] = ne1; dst.ne [2] = 1; dst.ne [3] = 1;

    src0.nb[0] = sizeof(ggml_fp16_t);
    src0.nb[1] = src0.nb[0] * ne0;
    src0.nb[2] = src0.nb[1] * k;
    src0.nb[3] = src0.nb[2] * 1;

    src1.nb[0] = sizeof(float);
    src1.nb[1] = src1.nb[0] * ne1;
    src1.nb[2] = src1.nb[1] * k;
    src1.nb[3] = src1.nb[2] * 1;

    dst.nb[0] = sizeof(float);
    dst.nb[1] = dst.nb[0] * ne0;
    dst.nb[2] = dst.nb[1] * ne1;
    dst.nb[3] = dst.nb[2] * 1;

    dst.op     = GGML_OP_OUT_PROD;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    for (ggml_backend_dev_t dev : devices) {
        if (dev == nullptr) {
            continue;
        }
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        if (!ggml_backend_dev_supports_op(dev, &dst)) {
            return false;
        }
    }

    return true;
}

static void print_lora_usage() {
    printf("\n----- LoRA Fine-tuning Parameters -----\n");
    printf("  --lora-rank N              LoRA rank (default: 8, range: 1-512)\n");
    printf("  --lora-alpha N             LoRA alpha scaling factor (default: 16.0, range: 0.1-1000.0)\n");
    printf("  --lora-modules MODULES     Target modules as comma-separated list (default: attn_q,attn_k,attn_v,attn_o)\n");
    printf("                             Available modules: attn_q, attn_k, attn_v, attn_o, ffn_gate, ffn_up, ffn_down, output, all\n");
    printf("                             Examples: \"attn_q,attn_v\" or \"all\" or \"attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down\"\n");
    printf("  --output-adapter PATH      Output path for trained adapter (default: auto-generated)\n");
    printf("\nTraining Options:\n");
    printf("  --num-epochs N             Number of training epochs (default: 1)\n");
    printf("\nCheckpointing Options:\n");
    printf("  --checkpoint-save-steps N  Save checkpoint every N training steps (default: 100)\n");
    printf("  --checkpoint-save-dir PATH Directory for checkpoints (default: ./checkpoints)\n");
    printf("  --resume-from PATH         Resume training from specific checkpoint file\n");
    printf("  --auto-resume              Automatically resume from latest checkpoint in save dir\n");
    printf("\nExamples:\n");
    printf("  # Train with rank=16, alpha=32, all attention modules\n");
    printf("  %s -m model.gguf -f dataset.txt --lora-rank 16 --lora-alpha 32 --lora-modules attn_q,attn_k,attn_v,attn_o\n", "finetune-lora");
    printf("\n  # Fine-tune existing adapter with all modules\n");
    printf("  %s -m model.gguf -f dataset.txt --lora existing.gguf --output-adapter improved.gguf\n", "finetune-lora");
    printf("\n");
}

struct checkpoint_metadata {
    int32_t epoch;
    int32_t lora_rank;
    float lora_alpha;
    uint32_t target_modules;
};

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

static bool save_checkpoint(llama_context* ctx, llama_adapter_lora* adapter,  const checkpoint_metadata& metadata, const std::string& checkpoint_dir) {
    if (!std::filesystem::exists(checkpoint_dir)) {
        if (!std::filesystem::create_directories(checkpoint_dir)) {
            LOG_ERR("Failed to create checkpoint directory: %s\n", checkpoint_dir.c_str());
            return false;
        }
    }
    
    if (!llama_lora_save_checkpoint(adapter, checkpoint_dir.c_str(), llama_get_model(ctx), ctx)) {
        LOG_ERR("Failed to save LoRA checkpoint\n");
        return false;
    }
    
    std::string meta_path = checkpoint_dir + "/metadata.json";
    std::ofstream meta_file(meta_path);
    if (meta_file.is_open()) {
        meta_file << "epoch=" << metadata.epoch << "\n";
        meta_file << "lora_rank=" << metadata.lora_rank << "\n";
        meta_file << "lora_alpha=" << metadata.lora_alpha << "\n";
        meta_file << "target_modules=" << metadata.target_modules << "\n";
        meta_file.close();
    } else {
        LOG_ERR("Failed to save checkpoint metadata\n");
        return false;
    }
    
    LOG_INF("Checkpoint saved successfully to %s\n", checkpoint_dir.c_str());
    return true;
}

static bool validate_checkpoint_metadata(const std::string& checkpoint_path, checkpoint_metadata& metadata) {
    std::string checkpoint_dir = checkpoint_path;
    
    if (!std::filesystem::exists(checkpoint_dir)) {
        LOG_ERR("Checkpoint directory does not exist: %s\n", checkpoint_dir.c_str());
        return false;
    }
    
    LOG_INF("Loading checkpoint from: %s\n", checkpoint_dir.c_str());
    
    std::string meta_path = checkpoint_dir + "/metadata.json";
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
        } else {
            LOG_ERR("Failed to open checkpoint metadata file\n");
            return false;
        }
    } else {
        LOG_ERR("Checkpoint metadata file not found: %s\n", meta_path.c_str());
        return false;
    }
    
    LOG_INF("Checkpoint loaded successfully\n");
    return true;
}


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
    float learning_rate;
    std::string model_path;
    std::string dataset_path;
};

static void checkpoint_progress_callback(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);
    
    if (!train) return;
    
    checkpoint_callback_data* cb_data = g_checkpoint_data;
    
    if (!cb_data) {
        LOG_ERR("Checkpoint callback data is null!\n");
        return;
    }
    
    if (cb_data->checkpoint_save_steps <= 0) {
        return;
    }
    
    cb_data->global_step++;
    
    if (cb_data->global_step % cb_data->checkpoint_save_steps == 0) {
        if (!cb_data->ctx) {
            LOG_ERR("Context is null in checkpoint callback!\n");
            return;
        }
        
        if (!cb_data->adapter) {
            LOG_ERR("LoRA adapter is null in checkpoint callback!\n");
            return;
        }
        
        checkpoint_metadata meta = {
            /*epoch          =*/ cb_data->current_epoch,
            /*lora_rank      =*/ cb_data->lora_rank,
            /*lora_alpha     =*/ cb_data->lora_alpha,
            /*target_modules =*/ cb_data->target_modules,
        };
        
        std::string checkpoint_path = get_checkpoint_filename(cb_data->checkpoint_save_dir, cb_data->global_step);
        
        if (!save_checkpoint(cb_data->ctx, cb_data->adapter, meta, checkpoint_path)) {
            LOG_ERR("Failed to save checkpoint at step %ld\n", cb_data->global_step);
        }
    }
}

struct finetune_params {
    int32_t lora_rank = 8;
    float lora_alpha = 16.0f;
    std::string lora_modules_str;
    std::string output_adapter_path;
    
    int32_t num_epochs = 1;
    
    int32_t checkpoint_save_steps = 100;
    std::string checkpoint_save_dir = "./checkpoints";
    std::string resume_from_checkpoint;
    bool auto_resume = false;
};

static bool parse_finetune_args(int& argc, char** argv, finetune_params& ft_params) {
    auto remove_arg_pair = [&](int i) {
        for (int j = i; j < argc - 2; j++) {
            argv[j] = argv[j + 2];
        }
        argc -= 2;
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--lora-rank") == 0 && i + 1 < argc) {
            ft_params.lora_rank = std::atoi(argv[i + 1]);
            remove_arg_pair(i);
            i--;
        } else if (strcmp(argv[i], "--lora-alpha") == 0 && i + 1 < argc) {
            ft_params.lora_alpha = std::atof(argv[i + 1]);
            remove_arg_pair(i);
            i--;
        } else if (strcmp(argv[i], "--lora-modules") == 0 && i + 1 < argc) {
            ft_params.lora_modules_str = argv[i + 1];
            remove_arg_pair(i);
            i--;
        } else if (strcmp(argv[i], "--output-adapter") == 0 && i + 1 < argc) {
            ft_params.output_adapter_path = argv[i + 1];
            remove_arg_pair(i);
            i--;
        } else if (strcmp(argv[i], "--num-epochs") == 0 && i + 1 < argc) {
            ft_params.num_epochs = std::atoi(argv[i + 1]);
            remove_arg_pair(i);
            i--;
        } else if (strcmp(argv[i], "--checkpoint-save-steps") == 0 && i + 1 < argc) {
            ft_params.checkpoint_save_steps = std::atoi(argv[i + 1]);
            remove_arg_pair(i);
            i--;
        } else if (strcmp(argv[i], "--checkpoint-save-dir") == 0 && i + 1 < argc) {
            ft_params.checkpoint_save_dir = argv[i + 1];
            remove_arg_pair(i);
            i--;
        } else if (strcmp(argv[i], "--resume-from") == 0 && i + 1 < argc) {
            ft_params.resume_from_checkpoint = argv[i + 1];
            remove_arg_pair(i);
            i--;
        } else if (strcmp(argv[i], "--auto-resume") == 0) {
            ft_params.auto_resume = true;
            for (int j = i; j < argc - 1; j++) {
                argv[j] = argv[j + 1];
            }
            argc--;
            i--;
        }
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_lora_usage();
        }
    }
    
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    finetune_params ft_params;

    params.escape = false;
    parse_finetune_args(argc, argv, ft_params);

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }

    LOG_INF("Using LoRA parameters: rank=%d, alpha=%.1f\n", ft_params.lora_rank, ft_params.lora_alpha);
    LOG_INF("Training for %d epochs\n", ft_params.num_epochs);
    
    // Handle checkpoint auto-resume before model initialization
    if (ft_params.auto_resume && ft_params.resume_from_checkpoint.empty()) {
        std::string latest_checkpoint = find_latest_checkpoint(ft_params.checkpoint_save_dir);
        if (!latest_checkpoint.empty()) {
            ft_params.resume_from_checkpoint = latest_checkpoint;
            LOG_INF("Auto-resume: found checkpoint %s\n", ft_params.resume_from_checkpoint.c_str());
        }
    }
    
    // Load checkpoint LoRA adapter from directory structure (model.gguf)
    if (!ft_params.resume_from_checkpoint.empty()) {
        std::filesystem::path checkpoint_dir(ft_params.resume_from_checkpoint);
        std::filesystem::path model_path = checkpoint_dir / "model.gguf";
        
        LOG_INF("Loading checkpoint LoRA adapter: %s\n", model_path.c_str());
        common_adapter_lora_info lora_adapter;
        lora_adapter.path = model_path.string();
        lora_adapter.scale = 1.0f;
        lora_adapter.ptr = nullptr;
        params.lora_adapters.clear(); // Remove any existing adapters
        params.lora_adapters.push_back(lora_adapter);
        LOG_INF("Checkpoint LoRA adapter added to params\n");
    }

    if (params.use_mmap) {
        LOG_INF("%s: force disabling memory mapping because it would result in-read-only pointers to the weights\n", __func__);
        params.use_mmap = false;
    }
    const bool supports_out_prod_f16 = training_supports_out_prod_f16(params);
    if (!supports_out_prod_f16) {
        if (params.cache_type_k != GGML_TYPE_F32) {
            LOG_INF("%s: force changing k cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
            params.cache_type_k = GGML_TYPE_F32;
        }
        if (params.cache_type_v != GGML_TYPE_F32) {
            LOG_INF("%s: force changing v cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
            params.cache_type_v = GGML_TYPE_F32;
        }
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result llama_init = common_init_from_params(params);
    llama_model_ptr   & model = llama_init.model;
    llama_context_ptr & ctx   = llama_init.context;

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    uint32_t target_modules = parse_lora_modules(ft_params.lora_modules_str);
    if (target_modules == 0) {
        return 1;
    }

    struct llama_lora_training_params lora_params = {
        /*target_modules =*/ target_modules,
        /*rank           =*/ ft_params.lora_rank,
        /*alpha          =*/ ft_params.lora_alpha,
        /*dropout        =*/ 0.0f,
        /*init_std       =*/ 0.02f,
    };

    bool has_existing_lora = !params.lora_adapters.empty();
    struct llama_adapter_lora * trained_adapter = nullptr;

    if (has_existing_lora) {
        LOG_INF("Finetuning existing LoRA adapters\n");
        LOG_INF("Found %zu existing LoRA adapters to train\n", params.lora_adapters.size());\
        trained_adapter = params.lora_adapters[0].ptr;
        if (!trained_adapter) {
            LOG_ERR("Existing LoRA adapter is null\n");
            return 1;
        }
    } else {
        LOG_INF("Target modules: Q=%s, K=%s, V=%s, O=%s, GATE=%s, UP=%s, DOWN=%s, OUTPUT=%s\n",
            (lora_params.target_modules & LLAMA_LORA_TARGET_ATTN_Q) ? "yes" : "no",
            (lora_params.target_modules & LLAMA_LORA_TARGET_ATTN_K) ? "yes" : "no",
            (lora_params.target_modules & LLAMA_LORA_TARGET_ATTN_V) ? "yes" : "no",
            (lora_params.target_modules & LLAMA_LORA_TARGET_ATTN_O) ? "yes" : "no",
            (lora_params.target_modules & LLAMA_LORA_TARGET_FFN_GATE) ? "yes" : "no",
            (lora_params.target_modules & LLAMA_LORA_TARGET_FFN_UP) ? "yes" : "no",
            (lora_params.target_modules & LLAMA_LORA_TARGET_FFN_DOWN) ? "yes" : "no",
            (lora_params.target_modules & LLAMA_LORA_TARGET_OUTPUT) ? "yes" : "no");

        LOG_INF("LoRA configuration: rank=%d, alpha=%.1f (scaling=%.3f)\n",
            lora_params.rank, lora_params.alpha, lora_params.alpha / lora_params.rank);

        trained_adapter = llama_lora_training_init(ctx.get(), model.get(), &lora_params);
        if (!trained_adapter) {
            LOG_ERR("%s: LoRA training initialization failed\n", __func__);
            return 1;
        }
    }

    constexpr float val_split = 0.05f;

    std::vector<llama_token> tokens = common_tokenize(ctx.get(), params.prompt, true);
    ggml_opt_dataset_t dataset = common_opt_dataset_init(ctx.get(), tokens, llama_n_ctx(ctx.get())/2);

    int start_epoch = 0;
    int64_t start_step = 0;
    checkpoint_metadata checkpoint_meta = {};
    bool checkpoint_loaded = false;
    
    if (!ft_params.resume_from_checkpoint.empty()) {
        if (validate_checkpoint_metadata(ft_params.resume_from_checkpoint, checkpoint_meta)) {
            start_epoch = checkpoint_meta.epoch;
            checkpoint_loaded = true;
            
            if (checkpoint_meta.lora_rank != ft_params.lora_rank) {
                LOG_ERR("Checkpoint LoRA rank (%d) doesn't match current rank (%d). Use --resume-from to manually specify a compatible checkpoint.\n", 
                        checkpoint_meta.lora_rank, ft_params.lora_rank);
                return 1;
            }
            if (checkpoint_meta.lora_alpha != ft_params.lora_alpha) {
                LOG_ERR("Checkpoint LoRA alpha (%.3f) doesn't match current alpha (%.3f)\n", 
                        checkpoint_meta.lora_alpha, ft_params.lora_alpha);
                return 1;
            }
            if (checkpoint_meta.target_modules != target_modules) {
                LOG_ERR("Checkpoint target_modules doesn't match current target_modules\n");
                return 1;
            }
            
        } else {
            LOG_ERR("Failed to load checkpoint, starting from scratch\n");
        }
    }
    
    struct ggml_opt_optimizer_params optimizer_params = ggml_opt_get_default_optimizer_params(nullptr);
    optimizer_params.adamw.alpha = 1e-5f; // learning rate

    std::string optimizer_checkpoint_path;
    if (checkpoint_loaded && !ft_params.resume_from_checkpoint.empty()) {
        std::filesystem::path checkpoint_dir(ft_params.resume_from_checkpoint);
        optimizer_checkpoint_path = (checkpoint_dir / "optimizer.gguf").string();
    }

    struct llama_opt_params lopt_params {
        /*n_ctx_train          =*/  0,
        /*param_filter         =*/  llama_opt_param_filter_lora,
        /*param_filter_ud      =*/  nullptr,
        /*get_opt_pars         =*/  ggml_opt_get_constant_optimizer_params,
        /*get_opt_pars_ud      =*/  &optimizer_params,
        /*optimizer_type       =*/  GGML_OPT_OPTIMIZER_TYPE_ADAMW,
        /*checkpoint_path      =*/  checkpoint_loaded ? optimizer_checkpoint_path.c_str() : nullptr,
        /*load_optimizer_state =*/  checkpoint_loaded,
    };
    
    llama_opt_init(ctx.get(), model.get(), lopt_params);
    
    if (checkpoint_loaded) {
        start_step = llama_opt_get_iter(ctx.get());
    }
    
    if (!trained_adapter) {
        LOG_ERR("No trained adapter available for checkpointing\n");
        return 1;
    }
    
    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - val_split);
    const int64_t training_batches_per_epoch = idata_split;

    if (start_step > 0) {
        int64_t completed_epochs = start_step / training_batches_per_epoch;
        start_epoch = (int)completed_epochs;
    }

    checkpoint_callback_data cb_data = {
        /*ctx                   =*/ ctx.get(),
        /*adapter               =*/ trained_adapter,
        /*checkpoint_save_steps =*/ ft_params.checkpoint_save_steps,
        /*checkpoint_save_dir   =*/ ft_params.checkpoint_save_dir,
        /*global_step           =*/ start_step,
        /*initial_step          =*/ start_step,
        /*current_epoch         =*/ start_epoch,
        /*lora_rank             =*/ ft_params.lora_rank,
        /*lora_alpha            =*/ ft_params.lora_alpha,
        /*target_modules        =*/ target_modules,
        /*learning_rate         =*/ optimizer_params.adamw.alpha,
        /*model_path            =*/ params.model.path,
        /*dataset_path          =*/ params.prompt_file,
    };
    g_checkpoint_data = &cb_data;

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    for (int epoch = start_epoch; epoch < ft_params.num_epochs; ++epoch) {
        LOG_INF("Starting epoch %d (step %ld)\n", epoch, cb_data.global_step);
        cb_data.current_epoch = epoch;
        
        int64_t resume_batch = 0;
        if (start_step > 0 && epoch == start_epoch) {
            resume_batch = start_step % training_batches_per_epoch;
        }
        
        ggml_opt_epoch_callback train_callback = (ft_params.checkpoint_save_steps <= 0) ? 
            ggml_opt_epoch_callback_progress_bar : checkpoint_progress_callback;
        ggml_opt_epoch_callback eval_callback = (ft_params.checkpoint_save_steps <= 0) ? 
            ggml_opt_epoch_callback_progress_bar : checkpoint_progress_callback;

        if (resume_batch > 0) {
            LOG_INF("Resuming training from epoch %d, step %ld \n", epoch, resume_batch);
        } else if (ft_params.checkpoint_save_steps > 0) {
            LOG_INF("Checkpointing enabled, saving every %d steps\n", ft_params.checkpoint_save_steps);
        } else {
            LOG_INF("Checkpointing disabled, using standard progress callback\n");
        }

        llama_opt_epoch(ctx.get(), dataset, result_train, result_eval, idata_split,
            train_callback, eval_callback, resume_batch);
        fprintf(stderr, "\n");

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }

    g_checkpoint_data = nullptr;
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    std::string adapter_filename;
    if (!ft_params.output_adapter_path.empty()) {
        adapter_filename = ft_params.output_adapter_path;
    } else if (has_existing_lora) {
        adapter_filename = "finetuned-lora-adapter.gguf";
        LOG_INF("Finetuned existing lora adapter, saving as: %s\n", adapter_filename.c_str());
    } else {
        adapter_filename = "trained-lora-adapter.gguf";
        LOG_INF("Saving new lora adapter: %s\n", adapter_filename.c_str());
    }

    if (trained_adapter) {
        if (llama_lora_save_adapter(trained_adapter, adapter_filename.c_str(), model.get())) {
            std::ifstream adapter_file(adapter_filename, std::ios::binary | std::ios::ate);
            if (adapter_file.is_open()) {
                std::streamsize adapter_size = adapter_file.tellg();
                LOG_INF("LoRA adapter saved: %s (%.2f MB)\n",
                        adapter_filename.c_str(), adapter_size / (1024.0 * 1024.0));
                adapter_file.close();
            }
        } else {
            LOG_ERR("Failed to save LoRA adapter\n");
        }
    } else {
        LOG_ERR("No trained adapter available for saving\n");
    }

    llama_backend_free();

    return 0;
}
