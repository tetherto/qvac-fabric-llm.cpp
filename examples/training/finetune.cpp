#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)  // possible loss of data
#endif

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

int main(int argc, char ** argv) {
    common_params params;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    if (params.use_mmap) {
        LOG_INF("%s: force disabling memory mapping because it would result in-read-only pointers to the weights\n",
                __func__);
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
    // load the model and apply lora adapter, if any
    common_init_result   llama_init = common_init_from_params(params);
    llama_model_ptr    & model      = llama_init.model;
    llama_context_ptr  & ctx        = llama_init.context;

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    std::vector<llama_token> tokens = common_tokenize(ctx.get(), params.prompt, true);
    ggml_opt_dataset_t dataset = common_opt_dataset_init(ctx.get(), tokens, llama_n_ctx(ctx.get())/2);

    struct lr_opt & lr = params.lr;
    LOG_INF("-optimizer %s -lr0 %.2g -wd %.2g -lr-min %.2g -min-epochs %.2g -epochs %d -period %.2g -val %.2g\n",
            ggml_opt_optimizer_name(params.optimizer), (double) lr.lr0, (double) lr.wd, (double) lr.lr_min, (double) lr.decay_epochs,
            (unsigned) lr.epochs, (double) params.n_batch / params.n_ubatch, (double) params.val_split);

    struct llama_opt_params lopt_params = llama_opt_default_params();
    lopt_params.get_opt_pars         = common_opt_lr_pars;
    lopt_params.get_opt_pars_ud      = &params.lr;
    lopt_params.optimizer_type       = params.optimizer;
    llama_opt_init(ctx.get(), model.get(), lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    for (lr.epoch = 0; lr.epoch < lr.epochs; ++lr.epoch) {
        llama_opt_epoch(ctx.get(), dataset, result_train, result_eval, idata_split,
            ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    llama_model_save_to_file(model.get(), params.out_file.c_str());

    llama_backend_free();

    return 0;
}
