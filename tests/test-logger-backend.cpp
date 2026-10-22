#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"

static int backend_score() {
    GGML_LOG_INFO("logger reload test backend\n");
    return 0;
}

GGML_BACKEND_DL_SCORE_IMPL(backend_score)
