#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct moss_coreml_model;

struct moss_coreml_model * moss_coreml_load(const char * path_mlmodelc, const char * compute_units);

// Loads one function of a multi-function model (macOS 15 / iOS 18); null
// when the runtime predates multi-function models or the function is absent.
struct moss_coreml_model * moss_coreml_load_function(const char * path_mlmodelc, const char * compute_units,
                                                     const char * function_name);

// Writes the model's function names, comma-separated, into `names` without
// loading or compiling any of them. Returns the count, or -1 when the model
// cannot be read, the runtime predates multi-function models, or the list
// does not fit in `capacity` bytes.
int moss_coreml_function_names(const char * path_mlmodelc, char * names, int capacity);

void moss_coreml_free(struct moss_coreml_model * model);

int moss_coreml_feature_dims(const struct moss_coreml_model * model, const char * name, int64_t * dims,
                             int max_dims);

int moss_coreml_has_state(const struct moss_coreml_model * model);

int moss_coreml_reset_state(struct moss_coreml_model * model);

// Shape of a state buffer of a stateful model; returns the rank, or 0 when the
// model has no such state or it has more than `max_dims` dimensions.
int moss_coreml_state_dims(const struct moss_coreml_model * model, const char * name, int64_t * dims, int max_dims);

// Copies `data`, row-major over `count`, into the box of state `name` that
// starts at `start` (both `rank` long, the state's rank), converting to the
// state's element type. Returns 0, or nonzero (nothing written) when the model
// is not stateful, the state is absent or the box leaves its buffer.
int moss_coreml_write_state(struct moss_coreml_model * model, const char * name, const float * data,
                            const int64_t * start, const int64_t * count, int rank);

// Copies the model's user-defined metadata value for `key` into `value`.
// Returns its length, or -1 when the key is absent or does not fit `capacity`.
int moss_coreml_metadata(const struct moss_coreml_model * model, const char * key, char * value, int capacity);

int moss_coreml_predict(struct moss_coreml_model * model,
                        const char * const * input_names, const float * const * inputs, int n_inputs,
                        const char * const * output_names, float * const * outputs, int n_outputs);

int moss_coreml_predict_batch(struct moss_coreml_model * model,
                              const char * const * input_names, const float * const * inputs, int n_inputs,
                              const char * const * output_names, float * const * outputs, int n_outputs,
                              int batch_size);

const char * moss_coreml_label(const struct moss_coreml_model * model);

#if defined(__cplusplus)
}
#endif
