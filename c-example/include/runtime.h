// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#ifndef C_EXAMPLE_INCLUDE_RUNTIME_H_
#define C_EXAMPLE_INCLUDE_RUNTIME_H_

#include "interface.h"  // NOLINT[build/include]

// The OAAX 2.0 runtime functions, resolved from a runtime shared library.
typedef struct Runtime {
  RuntimeStatus (*init)(Config config);
  RuntimeStatus (*load_models)(int num_models,
                               const ModelConfig *model_configs);
  RuntimeStatus (*enqueue_input)(int model_id, Tensors *input_tensors);
  RuntimeStatus (*retrieve_output)(int *model_id, Tensors **output_tensors,
                                   int timeout_ms);
  RuntimeStatus (*cleanup)(void);
  const char *(*get_error)(void);
  const char *(*get_version)(void);
  const char *(*get_name)(void);
  const char *(*get_info)(void);

  void *handle;  // Handle of the loaded library
} Runtime;

/**
 * @brief Load a runtime library and resolve all the OAAX 2.0 functions.
 * @param [in] library_path Path to the runtime shared library
 * @param [out] runtime Runtime to fill in
 * @return 0 on success, -1 on failure (an error is printed to stderr)
 */
int load_runtime_library(const char *library_path, Runtime *runtime);

/**
 * @brief Unload the runtime library.
 * @note Call `runtime->cleanup()` first: unloading a runtime that is still
 * initialized can deadlock.
 */
void unload_runtime_library(Runtime *runtime);

/**
 * @brief Print a failed call and the runtime's error message to stderr.
 */
void report_error(const Runtime *runtime, const char *what,
                  RuntimeStatus status);

#endif  // C_EXAMPLE_INCLUDE_RUNTIME_H_
