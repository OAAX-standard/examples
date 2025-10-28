#pragma once

#include "lib_loader.h"
#include "tensors_struct.h"
#include <string>
#include <stdexcept>
#include <spdlog/spdlog.h>

using namespace std;

// This file contains the C-style interface for the runtime.
// It is used to load the runtime as a shared library and call its functions.

typedef struct Runtime {
    int (*runtime_initialization)();
    int (*runtime_initialization_with_args)(int, const char **, const void **);
    int (*runtime_model_loading)(const char *);
    int (*send_input)(tensors_struct *);
    int (*receive_output)(tensors_struct **);
    int (*runtime_destruction)();
    const char *(*runtime_error_message)();
    const char *(*runtime_version)();
    const char *(*runtime_name)();

    // Internal fields
    void *handle;  // Handle to the loaded library
} Runtime;

Runtime *load_runtime_library(const string &library_path);
void destroy_runtime(Runtime *runtime);
