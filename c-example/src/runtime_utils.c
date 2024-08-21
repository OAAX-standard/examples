#include "runtime_utils.h"
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

void destroy_runtime(Runtime *runtime) {
    if (runtime == NULL)
        return;

    if(runtime->runtime_destruction != NULL){
        runtime->runtime_destruction();
    }

    if (runtime->_library_path != NULL) {
        free(runtime->_library_path);
        runtime->_library_path = NULL;
    }

    if (runtime->_handle != NULL) {
        dlclose(runtime->_handle);
        runtime->_handle = NULL;
    }

    free(runtime);
    runtime = NULL;
}

Runtime *initialize_runtime(const char *library_path) {
    Runtime *runtime = (Runtime *) malloc(sizeof(Runtime));
    if (runtime == NULL) {
        printf("Failed to allocate memory for Runtime.\n");
        return NULL;
    }

    runtime->_library_path = NULL;
    runtime->_handle = NULL;

    // Copy the library path
    runtime->_library_path = malloc(strlen(library_path) + 1);
    if (runtime->_library_path == NULL) {
        destroy_runtime(runtime);
        printf("Failed to allocate memory for library path variable.\n");
        return NULL;
    }
    strcpy(runtime->_library_path, library_path);

    // Load the shared library
    runtime->_handle = dlopen(library_path, RTLD_LAZY);
    if (runtime->_handle == NULL) {
        destroy_runtime(runtime);
        printf("Failed to load library: %s\n", dlerror());
        return NULL;
    }

    runtime->runtime_initialization = dlsym(runtime->_handle, "runtime_initialization");
    if (runtime->runtime_initialization == NULL) {
        printf("`runtime_initialization` not implemented: %s.\n", dlerror());
    }
    runtime->runtime_initialization_with_args = dlsym(runtime->_handle, "runtime_initialization_with_args");
    if (runtime->runtime_initialization_with_args == NULL) {
        printf("`runtime_initialization_with_args` not implemented: %s.\n", dlerror());
    }
    runtime->runtime_model_loading = dlsym(runtime->_handle, "runtime_model_loading");
    if (runtime->runtime_model_loading == NULL) {
        destroy_runtime(runtime);
        printf("Failed to load `runtime_model_loading` function: %s.\n", dlerror());
        return NULL;
    }
    runtime->send_input = dlsym(runtime->_handle, "send_input");
    if (runtime->send_input == NULL) {
        destroy_runtime(runtime);
        printf("Failed to load `send_input` function: %s.\n", dlerror());
        return NULL;
    }
    runtime->receive_output = dlsym(runtime->_handle, "receive_output");
    if (runtime->receive_output == NULL) {
        destroy_runtime(runtime);
        printf("Failed to load `receive_output` function: %s.\n", dlerror());
        return NULL;
    }
    runtime->runtime_destruction = dlsym(runtime->_handle, "runtime_destruction");
    if (runtime->runtime_destruction == NULL) {
        destroy_runtime(runtime);
        printf("Failed to load `runtime_destruction` function: %s.\n", dlerror());
        return NULL;
    }
    runtime->runtime_error_message = dlsym(runtime->_handle, "runtime_error_message");
    if (runtime->runtime_error_message == NULL) {
        destroy_runtime(runtime);
        printf("Failed to load `runtime_error_message` function: %s.\n", dlerror());
        return NULL;
    }
    runtime->runtime_version = dlsym(runtime->_handle, "runtime_version");
    if (runtime->runtime_version == NULL) {
        destroy_runtime(runtime);
        printf("Failed to load `runtime_version` function: %s.\n", dlerror());
        return NULL;
    }
    runtime->runtime_name = dlsym(runtime->_handle, "runtime_name");
    if (runtime->runtime_name == NULL) {
        destroy_runtime(runtime);
        printf("Failed to load `runtime_name` function: %s.\n", dlerror());
        return NULL;
    }

    return runtime;
}


