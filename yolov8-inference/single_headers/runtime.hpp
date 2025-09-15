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

Runtime *load_runtime_library(const string &library_path) {
    void *handle = nullptr;
    try {
        // Load the runtime library using the custom loader
        handle = load_dynamic_library(library_path.c_str());
        // Load runtime interface symbols
        if (!handle) {
            throw std::runtime_error("Failed to load library: " + library_path);
        }
        Runtime *runtime = new Runtime;
        runtime->runtime_initialization = reinterpret_cast<int (*)(void)>(
            get_symbol_address(handle, "runtime_initialization"));
        if (!runtime->runtime_initialization) {
            throw std::runtime_error(
                "Failed to load symbol: runtime_initialization");
        }
        runtime->runtime_initialization_with_args =
            reinterpret_cast<int (*)(int, const char **, const void **)>(
                get_symbol_address(handle, "runtime_initialization_with_args"));
        if (!runtime->runtime_initialization_with_args) {
            throw std::runtime_error(
                "Failed to load symbol: runtime_initialization_with_args");
        }
        runtime->runtime_model_loading =
            reinterpret_cast<int (*)(const char *)>(
                get_symbol_address(handle, "runtime_model_loading"));
        if (!runtime->runtime_model_loading) {
            throw std::runtime_error(
                "Failed to load symbol: runtime_model_loading");
        }
        runtime->send_input = reinterpret_cast<int (*)(tensors_struct *)>(
            get_symbol_address(handle, "send_input"));
        if (!runtime->send_input) {
            throw std::runtime_error("Failed to load symbol: send_input");
        }
        runtime->receive_output = reinterpret_cast<int (*)(tensors_struct **)>(
            get_symbol_address(handle, "receive_output"));
        if (!runtime->receive_output) {
            throw std::runtime_error("Failed to load symbol: receive_output");
        }
        runtime->runtime_destruction = reinterpret_cast<int (*)(void)>(
            get_symbol_address(handle, "runtime_destruction"));
        if (!runtime->runtime_destruction) {
            throw std::runtime_error(
                "Failed to load symbol: runtime_destruction");
        }
        runtime->runtime_error_message =
            reinterpret_cast<const char *(*)(void)>(
                get_symbol_address(handle, "runtime_error_message"));
        if (!runtime->runtime_error_message) {
            throw std::runtime_error(
                "Failed to load symbol: runtime_error_message");
        }
        runtime->runtime_version = reinterpret_cast<const char *(*)(void)>(
            get_symbol_address(handle, "runtime_version"));
        if (!runtime->runtime_version) {
            throw std::runtime_error("Failed to load symbol: runtime_version");
        }
        runtime->runtime_name = reinterpret_cast<const char *(*)(void)>(
            get_symbol_address(handle, "runtime_name"));
        if (!runtime->runtime_name) {
            throw std::runtime_error("Failed to load symbol: runtime_name");
        }
        runtime->handle = handle;  // Store the handle in the Runtime struct
        return runtime;            // Return the initialized Runtime struct
    } catch (const std::exception &e) {
        spdlog::error("Error loading library: {}", e.what());
        exit(EXIT_FAILURE);
    }
}

void destroy_runtime(Runtime *runtime) {
    if (runtime) {
        // Call the runtime destruction function
        if (runtime->runtime_destruction) {
            runtime->runtime_destruction();
        }
        // Close the library handle
        if (runtime->handle) {
            close_dynamic_library(runtime->handle);
        }
        delete runtime;  // Free the Runtime struct
    }
}
