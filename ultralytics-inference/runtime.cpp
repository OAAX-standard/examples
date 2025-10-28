#include "single_headers/runtime.hpp"
#include "lib_loader.h"

Runtime *load_runtime_library(const string &library_path) {
  try {
    Runtime *runtime = new Runtime();
    if (!runtime) {
      throw std::runtime_error("Failed to allocate memory for Runtime.");
    }
    runtime->handle = load_dynamic_library(library_path.c_str());
    if (!runtime->handle) {
      delete runtime;
      throw std::runtime_error("Failed to load library: " + library_path);
    }
    runtime->runtime_name = (const char *(*)())get_symbol_address(
        runtime->handle, "runtime_name");
    if (!runtime->runtime_name) {
      destroy_runtime(runtime);
      throw std::runtime_error(
          "Failed to load symbol: runtime_name");
    }
    runtime->runtime_version = (const char *(*)())get_symbol_address(
        runtime->handle, "runtime_version");
    if (!runtime->runtime_version) {
      destroy_runtime(runtime);
      throw std::runtime_error(
          "Failed to load symbol: runtime_version");
    }
    runtime->runtime_error_message = (const char *(*)())get_symbol_address(
        runtime->handle, "runtime_error_message");
    if (!runtime->runtime_error_message) {
        destroy_runtime(runtime);
        throw std::runtime_error("Failed to load symbol: runtime_error_message");
    }
    runtime->runtime_initialization = (int (*)())get_symbol_address(
        runtime->handle, "runtime_initialization");
    runtime->runtime_initialization_with_args = (int (*)(int, const char **, const void **))get_symbol_address(
        runtime->handle, "runtime_initialization_with_args");
    runtime->runtime_model_loading = (int (*)(const char *))get_symbol_address(
        runtime->handle, "runtime_model_loading");
    runtime->send_input = (int (*)(tensors_struct *))get_symbol_address(
        runtime->handle, "send_input");
    if (!runtime->send_input) {
        destroy_runtime(runtime);
        throw std::runtime_error("Failed to load symbol: send_input");
    }
    runtime->receive_output = (int (*)(tensors_struct **))get_symbol_address(
        runtime->handle, "receive_output");
    if (!runtime->receive_output) {
        destroy_runtime(runtime);
        throw std::runtime_error("Failed to load symbol: receive_output");
    }
    runtime->runtime_destruction = (int (*)())get_symbol_address(
        runtime->handle, "runtime_destruction");
    if (!runtime->runtime_destruction) {
        destroy_runtime(runtime);
        throw std::runtime_error("Failed to load symbol: runtime_destruction");
    }
    return runtime;
  } catch (const std::exception &e) {
    spdlog::error("Error loading library: {}", e.what());
    return nullptr;
  }
}

void destroy_runtime(Runtime *runtime) {
  if (runtime) {
    if (runtime->handle) {
      runtime->runtime_destruction();
      close_dynamic_library(runtime->handle);
    }
    delete runtime;
  }
}
