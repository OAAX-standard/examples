// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include "runtime_utils.h"  // NOLINT[build/include]

// C Utilities
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cJSON.h"       // NOLINT[build/include]
#include "lib_loader.h"  // NOLINT[build/include]
#include "logger.h"      // NOLINT[build/include]

#ifdef _WIN32
#include <stdio.h>
#include <windows.h>
static char win32_dl_error_msg[512];
static const char *get_dl_error() {
  DWORD err = GetLastError();
  if (err == 0) return NULL;
  FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                 NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                 win32_dl_error_msg, sizeof(win32_dl_error_msg), NULL);
  return win32_dl_error_msg;
}
#define DL_ERROR get_dl_error()
#else
#include <dlfcn.h>
#define DL_ERROR dlerror()
#endif

extern Logger *logger;

static char *copy_string(const char *str) {
  if (str == NULL) return NULL;

  size_t len = strlen(str);
  char *copy = (char *)malloc(len + 1);
  if (copy == NULL) return NULL;

  snprintf(copy, len + 1, "%s", str);
  return copy;
}

void destroy_runtime(Runtime *runtime) {
  if (runtime == NULL) return;

  if (runtime->runtime_destruction != NULL) {
    runtime->runtime_destruction();
  }

  if (runtime->_library_path != NULL) {
    free(runtime->_library_path);
    runtime->_library_path = NULL;
  }

  if (runtime->_handle != NULL) {
    close_dynamic_library(runtime->_handle);
    runtime->_handle = NULL;
  }

  free(runtime);
  runtime = NULL;
}

Runtime *initialize_runtime(const char *library_path) {
  Runtime *runtime = (Runtime *)calloc(1, sizeof(Runtime));
  if (runtime == NULL) {
    log_error(logger, "Failed to allocate memory for Runtime.");
    return NULL;
  }

  log_debug(logger, "Initializing runtime with library: %s", library_path);

  runtime->_library_path = NULL;
  runtime->_handle = NULL;

  // Copy the library path
  runtime->_library_path = malloc(strlen(library_path) + 1);
  if (runtime->_library_path == NULL) {
    destroy_runtime(runtime);
    log_error(logger, "Failed to load library: %s",
              DL_ERROR ? DL_ERROR : "Unknown error");
    return NULL;
  }
  snprintf(runtime->_library_path, strlen(library_path) + 1, "%s",
           library_path);

  // Load the shared library
  runtime->_handle = load_dynamic_library(library_path);
  if (runtime->_handle == NULL) {
    destroy_runtime(runtime);
    log_error(logger, "Failed to load library: %s", DL_ERROR);
    return NULL;
  }
  log_debug(logger, "Loaded library handle: %p", runtime->_handle);

  runtime->runtime_initialization =
      get_symbol_address(runtime->_handle, "runtime_initialization");
  if (runtime->runtime_initialization == NULL) {
    log_error(logger, "`runtime_initialization` not implemented: %s.",
              DL_ERROR);
  }
  log_debug(logger, "Loaded `runtime_initialization` function: %p",
            runtime->runtime_initialization);
  runtime->runtime_initialization_with_args =
      get_symbol_address(runtime->_handle, "runtime_initialization_with_args");
  if (runtime->runtime_initialization_with_args == NULL) {
    log_error(logger, "`runtime_initialization_with_args` not implemented: %s.",
              DL_ERROR);
  }
  log_debug(logger, "Loaded `runtime_initialization_with_args` function: %p",
            runtime->runtime_initialization_with_args);
  runtime->runtime_model_loading =
      get_symbol_address(runtime->_handle, "runtime_model_loading");
  if (runtime->runtime_model_loading == NULL) {
    destroy_runtime(runtime);
    log_error(logger, "Failed to load `runtime_model_loading` function: %s.",
              DL_ERROR);
    return NULL;
  }
  log_debug(logger, "Loaded `runtime_model_loading` function: %p",
            runtime->runtime_model_loading);
  runtime->send_input = get_symbol_address(runtime->_handle, "send_input");
  if (runtime->send_input == NULL) {
    destroy_runtime(runtime);
    log_error(logger, "Failed to load `send_input` function: %s.", DL_ERROR);
    return NULL;
  }
  runtime->receive_output =
      get_symbol_address(runtime->_handle, "receive_output");
  if (runtime->receive_output == NULL) {
    destroy_runtime(runtime);
    log_error(logger, "Failed to load `receive_output` function: %s.",
              DL_ERROR);
    return NULL;
  }
  runtime->runtime_destruction =
      get_symbol_address(runtime->_handle, "runtime_destruction");
  if (runtime->runtime_destruction == NULL) {
    destroy_runtime(runtime);
    log_error(logger, "Failed to load `runtime_destruction` function: %s.",
              DL_ERROR);
    return NULL;
  }
  log_debug(logger, "Loaded `runtime_destruction` function: %p",
            runtime->runtime_destruction);
  runtime->runtime_error_message =
      get_symbol_address(runtime->_handle, "runtime_error_message");
  if (runtime->runtime_error_message == NULL) {
    destroy_runtime(runtime);
    log_error(logger, "Failed to load `runtime_error_message` function: %s.",
              DL_ERROR);
    return NULL;
  }
  log_debug(logger, "Loaded `runtime_error_message` function: %p",
            runtime->runtime_error_message);
  runtime->runtime_version =
      get_symbol_address(runtime->_handle, "runtime_version");
  if (runtime->runtime_version == NULL) {
    destroy_runtime(runtime);
    log_error(logger, "Failed to load `runtime_version` function: %s.",
              DL_ERROR);
    return NULL;
  }
  log_debug(logger, "Loaded `runtime_version` function: %p",
            runtime->runtime_version);
  runtime->runtime_name = get_symbol_address(runtime->_handle, "runtime_name");
  if (runtime->runtime_name == NULL) {
    destroy_runtime(runtime);
    log_error(logger, "Failed to load `runtime_name` function: %s.", DL_ERROR);
    return NULL;
  }
  log_debug(logger, "Loaded `runtime_name` function: %p",
            runtime->runtime_name);

  return runtime;
}

void resize_image(const unsigned char *image, int width, int height,
                  int new_width, int new_height, float *resized_image) {
  if (image == NULL) return;
  if (resized_image == NULL) return;

  // Simple resizing algorithm: nearest neighbor interpolation
  double x_ratio = (double)width / new_width;
  double y_ratio = (double)height / new_height;

  for (int y = 0; y < new_height; y++) {
    for (int x = 0; x < new_width; x++) {
      int px = (int)(x * x_ratio);
      int py = (int)(y * y_ratio);
      resized_image[(y * new_width + x) * 3] = image[(py * width + px) * 3];
      resized_image[(y * new_width + x) * 3 + 1] =
          image[(py * width + px) * 3 + 1];
      resized_image[(y * new_width + x) * 3 + 2] =
          image[(py * width + px) * 3 + 2];
    }
  }
}

void *load_image(const char *image_path, int new_width, int new_height,
                 float mean, float std, bool nchw) {
  FILE *input_file = fopen(image_path, "rb");
  if (!input_file) {
    log_error(logger, "Error: Couldn't open the image file.");
    return NULL;
  }

  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, input_file);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  int width = cinfo.output_width;
  int height = cinfo.output_height;
  int num_channels = cinfo.output_components;

  unsigned char *image = (unsigned char *)malloc(width * height * num_channels *
                                                 sizeof(unsigned char));
  float *resized_image =
      (float *)malloc(new_height * new_width * num_channels * sizeof(float));

  while (cinfo.output_scanline < cinfo.output_height) {
    unsigned char *row = &image[cinfo.output_scanline * cinfo.output_width *
                                cinfo.output_components];
    jpeg_read_scanlines(&cinfo, &row, 1);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(input_file);

  // resize and convert the image to float
  resize_image(image, width, height, new_width, new_height, resized_image);

  // Normalize the image
  for (int i = 0; i < new_height * new_width * num_channels; i++)
    resized_image[i] = (resized_image[i] - mean) / std;

  // Convert to NCHW format
  if (nchw) {
    float *transposed_image =
        (float *)malloc(new_height * new_width * num_channels * sizeof(float));
    for (int c = 0; c < num_channels; c++) {
      for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
          transposed_image[c * new_height * new_width + y * new_width + x] =
              resized_image[y * new_width * num_channels + x * num_channels +
                            c];
        }
      }
    }
    free(resized_image);
    resized_image = transposed_image;
  }

  free(image);
  return (void *)resized_image;
}

tensors_struct *build_tensors_struct(uint8_t *data, size_t height, size_t width,
                                     size_t channels) {
  tensors_struct *input_tensors =
      (tensors_struct *)malloc(sizeof(tensors_struct));

  if (input_tensors == NULL) {
    log_error(logger, "Failed to allocate memory for input tensors.");
    return NULL;
  }
  input_tensors->num_tensors = 1;
  input_tensors->names =
      (char **)malloc(input_tensors->num_tensors * sizeof(char *));
  input_tensors->data_types = (tensor_data_type *)malloc(
      input_tensors->num_tensors * sizeof(tensor_data_type));
  input_tensors->ranks =
      (size_t *)malloc(input_tensors->num_tensors * sizeof(size_t));
  input_tensors->shapes =
      (size_t **)malloc(input_tensors->num_tensors * sizeof(size_t *));
  input_tensors->data =
      (void **)malloc(input_tensors->num_tensors * sizeof(void *));

  // First tensor: input image
  input_tensors->names[0] = copy_string("image-");
  input_tensors->data_types[0] = DATA_TYPE_FLOAT;
  input_tensors->ranks[0] = 4;
  input_tensors->shapes[0] =
      (size_t *)malloc(input_tensors->ranks[0] * sizeof(size_t));
  input_tensors->shapes[0][0] = 1;
  input_tensors->shapes[0][1] = channels;
  input_tensors->shapes[0][2] = height;
  input_tensors->shapes[0][3] = width;
  input_tensors->data[0] = (void *)data;

  return input_tensors;
}

int is_numeric(const char *str) {
  if (str == NULL || *str == '\0') {
    return 0;  // Empty or NULL string is not numeric
  }

  while (*str) {
    if (!isdigit((unsigned char)*str)) {
      return 0;  // Non-digit character found
    }
    str++;
  }
  return 1;  // All characters are digits
}

Runtime *init_runtime_module(const char *library_path, int num_pairs,
                             const char **arg_keys, const void **arg_values,
                             const char *model_path) {
  Runtime *runtime = initialize_runtime(library_path);
  if (runtime == NULL) {
    log_error(logger, "Failed to initialize runtime.");
    return NULL;
  }
  log_info(logger, "Runtime name: %s - Runtime version: %s",
           runtime->runtime_name(), runtime->runtime_version());

  if (runtime->runtime_initialization_with_args(num_pairs, arg_keys,
                                                arg_values) != 0) {
    log_error(logger, "Failed to initialize runtime environment.");
    destroy_runtime(runtime);
    return NULL;
  }

  if (runtime->runtime_model_loading(model_path) != 0) {
    log_error(logger, "Failed to load model.");
    destroy_runtime(runtime);
    return NULL;
  }
  return runtime;
}

// Load image and build input tensor struct
tensors_struct *prepare_input(const char *image_path, int width, int height,
                              float mean, float std, int nchw) {
  uint8_t *data =
      (uint8_t *)load_image(image_path, width, height, mean, std, nchw);
  if (data == NULL) {
    log_error(logger, "Failed to load image.");
    return NULL;
  }
  tensors_struct *ts = build_tensors_struct(data, height, width, 3);
  if (ts == NULL) {
    log_error(logger, "Failed to build input tensors.");
    free(data);
    return NULL;
  }
  log_info(logger, "Input tensors created with %zu tensors.", ts->num_tensors);
  return ts;
}

void print_usage(const char *prog_name) {
  log_error(logger,
            "Usage: %s <library_path> <model_path> <image_path> "
            "<number_of_inferences> <input_height> <input_width> <nchw> "
            "<mean> <std>"
            "[key1 value1 key2 value2 ...]",
            prog_name);
}

int parse_args(int n_required_args, int argc, char **argv, char **library_path,
               char **model_path, char **image_path, int *num_inferences,
               int *input_height, int *input_width, int *nchw, float *mean,
               float *std, int *num_pairs, const char ***arg_keys,
               const void ***arg_values, int **int_values) {
  if (argc < n_required_args) {
    print_usage(argv[0]);
    return -1;
  }
  *library_path = argv[1];
  *model_path = argv[2];
  *image_path = argv[3];
  *num_inferences = atoi(argv[4]);
  *input_height = atoi(argv[5]);
  *input_width = atoi(argv[6]);
  *nchw = atoi(argv[7]);
  if (*nchw != 0 && *nchw != 1) {
    log_error(logger, "Invalid value for nchw. Must be 0 (NHWC) or 1 (NCHW).");
    return -1;
  }
  *mean = atof(argv[8]);
  *std = atof(argv[9]);

  int extra = argc - n_required_args;
  *num_pairs = extra / 2;
  if (extra % 2 != 0) {
    log_error(logger,
              "Invalid number of extra arguments. Must be in key-value pairs.");
    return -1;
  }
  if (*num_pairs > 0) {
    *arg_keys = malloc(*num_pairs * sizeof(char *));
    *arg_values = malloc(*num_pairs * sizeof(void *));
    *int_values = malloc(*num_pairs * sizeof(int));
    if (!*arg_keys || !*arg_values || !*int_values) {
      log_error(logger, "Memory allocation failed for runtime args.");
      return -1;
    }
    for (int i = 0; i < *num_pairs; i++) {
      int idx = n_required_args + i * 2;
      (*arg_keys)[i] = argv[idx];
      if (is_numeric(argv[idx + 1])) {
        (*int_values)[i] = atoi(argv[idx + 1]);
        (*arg_values)[i] = &(*int_values)[i];
      } else {
        (*int_values)[i] = -1;
        (*arg_values)[i] = argv[idx + 1];
      }
    }
  } else {
    *arg_keys = NULL;
    *arg_values = NULL;
    *int_values = NULL;
  }
  // Log argument values
  log_info(logger, "Library path: %s", *library_path);
  log_info(logger, "Model path: %s", *model_path);
  log_info(logger, "Image path: %s", *image_path);
  log_info(logger, "Number of inferences: %d", *num_inferences);
  log_info(logger, "Input height: %d", *input_height);
  log_info(logger, "Input width: %d", *input_width);
  log_info(logger, "NCHW: %d", *nchw);
  log_info(logger, "Mean: %f", *mean);
  log_info(logger, "Std: %f", *std);
  return 0;
}

void save_metrics_json(const char *runtime_name, const char *runtime_version,
                       const char *model_name, int input_width,
                       int input_height, float number_of_inferences,
                       float avg_throughput, float cpu_usage, float ram_usage,
                       int n_required_args, int argc, char **argv,
                       const char *json_path) {
  cJSON *root = cJSON_CreateObject();
  if (root == NULL) {
    fprintf(stderr, "Failed to create JSON object\n");
    return;
  }
  // Add current date and time
  time_t now = time(NULL);
  char time_buffer[256];
  strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%dT%H:%M:%SZ",
           gmtime(&now));
  cJSON_AddStringToObject(root, "datetime", time_buffer);

  // Add runtime info
  cJSON_AddStringToObject(root, "runtime_name", runtime_name);
  cJSON_AddStringToObject(root, "runtime_version", runtime_version);

  //// Add runtime config
  for (int i = n_required_args; i < argc; i += 2) {
    cJSON_AddStringToObject(root, argv[i], argv[i + 1]);
  }

  // Add model info
  cJSON_AddStringToObject(root, "model_name", model_name);
  cJSON_AddNumberToObject(root, "input_width", input_width);
  cJSON_AddNumberToObject(root, "input_height", input_height);

  // Add run metrics
  cJSON_AddNumberToObject(root, "number_of_inferences", number_of_inferences);
  cJSON_AddNumberToObject(root, "throughput", avg_throughput);
  cJSON_AddNumberToObject(root, "cpu_usage", cpu_usage);
  cJSON_AddNumberToObject(root, "ram_usage", ram_usage);

  // Save the JSON object to a file
  FILE *file = fopen(json_path, "a");
  if (file != NULL) {
    char *json_string = cJSON_PrintUnformatted(root);
    if (json_string != NULL) {
      fprintf(file, "%s\n", json_string);
      free(json_string);
    }
    fclose(file);
  }

  cJSON_Delete(root);
}