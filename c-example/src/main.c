// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

// Description: This is an example of a C program that uses the OAAX runtime to
// send inputs and receive outputs in separate threads.

// Standard libraries
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "metrics.h"        // NOLINT[build/include]
#include "runtime_utils.h"  // NOLINT[build/include]

// C utilities
#include <stdbool.h>

#include "logger.h"     // NOLINT[build/include]
#include "memory.h"     // NOLINT[build/include]
#include "sysinfo.h"    // NOLINT[build/include]
#include "threading.h"  // NOLINT[build/include]
#include "timer.h"      // NOLINT[build/include]
#include "utils.h"      // NOLINT[build/include]

// Flag to control system info recording thread
static volatile bool record_info_running = false;

// Number of required arguments for the program
#define N_REQUIRED_ARGS 10
// Maximum number of inputs in the pipeline
#define MAX_INPUTS_IN_PIPELINE 30

// Logger
Logger *logger = NULL;

// Global variable to hold the original input tensors
tensors_struct *original_input_tensors = NULL;
// Number of outputs received from the runtime
static int received_outputs = 0;
static int number_of_inferences;
static SystemInfo system_info;
static RealTimeSystemInfo global_real_time_info, instant_real_time_info;

// Thread function for sending inputs
void *send_input_thread(void *arg) {
  Runtime *runtime = (Runtime *)arg;
  int code = 0;

  for (int i = 0; i < number_of_inferences; i++) {
    // Wait until the number of input tensors in the pipeline is less than the
    // maximum allowed in the pipeline
    int j = 0;
    while ((i - received_outputs) >= MAX_INPUTS_IN_PIPELINE && j < 500) {
      sleep_ms(10);  // Sleep for a short duration before checking again
      j++;
    }
    // Timeout after 500 iterations (5 seconds)
    if (j >= 500) {
      log_error(logger,
                "Timeout waiting for space in the input pipeline. "
                "Stopping the inference thread.");
      return NULL;  // Exit the thread if timeout occurs
    }
    // Deep copy the input tensors
    tensors_struct *input_tensors =
        deep_copy_tensors_struct(original_input_tensors);
    if (input_tensors == NULL) {
      log_error(logger, "Failed to deep copy input tensors.");
      continue;
    }

    // Send the input tensors
    code = runtime->send_input(input_tensors);
    if (code != 0) {
      log_error(logger, "Failed to send input tensors.");
      deep_free_tensors_struct(input_tensors);  // Free before returning
      return NULL;
    }

    // Ownership of input_tensors is transferred to the runtime
    // The inference thread will free input_tensors after processing

    log_debug(logger, "-> Sent input %d", i + 1);
  }

  return NULL;
}

// Thread function for receiving outputs
void *receive_output_thread(void *arg) {
  Runtime *runtime = (Runtime *)arg;
  tensors_struct *output_tensors = NULL;
  received_outputs = 0;  // Reset the received outputs counter

  while (received_outputs < number_of_inferences) {
    // Receive the output tensors
    int code;
    code = runtime->receive_output(&output_tensors);
    int j = 0;
    while (code != 0 && j < 50) {
      sleep_ms(100);  // Wait before retrying
      code = runtime->receive_output(&output_tensors);
      j++;
    }
    // Time out after 5 seconds (50 attempts)
    if (code != 0) {
      log_error(logger, "Failed to receive output tensors after 50 attempts.");
      return NULL;
    }
    print_tensors_metadata(output_tensors);

    // Free the output tensors
    deep_free_tensors_struct(output_tensors);
    output_tensors = NULL;
    received_outputs++;
    log_debug(logger, "<- Received output %d", received_outputs);
  }

  return NULL;
}

// Thread function to record real-time system info periodically
static void *record_info_thread(void *arg) {
  (void)arg;
  while (record_info_running) {
    get_real_time_system_info(&instant_real_time_info);
    sleep_ms(100);  // Record every 100ms
  }
  return NULL;
}

// Forward declarations for modular functions
static int init_logger_module();
static Runtime *init_runtime_module(const char *library_path, int num_pairs,
                                    const char **arg_keys,
                                    const void **arg_values,
                                    const char *model_path);
static tensors_struct *prepare_input(const char *image_path, int width,
                                     int height, float mean, float std,
                                     int nchw);
static void print_usage(const char *prog_name);
static int parse_args(int argc, char **argv, char **library_path,
                      char **model_path, char **image_path, int *num_inferences,
                      int *input_height, int *input_width, int *nchw,
                      float *mean, float *std, int *num_pairs,
                      const char ***arg_keys, const void ***arg_values,
                      int **int_values);
static void save_and_cleanup(const char *library_path, const char *model_path,
                             const char *image_path, Runtime *runtime,
                             tensors_struct *input_tensors, Timer *timer,
                             int num_pairs, const char **arg_keys,
                             const void **arg_values, int *int_values);
static int run_inference(Runtime *runtime, Timer *timer);

int main(int argc, char **argv) {
  Timer timer;
  if (init_logger_module() != 0) {
    return 1;
  }

  // Parse command-line arguments
  char *library_path, *model_path, *image_path;
  int num_inferences, input_height, input_width, nchw;
  float mean, std;
  int num_pairs;
  const char **arg_keys;
  const void **arg_values;
  int *int_values;
  if (parse_args(argc, argv, &library_path, &model_path, &image_path,
                 &number_of_inferences, &input_height, &input_width, &nchw,
                 &mean, &std, &num_pairs, &arg_keys, &arg_values,
                 &int_values) < 0) {
    return 1;
  }

  // Initialize and load runtime
  Runtime *runtime = init_runtime_module(library_path, num_pairs, arg_keys,
                                         arg_values, model_path);
  if (runtime == NULL) {
    return 1;
  }

  // Prepare input tensors
  original_input_tensors =
      prepare_input(image_path, input_width, input_height, mean, std, nchw);
  if (original_input_tensors == NULL) {
    destroy_runtime(runtime);
    return 1;
  }

  // Start background thread for system info recording
  record_info_running = true;
  ThreadHandle info_handle;
  if (thread_create(&info_handle, record_info_thread, NULL) != 0) {
    log_error(logger, "Failed to create record_info_thread.");
    record_info_running = false;
  }

  // Run inference and record timing
  if (run_inference(runtime, &timer) != 0) {
    deep_free_tensors_struct(original_input_tensors);
    record_info_running = false;
    thread_join(&info_handle);
    destroy_runtime(runtime);
    return 1;
  }

  // Stop system info recording
  record_info_running = false;
  thread_join(&info_handle);

  // Save metrics and cleanup
  save_and_cleanup(library_path, model_path, image_path, runtime,
                   original_input_tensors, &timer, num_pairs, arg_keys,
                   arg_values, int_values);
  return 0;
}

// Initialize logger and system info
static int init_logger_module() {
  get_system_info(&system_info);
  logger = create_logger("C example", "main.log", LOG_DEBUG, LOG_DEBUG);
  if (logger == NULL) {
    printf("Failed to create logger.\n");
    return -1;
  }
  return 0;
}

// Initialize, configure, and load model into runtime
static Runtime *init_runtime_module(const char *library_path, int num_pairs,
                                    const char **arg_keys,
                                    const void **arg_values,
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
static tensors_struct *prepare_input(const char *image_path, int width,
                                     int height, float mean, float std,
                                     int nchw) {
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

// Run inference: start threads, record timer, and join threads
static int run_inference(Runtime *runtime, Timer *timer) {
  ThreadHandle send_handle, recv_handle;
  if (thread_create(&send_handle, send_input_thread, runtime) != 0) {
    log_error(logger, "Failed to create send_input_thread.");
    return -1;
  }
  if (thread_create(&recv_handle, receive_output_thread, runtime) != 0) {
    log_error(logger, "Failed to create receive_output_thread.");
    return -1;
  }
  log_info(logger, "Threads created successfully. Starting inference...");

  start_recording(timer);
  thread_join(&send_handle);
  thread_join(&recv_handle);
  stop_recording(timer);

  return 0;
}

// Print program usage and exit
static void print_usage(const char *prog_name) {
  log_error(logger,
            "Usage: %s <library_path> <model_path> <image_path> "
            "<number_of_inferences> <input_height> <input_width> <nchw> "
            "<mean> <std>"
            "[key1 value1 key2 value2 ...]",
            prog_name);
}

// Parse and validate command-line arguments
static int parse_args(int argc, char **argv, char **library_path,
                      char **model_path, char **image_path, int *num_inferences,
                      int *input_height, int *input_width, int *nchw,
                      float *mean, float *std, int *num_pairs,
                      const char ***arg_keys, const void ***arg_values,
                      int **int_values) {
  if (argc < N_REQUIRED_ARGS) {
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

  int extra = argc - N_REQUIRED_ARGS;
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
      int idx = N_REQUIRED_ARGS + i * 2;
      (*arg_keys)[i] = argv[idx];
      if (is_numeric(argv[idx + 1])) {
        (*int_values)[i] = atoi(argv[idx + 1]);
        (*arg_values)[i] = &(*int_values)[i];
      } else {
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

// Save metrics, cleanup resources and exit
static void save_and_cleanup(const char *library_path, const char *model_path,
                             const char *image_path, Runtime *runtime,
                             tensors_struct *input_tensors, Timer *timer,
                             int num_pairs, const char **arg_keys,
                             const void **arg_values, int *int_values) {
  log_info(logger, "Inference completed. Cleaning up resources...");
  deep_free_tensors_struct(input_tensors);

  char *output_file_path = "./metrics.jsonl";
  float fps_rate = get_fps_rate(timer, number_of_inferences);

  save_metrics_json(
      model_path, image_path, library_path, num_pairs, arg_keys, arg_values,
      runtime->runtime_name(), runtime->runtime_version(), number_of_inferences,
      &system_info, &instant_real_time_info, fps_rate, output_file_path);

  if (arg_keys) {
    free((void *)arg_keys);
    free((void *)arg_values);
    free(int_values);
  }
  destroy_runtime(runtime);
  print_memory_usage("CLOSE");
  print_human_readable_stats(timer, number_of_inferences);
}
