// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

// Description: This is an example of a C program that uses the OAAX runtime to
// send inputs and receive outputs in separate threads.

// Standard libraries
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "metrics.h"        // NOLINT[build/include]
#include "runtime_utils.h"  // NOLINT[build/include]

// C utilities
#include "logger.h"     // NOLINT[build/include]
#include "memory.h"     // NOLINT[build/include]
#include "threading.h"  // NOLINT[build/include]
#include "timer.h"      // NOLINT[build/include]
#include "utils.h"      // NOLINT[build/include]

// Flag to control system info recording thread
static volatile bool record_info_running = false;

// Number of required arguments for the program
#define N_REQUIRED_ARGS 10
// Maximum number of inputs in the pipeline
// Ensure that it's not greater than the runtime's queue capacity
#define MAX_INPUTS_IN_PIPELINE 30

// Logger
Logger *logger = NULL;

// Global variable to hold the original input tensors
tensors_struct *original_input_tensors = NULL;
// Number of outputs received from the runtime
static int received_outputs = 0;
// Number of inferences to run
static int number_of_inferences;
// Average memory and CPU usage
static float average_cpu_usage = 0.0f;
static float average_ram_usage = 0.0f;

/************************************************************************************/
/*Thread function responsible for sending input tensors to the runtime */
/************************************************************************************/
void *send_input_thread(void *arg) {
  Runtime *runtime = (Runtime *)arg;
  int code = 0;

  for (int i = 0; i < number_of_inferences; i++) {
    // Wait until the number of input tensors in the pipeline (number of inputs
    // sent minus number of outputs received) is less than the maximum allowed
    // in the pipeline
    int j = 0;
    while ((i - received_outputs) >= MAX_INPUTS_IN_PIPELINE && j < 500) {
      sleep_ms(10);  // Sleep for a short duration before checking again
      j++;
    }
    // Timeout after 100 iterations (5 seconds)
    if (j >= 500) {
      log_error(logger,
                "Timeout waiting for space in the input pipeline. "
                "Stopping the input sending thread.");
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
    // The inference thread will free input_tensors after processing it

    log_debug(logger, "-> Sent input %d", i + 1);
  }

  return NULL;
}

/************************************************************************************/
/* Thread function responsible for receiving output tensors from the runtime */
/************************************************************************************/
void *receive_output_thread(void *arg) {
  Runtime *runtime = (Runtime *)arg;
  tensors_struct *output_tensors = NULL;
  received_outputs = 0;  // Reset the received outputs counter

  while (received_outputs < number_of_inferences) {
    // Query the runtime for the output tensors
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
    // Log tensors metadata
    print_tensors_metadata(output_tensors);

    // Free the output tensors
    deep_free_tensors_struct(output_tensors);
    output_tensors = NULL;
    // Flag that new output tensors were received
    received_outputs++;

    log_debug(logger, "<- Received output %d", received_outputs);
  }

  return NULL;
}

/************************************************************************************/
/** Thread function to record system information periodically
Here we can add code that can read usage metrics of the relevant NPU to
measure its activity.
*/
/************************************************************************************/
static void *record_info_thread(void *arg) {
  (void)arg;
  while (record_info_running) {
    // Measure CPU usage and RAM usage
    float cpu_usage, ram_kb;
    get_usage(&cpu_usage,
              &ram_kb);  // This sleeps for 1 second to measure cpu usage.
    // Aggregate the cpu_usage and ram_kb by using an exponentially moving
    // average
    float multiplier = 0.95;
    average_cpu_usage =
        (multiplier * cpu_usage) + ((1 - multiplier) * average_cpu_usage);
    average_ram_usage =
        (multiplier * ram_kb) + ((1 - multiplier) * average_ram_usage);
    sleep_ms(100);  // Sleep for another 100ms
  }
  return NULL;
}

// Static functions prototypes
static int init_logger_module();
static int run_inference(Runtime *runtime, Timer *timer);

int main(int argc, char **argv) {
  Timer timer;

  // Initialize the logger
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
  if (parse_args(N_REQUIRED_ARGS, argc, argv, &library_path, &model_path,
                 &image_path, &number_of_inferences, &input_height,
                 &input_width, &nchw, &mean, &std, &num_pairs, &arg_keys,
                 &arg_values, &int_values) < 0) {
    return 1;
  }

  // Initialize and load runtime
  Runtime *runtime = init_runtime_module(library_path, num_pairs, arg_keys,
                                         arg_values, model_path);
  if (runtime == NULL) {
    return 1;
  }

  // Create the input tensors after preprocessing the input image
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
  // Wait a second till the first recording is done
  sleep_ms(1000);

  // Run inference and record timing
  run_inference(runtime, &timer);

  // Stop system info recording
  record_info_running = false;
  thread_join(&info_handle);

  // Save run metrics and cleanup
  float fps_rate = get_fps_rate(&timer, number_of_inferences);
  const char *runtime_name = runtime->runtime_name();
  const char *runtime_version = runtime->runtime_version();

  char *output_file_path = "./metrics.jsonl";
  save_metrics_json(runtime_name, runtime_version, model_path, input_width,
                    input_height, number_of_inferences,
                    fps_rate, average_cpu_usage, average_ram_usage,
                    N_REQUIRED_ARGS, argc, argv, output_file_path);

  // Clean up allocated resources
  log_info(logger, "Inference completed. Cleaning up resources...");
  deep_free_tensors_struct(original_input_tensors);

  if (arg_keys) {
    free((void *)arg_keys);
    free((void *)arg_values);
    free(int_values);
  }
  destroy_runtime(runtime);
  print_human_readable_stats(&timer, number_of_inferences);

  return 0;
}

// Initialize logger and system info
static int init_logger_module() {
  logger = create_logger("C example", "main.log", LOG_DEBUG, LOG_DEBUG);
  if (logger == NULL) {
    printf("Failed to create logger.\n");
    return -1;
  }
  return 0;
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
