// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

// Description: This is an example of a C program that uses the OAAX runtime to
// send inputs and receive outputs in separate threads.

// Standard libraries
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "runtime_utils.h"  // NOLINT[build/include]

// C utilities
#include "logger.h"     // NOLINT[build/include]
#include "memory.h"     // NOLINT[build/include]
#include "threading.h"  // NOLINT[build/include]
#include "timer.h"      // NOLINT[build/include]
#include "utils.h"      // NOLINT[build/include]

// Number of required arguments for the program
#define N_REQUIRED_ARGS 10
// Maximum number of inputs in the pipeline
#define MAX_INPUTS_IN_PIPELINE 5

// Logger
Logger *logger = NULL;

// Global variable to hold the original input tensors
tensors_struct *original_input_tensors = NULL;
// Number of outputs received from the runtime
static int received_outputs = 0;
static int number_of_inferences;

// Thread function for sending inputs
void *send_input_thread(void *arg) {
  Runtime *runtime = (Runtime *)arg;
  int code = 0;

  for (int i = 0; i < number_of_inferences; i++) {
    // Wait until the number of input tensors in the pipeline is less than the
    // maximum allowed in the pipeline
    int j = 0;
    while ((i - received_outputs) >= MAX_INPUTS_IN_PIPELINE && j < 20) {
      log_debug(logger, "Waiting for space in the input pipeline...");
      sleep_ms(20);  // Sleep for a short duration before checking again
      j++;
    }
    if (j >= 20) {
      log_error(logger, "Timeout waiting for space in the input pipeline.");
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
    while (code != 0 && j < 10) {
      sleep_ms(100);  // Wait before retrying
      code = runtime->receive_output(&output_tensors);
      j++;
    }
    if (code != 0) {
      log_error(logger, "Failed to receive output tensors after 10 attempts.");
      return NULL;
    }

    // if last iteration print out the output
    if (received_outputs == number_of_inferences - 1) {
      print_tensors_metadata(output_tensors);
      // Print tensors data
      log_info(logger, "Output tensors data:");
      for (int64_t i = 0; i < output_tensors->num_tensors; i++) {
        log_info(logger, "Tensor %zu:", i);
        int64_t size =
            output_tensors->shapes[i][0] * output_tensors->shapes[i][1];
        log_info(logger, "Tensor %zu size: %lld", i, size);
        for (int64_t j = 0; j < size; j++) {
          // Print the first 10 elements of the tensor data
          printf("%f ", ((float *)output_tensors->data[i])[j]);
          if (j % 6 == 5) {  // Print a new line every 6 elements
            printf("\n");
          }
        }
      }
    }

    // Free the output tensors
    deep_free_tensors_struct(output_tensors);
    output_tensors = NULL;
    received_outputs++;
    log_debug(logger, "<- Received output %d", received_outputs);
  }

  return NULL;
}

int main(int argc, char **argv) {
  // Utils
  Timer timer;
  // Create a logger that prints and saves logs to a file
  // NOTE: adjust the logger params: file name, file log level, and console log
  // level See OAAX/examples/tools/c-utilities/include/logger.h for more details
  logger = create_logger("C example", "main.log", LOG_DEBUG, LOG_DEBUG);
  if (logger == NULL) {
    printf("Failed to create logger.\n");
    return 1;
  }
  // Check command-line arguments
  if (argc < N_REQUIRED_ARGS) {
    log_error(logger,
              "Usage: %s <library_path> <model_path> <image_path> "
              "<number_of_inferences> <input_height> <input_width> <nchw> "
              "<mean> <std>"
              "[key1 value1 key2 value2 ...]",
              argv[0]);
    return 1;
  }

  char *library_path = argv[1];
  char *model_path = argv[2];
  char *image_path = argv[3];
  number_of_inferences = atoi(argv[4]);
  int input_height = atoi(argv[5]);
  int input_width = atoi(argv[6]);
  int nchw = atoi(argv[7]);  // 0 for NHWC, 1 for NCHW
  if (nchw != 0 && nchw != 1) {
    log_error(logger, "Invalid value for nchw. Must be 0 (NHWC) or 1 (NCHW).");
    return 1;
  }
  float mean = atof(argv[8]);
  float std = atof(argv[9]);
  log_info(logger, "Library path: %s", library_path);
  log_info(logger, "Model path: %s", model_path);
  log_info(logger, "Image path: %s", image_path);
  log_info(logger, "Number of inferences: %d", number_of_inferences);
  log_info(logger, "Input height: %d", input_height);
  log_info(logger, "Input width: %d", input_width);
  log_info(logger, "NCHW: %d", nchw);
  log_info(logger, "Mean: %f", mean);
  log_info(logger, "Std: %f", std);

  // Initialize the runtime environment
  Runtime *runtime = initialize_runtime(library_path);
  if (runtime == NULL) {
    log_error(logger, "Failed to initialize runtime.");
    return 1;
  }

  log_info(logger, "Runtime name: %s - Runtime version: %s",
           runtime->runtime_name(), runtime->runtime_version());

  // Initialize the runtime with arguments
  // These parameters are runtime-specific and may vary based on the runtime you
  // are using.
  // Parse additional runtime initialization arguments from command line
  // Usage: <library_path> <model_path> <image_path> [key1 value1 key2 value2
  // ...]
  int num_extra_args = argc - N_REQUIRED_ARGS;
  int num_pairs = num_extra_args / 2;
  if (num_extra_args % 2 != 0) {
    log_error(logger,
              "Invalid number of extra arguments. Must be in key-value pairs.");
    destroy_runtime(runtime);
    return 1;
  }
  const char **arg_keys = NULL;
  const void **arg_values = NULL;
  int *int_values = NULL;  // To hold integer values if needed

  if (num_pairs > 0) {
    arg_keys = (const char **)malloc(num_pairs * sizeof(char *));
    arg_values = (const void **)malloc(num_pairs * sizeof(void *));
    int_values = (int *)malloc(num_pairs * sizeof(int));
    if (!arg_keys || !arg_values || !int_values) {
      log_error(logger, "Memory allocation failed for runtime args.");
      destroy_runtime(runtime);
      return 1;
    }
    for (int i = 0; i < num_pairs; i++) {
      int id = N_REQUIRED_ARGS + i * 2;  // Starting index for key-value pairs
      arg_keys[i] = argv[id];
      // Check if the argument is numeric
      if (is_numeric(argv[id + 1])) {
        int_values[i] = atoi(argv[id + 1]);
        arg_values[i] = &int_values[i];
      } else {
        // If not numeric, treat it as a string
        arg_values[i] = argv[id + 1];
      }
    }
  } else {
    // Pass empty arrays if no extra arguments are provided
    log_info(logger, "No extra arguments provided for runtime initialization.");
    num_pairs = 0;
    arg_keys = NULL;
    arg_values = NULL;
  }
  int return_code = runtime->runtime_initialization_with_args(
      num_pairs, arg_keys, arg_values);

  if (return_code != 0) {
    log_error(logger, "Failed to initialize runtime environment.");
    destroy_runtime(runtime);  // Clean up resources
    return 1;
  }

  // Load the model
  if (runtime->runtime_model_loading(model_path) != 0) {
    log_error(logger, "Failed to load model.");
    destroy_runtime(runtime);  // Clean up resources
    return 1;
  }

  // Load the image
  // NOTE: Depending on the model inputs, you may need to change the image size,
  // mean, std and the tensors struct Also, make sure to adapt the
  // `resize_image` and `build_tensors_struct` function to your needs
  uint8_t *data = (uint8_t *)load_image(image_path, input_width, input_height,
                                        mean, std, nchw);
  if (data == NULL) {
    log_error(logger, "Failed to load image.");
    destroy_runtime(runtime);  // Clean up resources
    return 1;
  }
  // Create the input tensors struct from the image
  // NOTE: Adjust the image size, mean, std and the tensors struct
  // Also, make sure to adapt the `resize_image` and `build_tensors_struct`
  // function to your needs
  original_input_tensors =
      build_tensors_struct(data, input_height, input_width, 3);
  if (original_input_tensors == NULL) {
    log_error(logger, "Failed to build input tensors.");
    free(data);                // Free the image data
    destroy_runtime(runtime);  // Clean up resources
    return 1;
  }
  log_info(logger, "Input tensors created with %zu tensors.",
           original_input_tensors->num_tensors);
  // Start sending inputs and receiving outputs
  ThreadHandle send_input_thread_handle, receive_output_thread_handle;

  if (thread_create(&send_input_thread_handle, send_input_thread, runtime) !=
      0) {
    log_error(logger, "Failed to create send_input_thread.");
    deep_free_tensors_struct(original_input_tensors);
    destroy_runtime(runtime);
    return 1;
  }

  if (thread_create(&receive_output_thread_handle, receive_output_thread,
                    runtime) != 0) {
    log_error(logger, "Failed to create receive_output_thread.");
    deep_free_tensors_struct(original_input_tensors);
    destroy_runtime(runtime);
    return 1;
  }
  log_info(logger, "Threads created successfully. Starting inference...");

  // Wait for threads to complete
  // record current timestamp
  start_recording(&timer);
  thread_join(&send_input_thread_handle);
  thread_join(&receive_output_thread_handle);
  stop_recording(&timer);

  // Clean up
  log_info(logger, "Inference completed. Cleaning up resources...");
  deep_free_tensors_struct(original_input_tensors);
  original_input_tensors = NULL;

  // Free allocated memory if used
  if (argc > 4) {
    free(arg_keys);
    free(arg_values);
    free(int_values);
  }

  destroy_runtime(runtime);

  // Optional: Print run stats
  print_memory_usage("CLOSE");
  print_human_readable_stats(&timer, number_of_inferences);

  return 0;
}
