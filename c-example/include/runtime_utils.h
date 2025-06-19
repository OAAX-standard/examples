// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#ifndef C_EXAMPLE_INCLUDE_RUNTIME_UTILS_H_
#define C_EXAMPLE_INCLUDE_RUNTIME_UTILS_H_

// clang-format off
#include <stdio.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <jpeglib.h>  // Jpeglib should always be included after stdio.h
// clang-format on

#include "tensors_struct.h"  // NOLINT[build/include]
#include "timer.h"           // NOLINT[build/include]

typedef struct Runtime {
  // Runtime interface functions
  int (*runtime_initialization)();
  int (*runtime_initialization_with_args)(int length, const char **keys,
                                          const void **values);
  int (*runtime_model_loading)(const char *file_path);
  int (*send_input)(tensors_struct *input_tensors);
  int (*receive_output)(tensors_struct **output_tensors);
  int (*runtime_destruction)();
  const char *(*runtime_error_message)();
  const char *(*runtime_version)();
  const char *(*runtime_name)();

  // Internal fields
  char *_library_path;
  void *_handle;
} Runtime;

// Function prototypes
Runtime *initialize_runtime(const char *library_path);
void destroy_runtime(Runtime *runtime_env);

/**
 * @brief Resize the image using nearest neighbor interpolation
 * @param [in] image Input image
 * @param [in] width Image width
 * @param [in] height Image height
 * @param [in] new_width New image width
 * @param [in] new_height New image height
 * @param [out] resized_image Pointer to the resized image in float format
 */
void resize_image(const unsigned char *image, int width, int height,
                  int new_width, int new_height, float *resized_image);

/**
 * @brief Load the image from the file path, resize it, normalize it and convert
 * it to float.
 * @param [in] image_path Path to the image file
 * @param [in] new_width Desired width of the image
 * @param [in] new_height Desired height of the image
 * @param [in] mean Float value to subtract from the image pixel values
 * @param [in] std Float value to divide the image pixel values (after mean
 * subtraction)
 * @param [in] nchw Boolean flag to indicate if the image should be in NCHW or
 * NHWC format
 * @return Pointer to the resized image in float format
 */
void *load_image(const char *image_path, int new_width, int new_height,
                 float mean, float std, bool nchw);

/**
 * @brief Build the tensors struct from the input data
 * @param [in] data Pointer to the preprocessed image data
 * @param [in] height Preprocessed image height
 * @param [in] width Preprocessed image width
 * @param [in] channels Number of channels in the image
 *
 * @return Pointer to the tensors struct
 */
tensors_struct *build_tensors_struct(uint8_t *data, size_t height, size_t width,
                                     size_t channels);

/**
 * @brief Check if a string is numeric
 * @param [in] str The string to check
 * @return 1 if the string is numeric, 0 otherwise
 */
int is_numeric(const char *str);

/**
 * @brief Initialize the runtime module with the given parameters
 * @param [in] library_path Path to the runtime library
 * @param [in] num_pairs Number of key-value argument pairs
 * @param [in] arg_keys Array of argument keys
 * @param [in] arg_values Array of argument values
 * @param [in] model_path Path to the model file
 * @return Pointer to the initialized Runtime structure
 */
Runtime *init_runtime_module(const char *library_path, int num_pairs,
                             const char **arg_keys, const void **arg_values,
                             const char *model_path);

/**
 * @brief Prepare the input tensors from the given image path and preprocessing
 * parameters
 * @param [in] image_path Path to the input image
 * @param [in] width Desired width of the input image
 * @param [in] height Desired height of the input image
 * @param [in] mean Mean value for normalization
 * @param [in] std Standard deviation value for normalization
 * @param [in] nchw Boolean flag indicating whether the image should be in NCHW
 * format
 * @return Pointer to the prepared tensors_struct
 */
tensors_struct *prepare_input(const char *image_path, int width, int height,
                              float mean, float std, int nchw);

/**
 * @brief Prints the usage instructions for the program.
 *
 * This function displays a message to the user explaining how to use
 * the program, including any required arguments or options.
 *
 * @note Ensure this function is called when the user provides incorrect
 *       input or requests help.
 */
void print_usage(const char *prog_name);

/**
 * @brief Parses command-line arguments and extracts relevant information.
 *
 * This function processes the command-line arguments provided to the program,
 * interprets them, and stores the results in a structured format for further
 * use.
 *
 * @param n_required_args The number of required arguments.
 * @param argc The number of command-line arguments.
 * @param argv An array of strings representing the command-line arguments.
 * @param library_path Pointer to a string that will hold the path to the
 *                    runtime library.
 * @param model_path Pointer to a string that will hold the path to the model
 * file.
 * @param image_path Pointer to a string that will hold the path to the input
 * image.
 * @param num_inferences Pointer to an integer that will hold the number of
 * inferences to perform.
 * @param input_height Pointer to an integer that will hold the input height.
 * @param input_width Pointer to an integer that will hold the input width.
 * @param nchw Pointer to an integer that indicates whether the input is in NCHW
 * @param mean Pointer to a float that will hold the mean value for
 * normalization.\
 * @param std Pointer to a float that will hold the standard deviation value for
 * normalization.
 * @param num_pairs Pointer to an integer that will hold the number of key-value
 * pairs.
 * @param arg_keys Pointer to an array of strings that will hold the keys of the
 *                additional arguments.
 * @param arg_values Pointer to an array of void pointers that will hold the
 * values of the additional arguments.
 * @param int_values Pointer to an array of integers that will hold the integer
 * values of the additional arguments.
 * @return A status code indicating success or failure of the parsing operation.
 */
int parse_args(int n_required_args, int argc, char **argv, char **library_path,
               char **model_path, char **image_path, int *num_inferences,
               int *input_height, int *input_width, int *nchw, float *mean,
               float *std, int *num_pairs, const char ***arg_keys,
               const void ***arg_values, int **int_values);

void save_metrics_json(const char *runtime_name, const char *runtime_version,
                       const char *model_name, int input_width,
                       int input_height, float number_of_inferences,
                       float avg_throughput, float cpu_usage, float ram_usage,
                       int n_required_args, int argc, char **argv,
                       const char *json_path);

#endif  // C_EXAMPLE_INCLUDE_RUNTIME_UTILS_H_
