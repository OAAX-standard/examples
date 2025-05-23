#ifndef INTERFACE_H
#define INTERFACE_H

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <jpeglib.h>

#include "tensors_struct.h"

typedef struct Runtime
{
    // Runtime interface functions
    int (*runtime_initialization)(); // Default initialization function
    int (*runtime_initialization_with_args)(int length, const char **keys, const void **values);
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
void resize_image(const unsigned char *image,
                  int width,
                  int height,
                  int new_width,
                  int new_height,
                  float *resized_image);

/**
 * @brief Load the image from the file path, resize it, normalize it and convert it to float.
 * @param [in] image_path Path to the image file
 * @param [in] new_width Desired width of the image
 * @param [in] new_height Desired height of the image
 * @param [in] mean Float value to subtract from the image pixel values
 * @param [in] std Float value to divide the image pixel values (after mean subtraction)
 * @param [in] nchw Boolean flag to indicate if the image should be in NCHW or NHWC format
 * @return Pointer to the resized image in float format
 */
void *load_image(const char *image_path, int new_width, int new_height, float mean, float std, bool nchw);

/**
 * @brief Build the tensors struct from the input data
 * @param [in] data Pointer to the preprocessed image data
 * @param [in] height Preprocessed image height
 * @param [in] width Preprocessed image width
 * @param [in] channels Number of channels in the image
 *
 * @return Pointer to the tensors struct
 */
tensors_struct * build_tensors_struct(uint8_t *data, size_t height, size_t width, size_t channels);


#endif // INTERFACE_H