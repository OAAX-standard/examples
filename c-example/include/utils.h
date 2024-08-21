#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <jpeglib.h>
#include "runtime_utils.h"

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

/**
 * @brief Print information about the output tensors
 * @param [in] output_tensors Pointer to the output tensors struct
 */
void print_output_tensors(tensors_struct *output_tensors);

/**
 * @brief Free the tensors struct
 * @param [in] tensors Pointer to the tensors struct
 */
void free_tensors_struct(tensors_struct *tensors);