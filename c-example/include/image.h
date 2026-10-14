// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#ifndef C_EXAMPLE_INCLUDE_IMAGE_H_
#define C_EXAMPLE_INCLUDE_IMAGE_H_

#include <stdbool.h>

/**
 * @brief Load a JPEG image, resize it (nearest neighbor), normalize it and
 * convert it to float.
 * @param [in] image_path Path to the (RGB) JPEG file
 * @param [in] new_width Desired width of the image
 * @param [in] new_height Desired height of the image
 * @param [in] mean Value subtracted from every pixel
 * @param [in] std Value every pixel is divided by (after mean subtraction)
 * @param [in] nchw True for NCHW layout, false for NHWC
 * @return Heap-allocated float array of 3 * new_width * new_height elements
 * that the caller must free, or NULL on failure
 */
float *load_image(const char *image_path, int new_width, int new_height,
                  float mean, float std, bool nchw);

#endif  // C_EXAMPLE_INCLUDE_IMAGE_H_
