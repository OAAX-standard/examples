// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#ifndef YOLOX_EXAMPLE_INCLUDE_IMAGE_H_
#define YOLOX_EXAMPLE_INCLUDE_IMAGE_H_

#include <stdint.h>

#include "yolox.h"  // NOLINT[build/include]

/**
 * @brief Load a JPEG, PNG or BMP image as RGB, 8 bits per channel.
 * @note EXIF rotation is ignored.
 * @return The pixels (free with `free`), or NULL (an error is printed)
 */
uint8_t* load_image(const char* path, int* width, int* height);

/**
 * @brief Draw a box and a "class score" label for each detection on an RGB
 * image and save it as a PNG (or a JPEG if `path` ends with .jpg or .jpeg).
 * @return 0 on success, -1 on failure (an error is printed)
 */
int save_detections_png(const char* path, uint8_t* rgb, int width, int height,
                        const Detection* detections, int count);

#endif  // YOLOX_EXAMPLE_INCLUDE_IMAGE_H_
