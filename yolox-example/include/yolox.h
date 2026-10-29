// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#ifndef YOLOX_EXAMPLE_INCLUDE_YOLOX_H_
#define YOLOX_EXAMPLE_INCLUDE_YOLOX_H_

#include <stddef.h>
#include <stdint.h>

// The official YOLOX ONNX export (yolox_s.onnx, yolox_tiny.onnx, ...) takes a
// letterboxed image and returns one row per anchor: dx, dy, log(w), log(h),
// objectness and 80 COCO class scores (85 floats). This file holds everything
// YOLOX-specific, written like the official ONNX demo; it knows nothing about
// OAAX.

#define YOLOX_NUM_CLASSES 80
#define YOLOX_ROW_SIZE (5 + YOLOX_NUM_CLASSES)

extern const char* const YOLOX_CLASS_NAMES[YOLOX_NUM_CLASSES];

typedef struct Detection {
  float x1, y1, x2, y2;  // Pixels of the original image
  float score;           // Objectness x class score
  int class_id;
} Detection;

/**
 * @brief Resize an image to fit in `size` x `size` (keeping the aspect ratio,
 * bilinear) and put it in the top-left corner of a canvas filled with 114.
 * @param [in] rgb Image, `width` x `height` pixels, RGB, 8 bits
 * @param [out] bgr_canvas `size` x `size` x 3 bytes, BGR, filled in
 * @return The scale applied to the image
 */
float yolox_letterbox(const uint8_t* rgb, int width, int height, int size,
                      uint8_t* bgr_canvas);

/**
 * @brief Number of anchors (rows) of a model input of `size` x `size`.
 */
int yolox_num_anchors(int size);

/**
 * @brief Turn the raw model output into detections.
 *
 * Decodes the boxes (grid + stride, strides 8, 16 and 32), keeps the best class
 * of each anchor when objectness x class score >= `conf`, runs a class-aware
 * NMS with the IoU threshold `nms` and maps the boxes back to the original
 * image (`scale` from yolox_letterbox, clipped to `width` x `height`).
 * @param [in] rows `num_rows` x YOLOX_ROW_SIZE floats
 * @param [out] detections Room for `num_rows` detections, best first
 * @return The number of detections, or -1 if `num_rows` does not match `size`
 */
int yolox_decode(const float* rows, int num_rows, int size, float scale,
                 int width, int height, float conf, float nms,
                 Detection* detections);

#endif  // YOLOX_EXAMPLE_INCLUDE_YOLOX_H_
