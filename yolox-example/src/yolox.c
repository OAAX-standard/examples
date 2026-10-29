// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include "yolox.h"  // NOLINT[build/include]

#include <math.h>
#include <stdlib.h>
#include <string.h>

// COCO class names, in the order of the model's class scores
const char* const YOLOX_CLASS_NAMES[YOLOX_NUM_CLASSES] = {
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush",
};

#define NUM_STRIDES 3
static const int kStrides[NUM_STRIDES] = {8, 16, 32};
#define PAD_VALUE 114  // Gray, as in the official demo

static float clampf(float value, float low, float high) {
  return value < low ? low : (value > high ? high : value);
}

float yolox_letterbox(const uint8_t* rgb, int width, int height, int size,
                      uint8_t* bgr_canvas) {
  memset(bgr_canvas, PAD_VALUE, (size_t)size * size * 3);

  // Same as cv2.resize(..., INTER_LINEAR): pixel centers, no antialiasing
  const double ratio = fmin((double)size / height, (double)size / width);
  const int new_width = (int)fmax(1.0, width * ratio);
  const int new_height = (int)fmax(1.0, height * ratio);
  const double step_x = (double)width / new_width;
  const double step_y = (double)height / new_height;

  for (int y = 0; y < new_height; y++) {
    const double src_y = fmax(0.0, (y + 0.5) * step_y - 0.5);
    const int y0 = (int)fmin(src_y, height - 1);
    const int y1 = y0 + 1 < height ? y0 + 1 : y0;
    const double wy = src_y - y0;
    for (int x = 0; x < new_width; x++) {
      const double src_x = fmax(0.0, (x + 0.5) * step_x - 0.5);
      const int x0 = (int)fmin(src_x, width - 1);
      const int x1 = x0 + 1 < width ? x0 + 1 : x0;
      const double wx = src_x - x0;
      for (int c = 0; c < 3; c++) {
        const double top = rgb[(y0 * width + x0) * 3 + c] * (1 - wx) +
                           rgb[(y0 * width + x1) * 3 + c] * wx;
        const double bottom = rgb[(y1 * width + x0) * 3 + c] * (1 - wx) +
                              rgb[(y1 * width + x1) * 3 + c] * wx;
        const double value = top * (1 - wy) + bottom * wy;
        bgr_canvas[(y * size + x) * 3 + (2 - c)] = (uint8_t)(value + 0.5);
      }
    }
  }
  return (float)ratio;
}

int yolox_num_anchors(int size) {
  int count = 0;
  for (int s = 0; s < NUM_STRIDES; s++) {
    count += (size / kStrides[s]) * (size / kStrides[s]);
  }
  return count;
}

static int compare_scores(const void* a, const void* b) {
  const float score_a = ((const Detection*)a)->score;
  const float score_b = ((const Detection*)b)->score;
  return (score_a < score_b) - (score_a > score_b);  // Descending
}

static float iou(const Detection* a, const Detection* b) {
  const float width = fminf(a->x2, b->x2) - fmaxf(a->x1, b->x1);
  const float height = fminf(a->y2, b->y2) - fmaxf(a->y1, b->y1);
  if (width <= 0 || height <= 0) return 0;
  const float inter = width * height;
  const float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
  const float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
  return inter / (area_a + area_b - inter);
}

int yolox_decode(const float* rows, int num_rows, int size, float scale,
                 int width, int height, float conf, float nms,
                 Detection* detections) {
  if (num_rows != yolox_num_anchors(size)) return -1;

  // 1. Decode every anchor that is confident enough
  int count = 0;
  const float* row = rows;
  for (int s = 0; s < NUM_STRIDES; s++) {
    const int stride = kStrides[s];
    const int grid = size / stride;
    for (int gy = 0; gy < grid; gy++) {
      for (int gx = 0; gx < grid; gx++, row += YOLOX_ROW_SIZE) {
        int best = 0;
        for (int c = 1; c < YOLOX_NUM_CLASSES; c++) {
          if (row[5 + c] > row[5 + best]) best = c;
        }
        const float score = row[4] * row[5 + best];
        if (!(score >= conf)) continue;  // Also drops NaN

        const float cx = (row[0] + gx) * stride;
        const float cy = (row[1] + gy) * stride;
        const float w = expf(row[2]) * stride;
        const float h = expf(row[3]) * stride;
        Detection* d = &detections[count++];
        d->x1 = clampf((cx - w / 2) / scale, 0, (float)width);
        d->y1 = clampf((cy - h / 2) / scale, 0, (float)height);
        d->x2 = clampf((cx + w / 2) / scale, 0, (float)width);
        d->y2 = clampf((cy + h / 2) / scale, 0, (float)height);
        d->score = score;
        d->class_id = best;
      }
    }
  }

  // 2. Class-aware NMS: best first, drop what overlaps a kept box of its class
  qsort(detections, (size_t)count, sizeof(Detection), compare_scores);
  int kept = 0;
  for (int i = 0; i < count; i++) {
    int suppressed = 0;
    for (int j = 0; j < kept && !suppressed; j++) {
      suppressed = detections[j].class_id == detections[i].class_id &&
                   iou(&detections[j], &detections[i]) > nms;
    }
    if (!suppressed) detections[kept++] = detections[i];
  }
  return kept;
}
