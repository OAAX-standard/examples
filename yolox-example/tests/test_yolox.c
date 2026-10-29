// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

// Unit tests of the YOLOX pre/post-processing (no runtime, no model).

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yolox.h"  // NOLINT[build/include]

static int failures = 0;

#define EXPECT(condition)                                            \
  do {                                                               \
    if (!(condition)) {                                              \
      fprintf(stderr, "FAILED line %d: %s\n", __LINE__, #condition); \
      failures++;                                                    \
    }                                                                \
  } while (0)

static int near(float a, float b) { return fabsf(a - b) < 1e-3f; }

static void test_letterbox(void) {
  // A 64x32 RGB image of one color on a 32x32 canvas: scale 0.5, the image
  // fills the top half as BGR, the rest is gray (114)
  uint8_t rgb[64 * 32 * 3];
  for (int i = 0; i < 64 * 32; i++) {
    rgb[i * 3] = 10;
    rgb[i * 3 + 1] = 20;
    rgb[i * 3 + 2] = 30;
  }
  uint8_t canvas[32 * 32 * 3];
  const float scale = yolox_letterbox(rgb, 64, 32, 32, canvas);
  EXPECT(near(scale, 0.5f));
  EXPECT(canvas[0] == 30 && canvas[1] == 20 && canvas[2] == 10);
  EXPECT(canvas[(15 * 32 + 31) * 3] == 30);  // Last image pixel
  EXPECT(canvas[(16 * 32) * 3] == 114);      // First padding row
  EXPECT(canvas[(31 * 32 + 31) * 3 + 2] == 114);
}

static void test_num_anchors(void) {
  EXPECT(yolox_num_anchors(640) == 8400);  // yolox_s
  EXPECT(yolox_num_anchors(416) == 3549);  // yolox_nano, yolox_tiny
}

// Fills the row of anchor `index` with a raw model output
static void set_row(float* rows, int index, float dx, float dy, float log_w,
                    float log_h, float objectness, int class_id, float cls) {
  float* row = rows + (size_t)index * YOLOX_ROW_SIZE;
  row[0] = dx;
  row[1] = dy;
  row[2] = log_w;
  row[3] = log_h;
  row[4] = objectness;
  row[5 + class_id] = cls;
}

static void test_decode(void) {
  // A 64x64 model has 64 + 16 + 4 anchors; the first 64 have stride 8
  const int num_rows = yolox_num_anchors(64);
  EXPECT(num_rows == 84);
  float* rows =
      (float*)calloc((size_t)num_rows * YOLOX_ROW_SIZE, sizeof(float));
  Detection detections[84];

  // Anchor (gx 3, gy 2): center ((0.5 + 3) * 8, (0.5 + 2) * 8) = (28, 20),
  // size (2 * 8, 3 * 8) = (16, 24): box (20, 8) - (36, 32). Score 0.9 * 0.8.
  set_row(rows, 2 * 8 + 3, 0.5f, 0.5f, logf(2), logf(3), 0.9f, 7, 0.8f);
  // The next anchor finds the same box, same class: removed by the NMS
  set_row(rows, 2 * 8 + 4, -0.5f, 0.5f, logf(2), logf(3), 0.9f, 7, 0.7f);
  // The same box with another class: kept
  set_row(rows, 2 * 8 + 5, -1.5f, 0.5f, logf(2), logf(3), 0.9f, 8, 0.6f);
  // Below the threshold: dropped
  set_row(rows, 0, 0, 0, 0, 0, 0.9f, 1, 0.2f);

  int count =
      yolox_decode(rows, num_rows, 64, 1.0f, 100, 100, 0.3f, 0.45f, detections);
  EXPECT(count == 2);
  EXPECT(detections[0].class_id == 7 && near(detections[0].score, 0.72f));
  EXPECT(near(detections[0].x1, 20) && near(detections[0].y1, 8));
  EXPECT(near(detections[0].x2, 36) && near(detections[0].y2, 32));
  EXPECT(detections[1].class_id == 8 && near(detections[1].score, 0.54f));

  // The image was scaled by 0.5 to fit the model: boxes go back to its pixels
  // and are clipped to its 50x50 area
  count =
      yolox_decode(rows, num_rows, 64, 0.5f, 50, 50, 0.3f, 0.45f, detections);
  EXPECT(count == 2);
  EXPECT(near(detections[0].x1, 40) && near(detections[0].y1, 16));
  EXPECT(near(detections[0].x2, 50) && near(detections[0].y2, 50));

  EXPECT(yolox_decode(rows, num_rows - 1, 64, 1.0f, 100, 100, 0.3f, 0.45f,
                      detections) == -1);
  free(rows);
}

int main(void) {
  test_letterbox();
  test_num_anchors();
  test_decode();
  if (failures == 0) printf("All YOLOX tests passed\n");
  return failures == 0 ? 0 : 1;
}
