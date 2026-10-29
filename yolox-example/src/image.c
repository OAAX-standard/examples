// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include "image.h"  // NOLINT[build/include]

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Single-header libraries in third_party/stb (public domain / MIT)
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_easy_font.h>
#include <stb/stb_image_write.h>

#define BOX_THICKNESS 3  // Pixels
#define TEXT_SCALE 2     // stb_easy_font is 7 pixels high

uint8_t* load_image(const char* path, int* width, int* height) {
  int channels;
  uint8_t* pixels = stbi_load(path, width, height, &channels, 3);
  if (pixels == NULL) {
    fprintf(stderr, "Error: cannot read image '%s': %s\n", path,
            stbi_failure_reason());
  }
  return pixels;
}

static void fill_rect(uint8_t* rgb, int width, int height, int x1, int y1,
                      int x2, int y2, const uint8_t color[3]) {
  if (x1 < 0) x1 = 0;
  if (y1 < 0) y1 = 0;
  if (x2 > width) x2 = width;
  if (y2 > height) y2 = height;
  for (int y = y1; y < y2; y++) {
    for (int x = x1; x < x2; x++) {
      for (int c = 0; c < 3; c++) rgb[(y * width + x) * 3 + c] = color[c];
    }
  }
}

// Draws white text on a colored background, its top-left corner at (x, y).
static void draw_label(uint8_t* rgb, int width, int height, int x, int y,
                       const char* text, const uint8_t color[3]) {
  static const uint8_t white[3] = {255, 255, 255};
  static char buffer[20000];  // stb_easy_font needs ~270 bytes per character
  char label[64];
  snprintf(label, sizeof(label), "%s", text);
  const int text_width = stb_easy_font_width(label) * TEXT_SCALE;
  const int text_height = stb_easy_font_height(label) * TEXT_SCALE;
  if (y - text_height - 4 >= 0) y -= text_height + 4;  // Above the box
  fill_rect(rgb, width, height, x, y, x + text_width + 4, y + text_height + 4,
            color);

  // The font is a list of axis-aligned quads (4 vertices of 16 bytes each)
  const int quads =
      stb_easy_font_print(0, 0, label, NULL, buffer, sizeof(buffer));
  for (int q = 0; q < quads; q++) {
    const float* top_left = (const float*)(buffer + q * 64);
    const float* bottom_right = (const float*)(buffer + q * 64 + 32);
    fill_rect(rgb, width, height, x + 2 + (int)(top_left[0] * TEXT_SCALE),
              y + 2 + (int)(top_left[1] * TEXT_SCALE),
              x + 2 + (int)(bottom_right[0] * TEXT_SCALE),
              y + 2 + (int)(bottom_right[1] * TEXT_SCALE), white);
  }
}

// A distinct, stable, vivid color per class (hue steps of the golden ratio)
static void class_color(int class_id, uint8_t color[3]) {
  const float hue = fmodf(class_id * 0.618034f, 1.0f) * 6;
  const float value = 0.85f, saturation = 0.9f;
  const int sector = (int)hue;
  const float f = hue - sector;
  const float p = value * (1 - saturation);
  const float q = value * (1 - saturation * f);
  const float t = value * (1 - saturation * (1 - f));
  const float rgb[6][3] = {{value, t, p}, {q, value, p}, {p, value, t},
                           {p, q, value}, {t, p, value}, {value, p, q}};
  for (int c = 0; c < 3; c++) color[c] = (uint8_t)(rgb[sector][c] * 255);
}

int save_detections_png(const char* path, uint8_t* rgb, int width, int height,
                        const Detection* detections, int count) {
  for (int i = 0; i < count; i++) {
    const Detection* d = &detections[i];
    uint8_t color[3];
    class_color(d->class_id, color);
    const int x1 = (int)d->x1, y1 = (int)d->y1, x2 = (int)d->x2,
              y2 = (int)d->y2;
    fill_rect(rgb, width, height, x1, y1, x2, y1 + BOX_THICKNESS, color);
    fill_rect(rgb, width, height, x1, y2 - BOX_THICKNESS, x2, y2, color);
    fill_rect(rgb, width, height, x1, y1, x1 + BOX_THICKNESS, y2, color);
    fill_rect(rgb, width, height, x2 - BOX_THICKNESS, y1, x2, y2, color);

    char text[64];
    snprintf(text, sizeof(text), "%s %.2f", YOLOX_CLASS_NAMES[d->class_id],
             d->score);
    draw_label(rgb, width, height, x1, y1, text, color);
  }
  const size_t length = strlen(path);
  const int is_jpeg = (length > 4 && strcmp(path + length - 4, ".jpg") == 0) ||
                      (length > 5 && strcmp(path + length - 5, ".jpeg") == 0);
  const int written =
      is_jpeg ? stbi_write_jpg(path, width, height, 3, rgb, 90)
              : stbi_write_png(path, width, height, 3, rgb, width * 3);
  if (!written) {
    fprintf(stderr, "Error: cannot write '%s'\n", path);
    return -1;
  }
  return 0;
}
