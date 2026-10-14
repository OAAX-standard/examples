// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include "image.h"  // NOLINT[build/include]

// clang-format off
#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>  // Jpeglib should always be included after stdio.h
// clang-format on

#define CHANNELS 3

// Decodes a JPEG into an interleaved RGB buffer. Returns NULL on failure.
static unsigned char *decode_jpeg(const char *path, int *width, int *height) {
  FILE *file = fopen(path, "rb");
  if (file == NULL) {
    fprintf(stderr, "Error: couldn't open the image file '%s'\n", path);
    return NULL;
  }

  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, file);
  jpeg_read_header(&cinfo, TRUE);
  cinfo.out_color_space = JCS_RGB;  // Also converts grayscale images
  jpeg_start_decompress(&cinfo);

  *width = (int)cinfo.output_width;
  *height = (int)cinfo.output_height;
  size_t row_size = (size_t)(*width) * CHANNELS;
  unsigned char *pixels = (unsigned char *)malloc(row_size * (size_t)(*height));
  while (pixels != NULL && cinfo.output_scanline < cinfo.output_height) {
    unsigned char *row = pixels + cinfo.output_scanline * row_size;
    jpeg_read_scanlines(&cinfo, &row, 1);
  }

  if (pixels != NULL) jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(file);
  return pixels;
}

float *load_image(const char *image_path, int new_width, int new_height,
                  float mean, float std, bool nchw) {
  int width = 0;
  int height = 0;
  unsigned char *pixels = decode_jpeg(image_path, &width, &height);
  if (pixels == NULL) return NULL;

  size_t plane = (size_t)new_width * new_height;
  float *output = (float *)malloc(plane * CHANNELS * sizeof(float));
  if (output == NULL) {
    free(pixels);
    return NULL;
  }

  // Nearest neighbor resize + normalization, written in the requested layout
  for (int y = 0; y < new_height; y++) {
    for (int x = 0; x < new_width; x++) {
      const unsigned char *src =
          pixels + ((size_t)(y * height / new_height) * width +
                    (size_t)(x * width / new_width)) *
                       CHANNELS;
      for (int c = 0; c < CHANNELS; c++) {
        size_t index = nchw ? c * plane + (size_t)y * new_width + x
                            : ((size_t)y * new_width + x) * CHANNELS + c;
        output[index] = ((float)src[c] - mean) / std;
      }
    }
  }

  free(pixels);
  return output;
}
