#pragma once

#include <opencv2/opencv.hpp>
#include <string>

// Enum for the resizing method
enum ResizeMethod { SQUASH, PAD };

// Struct to hold preprocessing results
struct PreprocessResult {
  cv::Mat image;
  float scale;
  int pad_x;
  int pad_y;
};

// Preprocesses the input image by resizing, normalizing, and converting its
// data type.
PreprocessResult preprocess_image(const cv::Mat &image, int input_width,
                                  int input_height, ResizeMethod method,
                                  const cv::Scalar &mean,
                                  const cv::Scalar &std);
