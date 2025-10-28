#pragma once

#include "preprocess.hpp"
#include "tensors_struct.h"
#include <opencv2/opencv.hpp>
#include <vector>

// Struct to hold detection results
struct Detection {
  cv::Rect box;
  int class_id;
  float score;
};

// Function to parse YOLOv8 output
std::vector<Detection> parse_yolo_output(
    tensors_struct *tensors, float confidence_threshold,
    const PreprocessResult &prep_result, const cv::Size &original_image_size);

// Function to perform Non-Maximum Suppression
std::vector<Detection> non_maximum_suppression(
    const std::vector<Detection> &detections, float iou_threshold);
