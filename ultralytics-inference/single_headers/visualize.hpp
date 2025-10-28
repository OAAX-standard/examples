#pragma once

#include "postprocess.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Function to draw detections on an image
void draw_detections(cv::Mat &image,
                     const std::vector<Detection> &detections,
                     const std::vector<std::string> &class_names);

// Function to get a color for a given class ID
cv::Scalar get_class_color(int class_id);
