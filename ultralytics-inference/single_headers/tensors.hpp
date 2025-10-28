#pragma once

#include "tensors_struct.h"
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <string>

using namespace std;

// Function to create tensors from an image
tensors_struct *create_tensors(cv::Mat &image, string &input_name,
                                 bool nchw,
                                 const string &input_dtype);

// Function to print tensors metadata
void print_tensors_metadata(tensors_struct *tensors);
