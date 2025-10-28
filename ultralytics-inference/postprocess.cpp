#include "single_headers/postprocess.hpp"
#include "single_headers/preprocess.hpp"
#include "tensors_struct.h"
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>

// Function to calculate Intersection over Union (IoU)
float calculate_iou(const cv::Rect &box1, const cv::Rect &box2) {
  float x1 = std::max(box1.x, box2.x);
  float y1 = std::max(box1.y, box2.y);
  float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
  float y2 = std::min(box1.y + box1.height, box2.y + box2.height);

  float intersection_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  float union_area =
      box1.width * box1.height + box2.width * box2.height - intersection_area;

  return intersection_area / union_area;
}

// Function to parse YOLOv8 output
std::vector<Detection>
parse_yolo_output(tensors_struct *tensors, float confidence_threshold,
                  const PreprocessResult &prep_result,
                  const cv::Size &original_image_size) {
  std::vector<Detection> detections;
  if (tensors == nullptr || tensors->num_tensors == 0) {
    return detections;
  }

  // Assuming the first tensor is the output
  float *data = static_cast<float *>(tensors->data[0]);

  // The output shape is (1, 84, 2100) for a 320x320 model
  int num_proposals = tensors->shapes[0][2];
  int num_classes_with_box = tensors->shapes[0][1];
  int num_classes = num_classes_with_box - 4;

  // Transpose the data from (1, 84, 2100) to (1, 2100, 84)
  std::vector<float> transposed_data(num_proposals * num_classes_with_box);
  for (int i = 0; i < num_classes_with_box; ++i) {
    for (int j = 0; j < num_proposals; ++j) {
      transposed_data[j * num_classes_with_box + i] = data[i * num_proposals + j];
    }
  }
  
  // Process each proposal
  for (int i = 0; i < num_proposals; ++i) {
    float *proposal_data = &transposed_data[i * num_classes_with_box];
    float *class_scores = proposal_data + 4;

    // Find the class with the highest score
    int best_class_id = -1;
    float max_score = 0.0;
    for (int j = 0; j < num_classes; ++j) {
      if (class_scores[j] > max_score) {
        max_score = class_scores[j];
        best_class_id = j;
      }
    }

    // If the score is above the threshold, create a detection
    if (max_score > confidence_threshold) {
      float cx = proposal_data[0];
      float cy = proposal_data[1];
      float w = proposal_data[2];
      float h = proposal_data[3];

      // Scale the box back to the original image size
      float x = (cx - prep_result.pad_x - (w / 2.0f)) / prep_result.scale;
      float y = (cy - prep_result.pad_y - (h / 2.0f)) / prep_result.scale;
      float width = w / prep_result.scale;
      float height = h / prep_result.scale;
      
      detections.push_back({cv::Rect(static_cast<int>(x), static_cast<int>(y),
                                     static_cast<int>(width), static_cast<int>(height)),
                            best_class_id, max_score});
    }
  }

  return detections;
}

// Function to perform Non-Maximum Suppression
std::vector<Detection> non_maximum_suppression(
    const std::vector<Detection> &detections, float iou_threshold) {
  std::vector<Detection> nms_detections;
  if (detections.empty()) {
    return nms_detections;
  }

  // Sort detections by score in descending order
  std::vector<Detection> sorted_detections = detections;
  std::sort(sorted_detections.begin(), sorted_detections.end(),
            [](const Detection &a, const Detection &b) {
              return a.score > b.score;
            });

  // Perform NMS
  while (!sorted_detections.empty()) {
    nms_detections.push_back(sorted_detections[0]);
    // Create a new vector to store the remaining detections
    std::vector<Detection> remaining_detections;
    for (size_t i = 1; i < sorted_detections.size(); ++i) {
        if (calculate_iou(nms_detections.back().box, sorted_detections[i].box) <= iou_threshold) {
            remaining_detections.push_back(sorted_detections[i]);
        }
    }
    sorted_detections = remaining_detections;
  }

  return nms_detections;
}

