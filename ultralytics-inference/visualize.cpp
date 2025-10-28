#include "single_headers/visualize.hpp"
#include "single_headers/postprocess.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Function to get a color for a given class ID
cv::Scalar get_class_color(int class_id) {
  cv::Scalar color;
  switch (class_id % 10) {
  case 0:
    color = cv::Scalar(255, 0, 0); // Red
    break;
  case 1:
    color = cv::Scalar(0, 255, 0); // Green
    break;
  case 2:
    color = cv::Scalar(0, 0, 255); // Blue
    break;
  case 3:
    color = cv::Scalar(255, 255, 0); // Yellow
    break;
  case 4:
    color = cv::Scalar(0, 255, 255); // Cyan
    break;
  case 5:
    color = cv::Scalar(255, 0, 255); // Magenta
    break;
  case 6:
    color = cv::Scalar(128, 0, 0); // Maroon
    break;
  case 7:
    color = cv::Scalar(0, 128, 0); // Olive
    break;
  case 8:
    color = cv::Scalar(0, 0, 128); // Navy
    break;
  case 9:
    color = cv::Scalar(128, 128, 0); // Teal
    break;
  default:
    color = cv::Scalar(0, 0, 0); // Black
    break;
  }
  return color;
}

// Function to draw detections on an image
void draw_detections(cv::Mat &image,
                     const std::vector<Detection> &detections,
                     const std::vector<std::string> &class_names) {
  for (const auto &detection : detections) {
    if (detection.class_id >= class_names.size()) {
        continue;
    }
    cv::Scalar color = get_class_color(detection.class_id);
    cv::rectangle(image, detection.box, color, 2);

    std::string label = class_names[detection.class_id] + " " +
                        std::to_string(detection.score).substr(0, 4);
    int baseline;
    cv::Size label_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::rectangle(
        image,
        cv::Point(detection.box.x, detection.box.y - label_size.height - 10),
        cv::Point(detection.box.x + label_size.width, detection.box.y),
        color, cv::FILLED);
    cv::putText(image, label, cv::Point(detection.box.x, detection.box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  }
}

