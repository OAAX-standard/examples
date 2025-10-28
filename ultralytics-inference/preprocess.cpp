#include "single_headers/preprocess.hpp"
#include <opencv2/opencv.hpp>
#include <stdexcept>

// Preprocesses the input image by resizing, normalizing, and converting its
// data type.
PreprocessResult preprocess_image(const cv::Mat &image, int input_width,
                                  int input_height, ResizeMethod method,
                                  const cv::Scalar &mean,
                                  const cv::Scalar &std) {
  // Validate that image is not empty
  if (image.empty()) {
    throw std::runtime_error("Input image is empty");
  }

  PreprocessResult result;
  result.pad_x = 0;
  result.pad_y = 0;

  // Resize the image
  cv::Mat resized_image;
  if (method == SQUASH) {
    result.scale = static_cast<float>(input_width) / image.cols;
    cv::resize(image, resized_image, cv::Size(input_width, input_height), 0, 0,
               cv::INTER_LINEAR);
  } else if (method == PAD) {
    // Letterbox resizing
    result.scale = std::min(static_cast<float>(input_width) / image.cols,
                          static_cast<float>(input_height) / image.rows);
    int new_unpad_w = static_cast<int>(round(image.cols * result.scale));
    int new_unpad_h = static_cast<int>(round(image.rows * result.scale));
    cv::resize(image, resized_image, cv::Size(new_unpad_w, new_unpad_h), 0, 0,
               cv::INTER_LINEAR);

    // Pad the image to the target size
    result.pad_y = static_cast<int>(round((input_height - new_unpad_h) / 2.0));
    result.pad_x = static_cast<int>(round((input_width - new_unpad_w) / 2.0));
    cv::copyMakeBorder(resized_image, resized_image, result.pad_y, result.pad_y,
                       result.pad_x, result.pad_x, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
  }
  
  // Normalize the image
  resized_image.convertTo(resized_image, CV_32F);
  resized_image -= mean;
  resized_image /= std;

  result.image = resized_image;
  return result;
}
