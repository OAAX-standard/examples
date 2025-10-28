#include <algorithm>
#include <string>
#include <vector>

#include "input_handler.hpp"

const std::vector<std::string> image_extensions = {".jpg", ".jpeg", ".png", ".bmp"};
const std::vector<std::string> video_extensions = {".mp4", ".avi", ".mkv", ".mov"};

InputType get_input_type(const std::string& input_path) {
  std::string extension = input_path.substr(input_path.find_last_of("."));
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 ::tolower);

  for (const auto& img_ext : image_extensions) {
    if (extension == img_ext) {
      return InputType::IMAGE;
    }
  }

  for (const auto& vid_ext : video_extensions) {
    if (extension == vid_ext) {
      return InputType::VIDEO;
    }
  }

  return InputType::UNKNOWN;
}
