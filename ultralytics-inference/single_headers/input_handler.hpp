#pragma once

#include <string>
#include <vector>

enum class InputType { IMAGE, VIDEO, UNKNOWN };

InputType get_input_type(const std::string &input_path);

extern const std::vector<std::string> image_extensions;
extern const std::vector<std::string> video_extensions;
