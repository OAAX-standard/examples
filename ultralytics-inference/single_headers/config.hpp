#pragma once

#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

// Function to load the configuration from a JSON file
json load_config(const std::string &config_path);