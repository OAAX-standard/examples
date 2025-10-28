#include "single_headers/config.hpp"
#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>

// Function to load the configuration from a JSON file
json load_config(const std::string &config_path) {
  std::ifstream config_file(config_path);
  if (!config_file.is_open()) {
    spdlog::error("Failed to open config file: {}", config_path);
    exit(EXIT_FAILURE);
  }

  json config;
  config_file >> config;

  return config; // Return the loaded JSON configuration
}
