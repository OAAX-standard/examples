#pragma once

#include <spdlog/spdlog.h>
#include <string>

using namespace std;

// Function to initialize the logger
spdlog::logger initialize_logger(const string &log_file,
                                 int file_log_level = 2,
                                 int console_log_level = 2);

// Function to destroy the logger
void destroy_logger();