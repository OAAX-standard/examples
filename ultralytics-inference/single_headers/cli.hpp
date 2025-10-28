#pragma once

#include <CLI/CLI.hpp>
#include <string>
#include <vector>

using namespace std;

// Utility function to parse command line arguments
// This function uses the CLI11 library to handle command line options
int parse_command_line(int argc, char **argv, string &library_path,
                       string &model_path, vector<string> &input_paths,
                       string &config_path, string &log_file, int &log_level);