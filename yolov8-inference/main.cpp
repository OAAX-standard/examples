#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "lib_loader.h"
#include "tensors_struct.h"

using namespace std;

#include "cli.hpp"
#include "config.hpp"
#include "logger.hpp"
#include "preprocess.hpp"
#include "runtime.hpp"
#include "tensors.hpp"
#include "threads.hpp"

int main(int argc, char **argv) {
  string library_path, model_path, input_path, log_file, config_path;
  int log_level;
  // Parse command line arguments
  int response =
      parse_command_line(argc, argv, library_path, model_path, input_path,
                         config_path, log_file, log_level);
  if (response != 0) {
    cerr << "Error parsing command line arguments.\n";
    return response;
  }
  // Initialize the logger
  auto logger = initialize_logger(log_file, log_level, log_level);

  // Log the initialization
  logger.info(
      "Initializing OAAX inference engine with the following "
      "parameters:");
  logger.info("Library Path: {}", library_path);
  logger.info("Model Path: {}", model_path);
  logger.info("Input Path: {}", input_path);
  logger.info("Configuration Path: {}", config_path);
  logger.info("Log File: {}", log_file);
  logger.info("Log Level: {}", log_level);

  // Load the runtime library
  Runtime *runtime = load_runtime_library(library_path);
  // Log the runtime name and version
  logger.info("Runtime Name: {}", runtime->runtime_name());
  logger.info("Runtime Version: {}", runtime->runtime_version());

  // Initialize the runtime and load the model
  int exit_code;
  const char *args[] = {"log_level"};
  const void *args_values[] = {"2"};  // Set log level to info
  exit_code = runtime->runtime_initialization_with_args(1, args, args_values);

  if (exit_code != 0) {
    logger.error("Runtime initialization failed: {}",
                 runtime->runtime_error_message());
    destroy_runtime(runtime);
    return EXIT_FAILURE;
  }
  logger.info("Runtime initialized successfully.");
  exit_code = runtime->runtime_model_loading(model_path.c_str());
  if (exit_code != 0) {
    logger.error("Model loading failed: {}", runtime->runtime_error_message());
    destroy_runtime(runtime);
    return EXIT_FAILURE;
  }
  logger.info("Model loaded successfully: {}", model_path);

  // Load the configuration file
  json config = load_config(config_path);
  // Log the configuration parameters
  logger.info("Configuration: {}", config.dump(4));

  // Ensure that mean and std are 3-element vectors
  if (config["model"]["mean"].size() != 3 ||
      config["model"]["std"].size() != 3) {
    logger.error("Mean and std must be 3-element vectors.");
    destroy_runtime(runtime);
    return EXIT_FAILURE;
  }

  // Preprocess the input image
  cv::Mat image =
      preprocess_image(input_path, config["model"]["input_width"].get<int>(),
                       config["model"]["input_height"].get<int>(),
                       SQUASH,  // Use SQUASH as the desired method
                       cv::Scalar(config["model"]["mean"][0].get<float>(),
                                  config["model"]["mean"][1].get<float>(),
                                  config["model"]["mean"][2].get<float>()),
                       cv::Scalar(config["model"]["std"][0].get<float>(),
                                  config["model"]["std"][1].get<float>(),
                                  config["model"]["std"][2].get<float>()));
  // Create the input tensors
  string input_name = config["model"]["input_name"].get<string>();
  tensors_struct *tensors;
  if (config["model"].contains("input_dtype")) {
    tensors =
        create_tensors(image, input_name, config["model"]["nchw"].get<int>(),
                       config["model"]["input_dtype"].get<string>());
  } else {
    tensors =
        create_tensors(image, input_name, config["model"]["nchw"].get<int>());
  }
  image.release();

  if (!tensors) {
    logger.error("Failed to create input tensors.");
    destroy_runtime(runtime);
    return EXIT_FAILURE;
  }
  print_tensors_metadata(tensors);

  spdlog::info("Starting input sending and output receiving threads...");
  // Start the input sending thread
  thread input_thread(send_input_tensors_routine, runtime, tensors);
  // Start the output receiving thread
  thread output_thread(receive_output_tensors_routine, runtime);
  // Wait for the threads to finish
  spdlog::info("Waiting for threads to finish...");
  input_thread.join();
  output_thread.join();

  spdlog::info("Threads finished successfully.");

  // Clean up resources
  logger.info("Terminating OAAX inference engine.");
  // Free the input tensors
  deep_free_tensors_struct(tensors);

  // Destroy the runtime
  destroy_runtime(runtime);

  // Destroy the logger
  destroy_logger();

  return 0;
}
