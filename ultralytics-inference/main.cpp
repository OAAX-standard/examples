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
#include "input_handler.hpp"
#include "postprocess.hpp"
#include "visualize.hpp"

int main(int argc, char **argv) {
  string library_path, model_path, log_file, config_path;
  vector<string> input_paths;
  int log_level;
  // Parse command line arguments
  int response =
      parse_command_line(argc, argv, library_path, model_path, input_paths,
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
  for (const auto& input_path : input_paths) {
    logger.info("Input Path: {}", input_path);
  }
  logger.info("Configuration Path: {}", config_path);
  logger.info("Log File: {}", log_file);
  logger.info("Log Level: {}", log_level);

  // Load the runtime library
  Runtime *runtime = load_runtime_library(library_path);
  // Log the runtime name and version
  logger.info("Runtime Name: {}", runtime->runtime_name());
  logger.info("Runtime Version: {}", runtime->runtime_version());

  // Initialize the runtime
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

  // Load the model
  exit_code = runtime->runtime_model_loading(model_path.c_str());
  if (exit_code != 0) {
    logger.error("Model loading failed: {}", runtime->runtime_error_message());
    destroy_runtime(runtime);
    return EXIT_FAILURE;
  }
  logger.info("Model loaded successfully: {}", model_path);

  // Load the configuration file once
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

  // Process the input files
  for (const auto& input_path : input_paths) {
    InputType input_type = get_input_type(input_path);
    // Process the input file based on its type
    switch (input_type) {
      case InputType::IMAGE: {
        logger.info("Processing image: {}", input_path);
        // Load the input image
        cv::Mat original_image = cv::imread(input_path);
        if (original_image.empty()) {
          logger.error("Failed to read image: {}", input_path);
          destroy_runtime(runtime);
          return EXIT_FAILURE;
        }
        // Convert image from BGR to RGB
        cv::cvtColor(original_image, original_image, cv::COLOR_BGR2RGB);
        // Preprocess the input image
        PreprocessResult prep_result = preprocess_image(
            original_image, config["model"]["input_width"].get<int>(),
            config["model"]["input_height"].get<int>(),
            PAD,  // Use PAD as the desired method
            cv::Scalar(config["model"]["mean"][0].get<float>(),
                       config["model"]["mean"][1].get<float>(),
                       config["model"]["mean"][2].get<float>()),
            cv::Scalar(config["model"]["std"][0].get<float>(),
                       config["model"]["std"][1].get<float>(),
                       config["model"]["std"][2].get<float>()));
        // Create the input tensors
        string input_name = config["model"]["input_name"].get<string>();
        tensors_struct* tensors;
        tensors = create_tensors(
              prep_result.image, input_name, config["model"]["nchw"].get<int>(),
              config["model"]["input_dtype"].get<string>());

        if (!tensors) {
          logger.error("Failed to create input tensors.");
          destroy_runtime(runtime);
          return EXIT_FAILURE;
        }
        print_tensors_metadata(tensors);

        spdlog::info("Starting input sending and output receiving...");
        // Send input tensors directly
        send_input_tensors_routine(runtime, tensors);
        // Receive output tensors directly
        tensors_struct *output_tensors = nullptr;
        receive_output_tensors_routine(runtime, &output_tensors);

        spdlog::info("Input/output operations completed successfully.");
        
        // Post-process the output
        std::vector<Detection> detections;
        if(output_tensors){
          detections = parse_yolo_output(output_tensors, config["postprocessing"]["confidence_threshold"].get<float>(), prep_result, original_image.size());
          original_image.release();
        }


        std::vector<Detection> nms_detections =
            non_maximum_suppression(detections, config["postprocessing"]["iou_threshold"].get<float>());

        logger.info("Detections after NMS: {}", nms_detections.size());

        // Visualize the detections
        cv::Mat output_image = cv::imread(input_path);
        std::vector<std::string> class_names = config["postprocessing"]["class_names"].get<std::vector<std::string>>();
        draw_detections(output_image, nms_detections, class_names);

        // Save the output image
        std::string output_path = "output_" + input_path.substr(input_path.find_last_of('/') + 1);
        cv::imwrite(output_path, output_image);
        logger.info("Saved output image to: {}", output_path);

        // Clean up resources
        logger.info("Terminating OAAX inference engine.");
        deep_free_tensors_struct(tensors);
        break;
      }
      case InputType::VIDEO: {
        logger.info("Processing video: {}", input_path);

        cv::VideoCapture cap(input_path);
        if (!cap.isOpened()) {
          logger.error("Failed to open video: {}", input_path);
          break;
        }

        // Get video properties
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);

        // Create video writer
        std::string output_path = "output_" + input_path.substr(input_path.find_last_of('/') + 1);
        cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                               fps, cv::Size(frame_width, frame_height));

        cv::Mat frame;
        while (cap.read(frame)) {
          // Convert frame from BGR to RGB
          cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
          // Preprocess the frame
          PreprocessResult prep_result = preprocess_image(
              frame, config["model"]["input_width"].get<int>(),
              config["model"]["input_height"].get<int>(),
              PAD, // Use PAD as the desired method
              cv::Scalar(config["model"]["mean"][0].get<float>(),
                         config["model"]["mean"][1].get<float>(),
                         config["model"]["mean"][2].get<float>()),
              cv::Scalar(config["model"]["std"][0].get<float>(),
                         config["model"]["std"][1].get<float>(),
                         config["model"]["std"][2].get<float>()));

          // Create input tensors
          string input_name = config["model"]["input_name"].get<string>();
          tensors_struct* tensors;
          tensors = create_tensors(
                prep_result.image, input_name, config["model"]["nchw"].get<int>(),
                config["model"]["input_dtype"].get<string>());

          // Run inference
          tensors_struct *output_tensors = nullptr;
          thread input_thread(send_input_tensors_routine, runtime, tensors);
          thread output_thread(receive_output_tensors_routine, runtime, &output_tensors);
          input_thread.join();
          output_thread.join();

          // Post-process and visualize
          std::vector<Detection> detections;
          if(output_tensors){
            detections = parse_yolo_output(output_tensors, config["postprocessing"]["confidence_threshold"].get<float>(), prep_result, frame.size());
          }

          std::vector<Detection> nms_detections =
              non_maximum_suppression(detections, config["postprocessing"]["iou_threshold"].get<float>());
          
          std::vector<std::string> class_names = config["postprocessing"]["class_names"].get<std::vector<std::string>>();
          draw_detections(frame, nms_detections, class_names);

          writer.write(frame);
          deep_free_tensors_struct(tensors);
        }

        cap.release();
        writer.release();

        logger.info("Saved output video to: {}", output_path);
        break;
      }
      case InputType::UNKNOWN: {
        logger.warn("Unknown input type for {}. Skipping.", input_path);
        break;
      }
    }
  }

  // Destroy the runtime
  destroy_runtime(runtime);

  // Destroy the logger
  destroy_logger();

  return 0;
}
