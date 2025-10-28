#include "single_headers/threads.hpp"
#include <chrono>

using namespace std;

void send_input_tensors_routine(Runtime *runtime,
                                tensors_struct *tensors) {
  if (!tensors) {
    spdlog::error("No input tensors provided to send.");
    return;
  }
  spdlog::info("Sending input tensors to the runtime...");
  
  int max_retries = 3;
  for (int attempt = 1; attempt <= max_retries; attempt++) {
    int exit_code = runtime->send_input(tensors);
    if (exit_code == 0) {
      spdlog::info("Input tensors sent successfully.");
      return;
    }
    spdlog::warn("Failed to send input tensors (attempt {}/{}): {}",
                 attempt, max_retries, runtime->runtime_error_message());
    if (attempt < max_retries) {
      this_thread::sleep_for(chrono::milliseconds(100));
    }
  }
  spdlog::error("Failed to send input tensors after {} attempts.", max_retries);
}

void receive_output_tensors_routine(Runtime *runtime,
                                    tensors_struct **output_tensors) {
  int max_retries = 3;
  for (int attempt = 1; attempt <= max_retries; attempt++) {
    int exit_code = runtime->receive_output(output_tensors);
    if (exit_code == 0) {
      spdlog::info("Output tensors received successfully.");
      return;
    }
    spdlog::warn("Failed to receive output tensors (attempt {}/{}): {}",
                 attempt, max_retries, runtime->runtime_error_message());
    if (attempt < max_retries) {
      this_thread::sleep_for(chrono::milliseconds(100));
    }
  }
  spdlog::error("Failed to receive output tensors after {} attempts.", max_retries);
  *output_tensors = nullptr;
}
