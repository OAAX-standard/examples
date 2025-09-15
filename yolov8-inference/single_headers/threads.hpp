#include <iostream>
#include <thread>
#include <vector>

static int number_of_received_outputs = 0;
static int max_number_of_nonprocessed_inputs = 10;
static int time_to_wait_for_output_before_sending_input = 100;  // milliseconds
static int max_number_of_consecutive_waits =
    1000;                        // Maximum consecutive waits before stopping
static int num_iterations = 10;  // Number of iterations for the routine
static bool input_thread_interrupted = false;

void send_input_tensors_routine(Runtime *runtime,
                                tensors_struct *original_tensors) {
  input_thread_interrupted = false;
  // This function would contain the logic to send input tensors to the
  // runtime For demonstration, we will just print a message
  if (!original_tensors) {
    spdlog::error("No input tensors provided to send.");
    return;
  }
  spdlog::info("Sending input tensors to the runtime...");
  int number_of_consecutive_waits = 0;
  int i = 0;
  int exit_code = 0;
  while (i < num_iterations) {
    if (i - number_of_received_outputs >= max_number_of_nonprocessed_inputs) {
      if (number_of_consecutive_waits >= max_number_of_consecutive_waits) {
        spdlog::error(
            "Too many consecutive waits without output. "
            "Stopping sending input tensors.");
        input_thread_interrupted = true;
        return;
      }
      this_thread::sleep_for(
          chrono::milliseconds(time_to_wait_for_output_before_sending_input));
      number_of_consecutive_waits++;
      continue;  // Skip sending if the limit is reached
    }
    // Deep copy the original tensors to avoid modifying them
    tensors_struct *tensors = deep_copy_tensors_struct(original_tensors);
    exit_code = runtime->send_input(tensors);
    if (exit_code != 0) {
      spdlog::warn("Failed to send input tensors: {}",
                   runtime->runtime_error_message());
      deep_free_tensors_struct(tensors);
    }
    spdlog::info("Sent input tensors: {}", i + 1);
    number_of_consecutive_waits = 0;  // Reset the wait counter
    i++;
  }
  // Simulate sending tensors
  spdlog::info("All input tensors sent successfully.");
}

void receive_output_tensors_routine(Runtime *runtime) {
  // This function would contain the logic to receive output tensors from the
  // runtime
  int number_of_consecutive_failures_to_receive_output = 0;
  int exit_code = 0;
  int i = 0;
  while (i < num_iterations) {
    if (input_thread_interrupted) {
      spdlog::error("Input thread interrupted, stopping receiving outputs.");
      return;
    }
    tensors_struct *output_tensors = nullptr;
    exit_code = runtime->receive_output(&output_tensors);
    if (exit_code != 0) {
      if (number_of_consecutive_failures_to_receive_output >= 20) {
        spdlog::error(
            "Too many consecutive failures to receive output. "
            "Stopping output receiving.");
        return;
      }
      // Sleep for a short duration before retrying
      this_thread::sleep_for(chrono::milliseconds(100));
      number_of_consecutive_failures_to_receive_output++;
      continue;  // Skip this iteration if receiving fails
    }
    // Print the received output tensors metadata
    // print_tensors_metadata(output_tensors);

    deep_free_tensors_struct(output_tensors);

    number_of_received_outputs++;
    number_of_consecutive_failures_to_receive_output =
        0;  // Reset the failure counter
    spdlog::info("Output tensors received: {}", number_of_received_outputs);
    i++;
  }
  spdlog::info("Output tensors received successfully.");
}