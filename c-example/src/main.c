// Description: This is an example of a C program that uses the OAAX runtime to send inputs and receive outputs in separate threads.
#include "runtime_utils.h"

// C utilities
#include "timer.h"
#include "memory.h"
#include "logger.h"
#include "utils.h"
#include "threading.h"

// Standard libraries
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

// Number of inferences to perform
// TODO: Adjust this as you see fit
#define NUMBER_OF_INFERENCES 10

// Logger
Logger *logger = NULL;

// Global variable to hold the original input tensors
tensors_struct *original_input_tensors = NULL;

// Thread function for sending inputs
void *send_input_thread(void *arg)
{
    Runtime *runtime = (Runtime *)arg;
    int code = 0;

    for (int i = 0; i < NUMBER_OF_INFERENCES; i++)
    {
        // Deep copy the input tensors
        tensors_struct *input_tensors = deep_copy_tensors_struct(original_input_tensors);
        if (input_tensors == NULL)
        {
            log_error(logger, "Failed to deep copy input tensors.");
            continue;
        }

        // Send the input tensors
        code = runtime->send_input(input_tensors);
        if (code != 0)
        {
            log_error(logger, "Failed to send input tensors.");
            deep_free_tensors_struct(input_tensors); // Free before returning
            return NULL;
        }

        // Ownership of input_tensors is transferred to the runtime
        // The inference thread will free input_tensors after processing

        log_debug(logger, "-> Sent input %d", i + 1);
    }

    return NULL;
}

// Thread function for receiving outputs
void *receive_output_thread(void *arg)
{
    Runtime *runtime = (Runtime *)arg;
    tensors_struct *output_tensors = NULL;
    int received_outputs = 0;
    int attempts = 0;
    // Maximum number of consecutive failed attempts to receive output tensors before giving up
    // This is useful in case the runtime is not able to provide output tensors for some reason
    // TODO: Adjust this as you see fit
    const int MAX_ATTEMPTS = 10;

    while (received_outputs < NUMBER_OF_INFERENCES)
    {
        // Receive the output tensors
        int code = runtime->receive_output(&output_tensors);
        if (code != 0)
        {
            log_warning(logger, "No more output tensors available. Attempt %d", attempts + 1);
            attempts++;
            if (attempts >= MAX_ATTEMPTS)
            {
                log_error(logger, "Exceeded maximum attempts to receive output tensors.");
                break;
            }
            sleep_ms(100); // Wait before retrying
            continue;
        }
        attempts = 0; // Reset attempts after a successful receive

        // if last iteration print out the output
        if (received_outputs == NUMBER_OF_INFERENCES - 1)
        {
            print_tensors_metadata(output_tensors);
        }

        // Free the output tensors
        deep_free_tensors_struct(output_tensors);
        output_tensors = NULL;
        received_outputs++;
        log_debug(logger, "<- Received output %d", received_outputs);
    }

    return NULL;
}

int main(int argc, char **argv)
{
    // Utils
    Timer timer;
    // Create a logger that prints and saves logs to a file
    // TODO: adjust the logger params: file name, file log level, and console log level
    // See OAAX/examples/tools/c-utilities/include/logger.h for more details
    logger = create_logger("C example", "main.log", LOG_DEBUG, LOG_DEBUG);
    if (logger == NULL)
    {
        printf("Failed to create logger.\n");
        return 1;
    }
    // Check command-line arguments
    if (argc != 4)
    {
        log_error(logger, "Usage: %s <library_path> <model_path> <image_path>", argv[0]);
        return 1;
    }

    char *library_path = argv[1];
    char *model_path = argv[2];
    char *image_path = argv[3];

    // Initialize the runtime environment
    Runtime *runtime = initialize_runtime(library_path);
    if (runtime == NULL)
    {
        log_error(logger, "Failed to initialize runtime.");
        return 1;
    }

    log_info(logger, "Runtime name: %s - Runtime version: %s",
             runtime->runtime_name(),
             runtime->runtime_version());

    // Initialize the runtime with arguments
    // These parameters are runtime-specific and may vary based on the runtime you are using
    // They are compatible for the CPU runtime and they server as
    // n_duplicates: Number of model duplicates that run asynchronously
    // n_threads_per_duplicate: Number of threads per model duplicate
    // TODO: Adjust these parameters as you see fit
    int n_duplicates = 1;
    int n_threads_per_duplicate = 1;
    int runtime_log_level = 0;
    int return_code = runtime->runtime_initialization_with_args(
        3,
        (const char *[]){"n_duplicates", "n_threads_per_duplicate", "runtime_log_level"},
        (const void *[]){&n_duplicates, &n_threads_per_duplicate, &runtime_log_level});

    if (return_code != 0)
    {
        log_error(logger, "Failed to initialize runtime environment.");
        destroy_runtime(runtime); // Clean up resources
        return 1;
    }

    // Load the model
    if (runtime->runtime_model_loading(model_path) != 0)
    {
        log_error(logger, "Failed to load model.");
        destroy_runtime(runtime); // Clean up resources
        return 1;
    }

    // Load the image
    // TODO: Depending on the model inputs, you may need to change the image size, mean, std and the tensors struct
    // Also, make sure to adapt the `resize_image` and `build_tensors_struct` function to your needs
    uint8_t *data = (uint8_t *)load_image(image_path, 320, 240, 127, 128, true);
    if (data == NULL)
    {
        log_error(logger, "Failed to load image.");
        destroy_runtime(runtime); // Clean up resources
        return 1;
    }
    // Create the input tensors struct from the image
    // TODO: Adjust the image size, mean, std and the tensors struct
    // Also, make sure to adapt the `resize_image` and `build_tensors_struct` function to your needs
    original_input_tensors = build_tensors_struct(data, 240, 320, 3);
    if (original_input_tensors == NULL)
    {
        log_error(logger, "Failed to build input tensors.");
        free(data);               // Free the image data
        destroy_runtime(runtime); // Clean up resources
        return 1;
    }
    // Start sending inputs and receiving outputs
    ThreadHandle send_input_thread_handle, receive_output_thread_handle;

    if (thread_create(&send_input_thread_handle, send_input_thread, runtime) != 0)
    {
        log_error(logger, "Failed to create send_input_thread.");
        deep_free_tensors_struct(original_input_tensors);
        destroy_runtime(runtime);
        return 1;
    }

    if (thread_create(&receive_output_thread_handle, receive_output_thread, runtime) != 0)
    {
        log_error(logger, "Failed to create receive_output_thread.");
        deep_free_tensors_struct(original_input_tensors);
        destroy_runtime(runtime);
        return 1;
    }

    // Wait for threads to complete
    // record current timestamp
    start_recording(&timer);
    thread_join(&send_input_thread_handle);
    thread_join(&receive_output_thread_handle);
    stop_recording(&timer);

    // Clean up
    deep_free_tensors_struct(original_input_tensors);
    original_input_tensors = NULL;

    destroy_runtime(runtime);

    // Optional: Print run stats
    print_memory_usage("CLOSE");
    print_human_readable_stats(&timer, NUMBER_OF_INFERENCES);

    return 0;
}
