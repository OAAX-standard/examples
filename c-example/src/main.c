#include "runtime_utils.h"
#include "utils.h"
#include "timer.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

// Number of inferences to perform
#define NUMBER_OF_INFERENCE 1000

// Global variable to hold the original input tensors
tensors_struct *original_input_tensors = NULL;

// Function to sleep for the requested number of milliseconds
int msleep(long msec) {
    struct timespec ts;
    int res;

    if (msec < 0) {
        errno = EINVAL;
        return -1;
    }

    ts.tv_sec  = msec / 1000;
    ts.tv_nsec = (msec % 1000) * 1000000;

    do {
        res = nanosleep(&ts, &ts);
    } while (res && errno == EINTR);

    return res;
}

// Thread function for sending inputs
void *send_input_thread(void *arg) {
    Runtime *runtime = (Runtime *)arg;
    int code         = 0;

    for (int i = 0; i < NUMBER_OF_INFERENCE; i++) {
        // Deep copy the input tensors
        tensors_struct *input_tensors = deep_copy_tensors_struct(original_input_tensors);
        if (input_tensors == NULL) {
            printf("Error: Failed to deep copy input tensors.\n");
            continue;
        }

        // Send the input tensors
        code = runtime->send_input(input_tensors);
        if (code != 0) {
            printf("Error: Failed to send input tensors.\n");
            free_tensors_struct(input_tensors); // Free before returning
            return NULL;
        }

        // Ownership of input_tensors is transferred to the runtime
        // The inference thread will free input_tensors after processing
    }

    return NULL;
}

// Thread function for receiving outputs
void *receive_output_thread(void *arg) {
    Runtime *runtime = (Runtime *)arg;
    tensors_struct *output_tensors = NULL;
    int received_outputs           = 0;
    int attempts                   = 0;
    const int MAX_ATTEMPTS         = 10;

    while (received_outputs < NUMBER_OF_INFERENCE) {
        // Receive the output tensors
        int code = runtime->receive_output(&output_tensors);
        if (code != 0) {
            printf("Warning: No more output tensors available. Attempt %d\n", attempts + 1);
            attempts++;
            if (attempts >= MAX_ATTEMPTS) {
                printf("Error: Exceeded maximum attempts to receive output tensors.\n");
                break;
            }
            msleep(100); // Wait before retrying
            continue;
        }
        attempts = 0; // Reset attempts after a successful receive

        // Process the output tensors if needed
        // ...

        // Free the output tensors
        free_tensors_struct(output_tensors);
        output_tensors = NULL;
        received_outputs++;
    }

    return NULL;
}

int main(int argc, char **argv) {
    // Utils
    Timer timer;

    // Check command-line arguments
    if (argc != 4) {
        printf("Usage: %s <library_path> <model_path> <image_path>\n", argv[0]);
        return 1;
    }

    char *library_path = argv[1];
    char *model_path   = argv[2];
    char *image_path   = argv[3];

    // Initialize the runtime environment
    Runtime *runtime = initialize_runtime(library_path);
    if (runtime == NULL) {
        printf("Error: Failed to initialize runtime.\n");
        return 1;
    }

    printf("Runtime name: %s - Runtime version: %s\n",
           runtime->runtime_name(),
           runtime->runtime_version());

    // Initialize the runtime with arguments
    int n_duplicates           = 1;
    int n_threads_per_duplicate = 1;
    int return_code = runtime->runtime_initialization_with_args(
            2,
            (const char *[]){"n_duplicates", "n_threads_per_duplicate"},
            (const void *[]){&n_duplicates, &n_threads_per_duplicate});

    if (return_code != 0) {
        printf("Error: Failed to initialize runtime environment.\n");
        destroy_runtime(runtime); // Clean up resources
        return 1;
    }

    // Load the model
    if (runtime->runtime_model_loading(model_path) != 0) {
        printf("Error: Failed to load model.\n");
        destroy_runtime(runtime); // Clean up resources
        return 1;
    }

    // Load the image
    uint8_t *data = (uint8_t *)load_image(image_path, 320, 240, 127, 128, true);
    if (data == NULL) {
        printf("Error: Failed to load image.\n");
        destroy_runtime(runtime); // Clean up resources
        return 1;
    }

    original_input_tensors = build_tensors_struct(data, 240, 320, 3);
    if (original_input_tensors == NULL) {
        printf("Error: Failed to build input tensors.\n");
        free(data);              // Free the image data
        destroy_runtime(runtime); // Clean up resources
        return 1;
    }

    // Start sending inputs and receiving outputs
    pthread_t send_input_thread_id, receive_output_thread_id;
    if (pthread_create(&send_input_thread_id, NULL, send_input_thread, runtime) != 0) {
        printf("Error: Failed to create send_input_thread.\n");
        free_tensors_struct(original_input_tensors);
        destroy_runtime(runtime);
        return 1;
    }

    if (pthread_create(&receive_output_thread_id, NULL, receive_output_thread, runtime) != 0) {
        printf("Error: Failed to create receive_output_thread.\n");
        pthread_cancel(send_input_thread_id); // Cancel the send thread
        free_tensors_struct(original_input_tensors);
        destroy_runtime(runtime);
        return 1;
    }

    // Wait for threads to complete
    pthread_join(send_input_thread_id, NULL);
    pthread_join(receive_output_thread_id, NULL);

    // Clean up
    free_tensors_struct(original_input_tensors);
    original_input_tensors = NULL;

    destroy_runtime(runtime);

    // Optional: Print memory usage
    print_memory_usage("CLOSE");

    // Optional: Sleep to monitor memory usage
    // sleep(1000); // Uncomment if needed

    return 0;
}
I