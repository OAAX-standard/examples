#include "runtime_utils.h"
#include "utils.h"

#include "memory.h"
#include "timer.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>


#define NUMBER_OF_INFERENCE 1

/* msleep(): Sleep for the requested number of milliseconds. */
int msleep(long msec)
{
    struct timespec ts;
    int res;

    if (msec < 0)
    {
        errno = EINVAL;
        return -1;
    }

    ts.tv_sec = msec / 1000;
    ts.tv_nsec = (msec % 1000) * 1000000;

    do {
        res = nanosleep(&ts, &ts);
    } while (res && errno == EINTR);

    return res;
}

// global vars
tensors_struct *original_input_tensors;

// Thread for sending inputs
void *send_input_thread(void *arg) {
    Runtime *runtime = (Runtime *) arg;
    int code = 0;
    for(int i = 0; i < NUMBER_OF_INFERENCE; i++) {
        // Deep copy the input tensors
        tensors_struct *input_tensors = deep_copy_tensors_struct(original_input_tensors);
        printf("input_tensors 1: %p\n", input_tensors);

        // Send the input tensors
        code = runtime->send_input(input_tensors);
        if (code != 0) {
            printf("!!!! Failed to send input tensors.\n");
            return NULL;
        }
    }

    return NULL;
}

// Thread for receiving outputs
void *receive_output_thread(void *arg) {
    Runtime *runtime = (Runtime *) arg;
    tensors_struct *output_tensors;

    for(int i = 0; i < NUMBER_OF_INFERENCE; i++) {
        // Receive the output tensors
        int code = runtime->receive_output(&output_tensors);
        if (code != 0) {
            printf("!!!! Failed to receive output tensors.\n");
            return NULL;
        }

        // Free the output tensors
        free_tensors_struct(output_tensors);
    }

    return NULL;
}

int main(int argc, char **argv) {
    // utils
    Timer timer;
    Memory *memory = create_memory_records(1000, 0.01);

    // args
    char *library_path;
    char *model_path;
    char *image_path;

    if (argc != 4) {
        printf("Usage: %s <library_path> <model_path> <image_path>\n", argv[0]);
        return 1;
    }

    library_path = argv[1];
    model_path = argv[2];
    image_path = argv[3];

    // Initialize the runtime environment
    Runtime *runtime = initialize_runtime(library_path);
    if(runtime == NULL)
        return 1;

    printf("Runtime name: %s - Runtime version: %s\n",
           runtime->runtime_name(),
           runtime->runtime_version());

    // Initialize the runtime
    int n_duplicates = 1;
    int n_threads_per_duplicate = 1;
    int return_code = runtime->runtime_initialization_with_args(2, (const char *[]){"n_duplicates", "n_threads_per_duplicate"},
                                                               (const void *[]){&n_duplicates, &n_threads_per_duplicate});
    if (return_code != 0) {
        printf("Failed to initialize runtime environment.\n");
        return 1;
    }

    // Load the model
    if (runtime->runtime_model_loading(model_path) != 0) {
        printf("Failed to load model.\n");
        return 1;
    }

    // Load the image
    // TODO: Depending on the model inputs, you may need to change the image size, mean, std and the tensors struct
    // Also, make sure to adapt the `resize_image` and `build_tensors_struct` function to your needs
    uint8_t *data = load_image(image_path, 320, 240, 127, 128, true);
    original_input_tensors = build_tensors_struct(data, 240, 320, 3);

    record_memory(memory);
    start_recording(&timer);
    
    // Start sending inputs and receiving outputs
    pthread_t send_input_thread_id, receive_output_thread_id;
    pthread_create(&send_input_thread_id, NULL, send_input_thread, runtime);
    pthread_create(&receive_output_thread_id, NULL, receive_output_thread, runtime);
    pthread_join(send_input_thread_id, NULL);
    pthread_join(receive_output_thread_id, NULL);

    // Record memory after the last iteration
    stop_recording(&timer);
    record_memory(memory);

    // Clean up
    free_tensors_struct(original_input_tensors);
    destroy_runtime(runtime);

    // Read statistics
    long first = get_first_record(memory);
    long last = get_last_record(memory);
    bool leaking = is_there_leak(memory);
    long elapsed_time_ms = get_elapsed_time_ms(&timer);

    printf("First memory record (B): %'ld, last memory record (B): %'ld, leaking: %d\n", first, last, leaking);
    printf("Memory difference: %'ld, difference per inference run: %'ld\n", last - first, (last - first) / NUMBER_OF_INFERENCE);
    printf("Elapsed time: %'li ms, Latency: %.2f, Throughput: %.2f\n", elapsed_time_ms, 
            (1.0f * elapsed_time_ms) / NUMBER_OF_INFERENCE, 
            (1000.0f * NUMBER_OF_INFERENCE) / elapsed_time_ms );

    // clean up
    save_to_file(memory);
    cleanup_memory_records(memory);
    return 0;
}
