#include "runtime_utils.h"
#include "utils.h"

#include "memory.h"
#include "timer.h"

#define NUMBER_OR_INFERENCE 1000

int main(int argc, char **argv) {
    // utils
    Timer timer;
    Memory *memory = create_memory_records(10000, 0.1);

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

    // Declare the input and output tensors
    tensors_struct input_tensors, output_tensors;

    // Initialize the runtime environment
    Runtime *runtime = initialize_runtime(library_path);
    if(runtime == NULL)
        return 1;

    printf("Runtime name: %s - Runtime version: %s\n",
           runtime->runtime_name(),
           runtime->runtime_version());

    // Initialize the runtime
    if (runtime->runtime_initialization() != 0) {
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
    build_tensors_struct(data, 240, 320, 3, &input_tensors);

    start_recording(&timer);
    for(int i=0; i<NUMBER_OR_INFERENCE; i++){
        // Record memory used by this program
        if(i > 1)
            record_memory(memory);

        // Perform inference
        runtime->runtime_inference_execution(&input_tensors, &output_tensors);

        // Extract the output tensors
        // print_output_tensors(&output_tensors);

        // Request the runtime to cleanup the output tensors
        runtime->runtime_inference_cleanup();
    }
    stop_recording(&timer);

    // Free the input tensors
    free_tensors_struct(&input_tensors);

    // Read statistics
    long first = get_first_record(memory);
    long last = get_last_record(memory);
    bool leaking = is_there_leak(memory);
    long elapsed_time_ms = get_elapsed_time_ms(&timer);

    printf("First memory record (B): %ld, last memory record (B): %ld, leaking: %d\n", first, last, leaking);
    printf("Memory difference: %ld, difference per inference run: %ld\n", last - first, (last - first) / NUMBER_OR_INFERENCE);
    printf("Elapsed time: %li ms, Latency: %.2f, Throughput: %.2f\n", elapsed_time_ms, 
            (1.0f * elapsed_time_ms) / NUMBER_OR_INFERENCE, 
            (1000.0f * NUMBER_OR_INFERENCE) / elapsed_time_ms );

    // clean up
    save_to_file(memory);
    cleanup_memory_records(memory);
    destroy_runtime(runtime);
    return 0;
}
