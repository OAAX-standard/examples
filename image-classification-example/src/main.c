#include "runtime_utils.h"
#include "utils.h"

int main(int argc, char **argv) {
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
    printf("Building the input tensors\n");
    uint8_t *data = load_image(image_path, 224, 224, 127, 128, false);
    build_tensors_struct(data, 224, 224, 3, false, &input_tensors);

    // Perform inference
    runtime->runtime_inference_execution(&input_tensors, &output_tensors);

    // Extract the output tensors
    print_output_tensors(&output_tensors);

    // Free the input tensors
    free_tensors_struct(&input_tensors);

    // Request the runtime to cleanup the output tensors
    runtime->runtime_inference_cleanup();
    destroy_runtime(runtime);

    return 0;
}
