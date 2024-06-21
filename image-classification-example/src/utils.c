#include "utils.h"
#include <string.h>

void resize_image(const unsigned char *image,
                  int width,
                  int height,
                  int new_width,
                  int new_height,
                  float *resized_image) {
    if (image == NULL) return;
    if (resized_image == NULL) return;

    // Simple resizing algorithm: nearest neighbor interpolation
    double x_ratio = (double) width / new_width;
    double y_ratio = (double) height / new_height;

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            int px = (int) (x * x_ratio);
            int py = (int) (y * y_ratio);
            resized_image[(y * new_width + x) * 3] = image[(py * width + px) * 3];
            resized_image[(y * new_width + x) * 3 + 1] = image[(py * width + px) * 3 + 1];
            resized_image[(y * new_width + x) * 3 + 2] = image[(py * width + px) * 3 + 2];
        }
    }
}

void *load_image(const char *image_path, int new_width, int new_height, float mean, float std, bool nchw) {
    FILE *input_file = fopen(image_path, "rb");
    if (!input_file) {
        printf("Error: Couldn't open the image file.\n");
        exit(1);
        return NULL;
    }

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, input_file);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    int width = cinfo.output_width;
    int height = cinfo.output_height;
    int num_channels = cinfo.output_components;

    unsigned char *image = (unsigned char *) malloc(width * height * num_channels * sizeof(unsigned char));
    float *resized_image = (float *) malloc(new_height * new_width * num_channels * sizeof(float));

    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char *row = &image[cinfo.output_scanline * cinfo.output_width * cinfo.output_components];
        jpeg_read_scanlines(&cinfo, &row, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(input_file);

    // resize and convert the image to float
    resize_image(image, width, height, new_width, new_height, resized_image);

    // Normalize the image
    for (int i = 0; i < new_height * new_width * num_channels; i++)
        resized_image[i] = (resized_image[i] - mean) / std;

    // Convert to NCHW format
    if (nchw) {
        float *transposed_image = (float *) malloc(new_height * new_width * num_channels * sizeof(float));
        for (int c = 0; c < num_channels; c++) {
            for (int y = 0; y < new_height; y++) {
                for (int x = 0; x < new_width; x++) {
                    transposed_image[c * new_height * new_width + y * new_width + x] =
                            resized_image[y * new_width * num_channels + x * num_channels + c];
                }
            }
        }
        free(resized_image);
        resized_image = transposed_image;
    }

    free(image);
    return (void *) resized_image;
}

void build_tensors_struct(uint8_t *data, size_t height, size_t width, size_t channels, bool nchw, tensors_struct *input_tensors) {
    input_tensors->num_tensors = 1;
    input_tensors->names = (char **) malloc(input_tensors->num_tensors * sizeof(char *));
    input_tensors->data_types = (tensor_data_type *) malloc(input_tensors->num_tensors * sizeof(tensor_data_type));
    input_tensors->ranks = (size_t *) malloc(input_tensors->num_tensors * sizeof(size_t));
    input_tensors->shapes = (size_t **) malloc(input_tensors->num_tensors * sizeof(size_t *));
    input_tensors->data = (void **) malloc(input_tensors->num_tensors * sizeof(void *));

    // First tensor: input image
    input_tensors->data_types[0] = DATA_TYPE_FLOAT;
    input_tensors->ranks[0] = 4;
    input_tensors->shapes[0] = (size_t *) malloc(input_tensors->ranks[0] * sizeof(size_t));
    if (nchw){
        input_tensors->shapes[0][0] = 1;
        input_tensors->shapes[0][1] = channels;
        input_tensors->shapes[0][2] = height;
        input_tensors->shapes[0][3] = width;
    } else {
        input_tensors->shapes[0][0] = 1;
        input_tensors->shapes[0][1] = height;
        input_tensors->shapes[0][2] = width;
        input_tensors->shapes[0][3] = channels;
    }
    input_tensors->data[0] = (void *) data;
    input_tensors->names[0] = (char *) malloc(100 * sizeof(char));
    strcpy(input_tensors->names[0], "image-");
}

void print_output_tensors(tensors_struct *output_tensors) {
    // Extract the output tensors
    printf("Number of output tensors: %zu\n", output_tensors->num_tensors);

    for (size_t i = 0; i < output_tensors->num_tensors; i++) {
        printf("Tensor name: %s\n", output_tensors->names[i]);
        printf("Tensor data type: %d\n", output_tensors->data_types[i]);
        printf("Tensor rank: %zu\n", output_tensors->ranks[i]);
        printf("Tensor shape: [");
        for (size_t j = 0; j < output_tensors->ranks[i]; j++) {
            printf("%zu, ", output_tensors->shapes[i][j]);
        }
        printf("]\n");
        printf("Tensor data: \n");
        float *data = (float *) output_tensors->data[i];
        size_t size = 1;
        for (size_t j = 0; j < output_tensors->ranks[i]; j++) {
            size *= output_tensors->shapes[i][j];
        }
        for (size_t j = 0; j < size; j++) {
            printf("%f, ", data[j]);
            if(j % output_tensors->shapes[0][1] == 5)
                printf("\n");
        }
        printf("\n");
    }

}


void free_tensors_struct(tensors_struct *tensors) {
    if (tensors->data_types != NULL) {
        free(tensors->data_types);
        tensors->data_types = NULL;
    }

    if (tensors->ranks != NULL) {
        free(tensors->ranks);
        tensors->ranks = NULL;
    }

    if (tensors->data != NULL) {
        for (size_t i = 0; i < tensors->num_tensors; i++) {
            if (tensors->data[i] != NULL)
                free(tensors->data[i]);
        }
        free(tensors->data);
        tensors->data = NULL;
    }

    if (tensors->shapes != NULL) {
        for (size_t i = 0; i < tensors->num_tensors; i++) {
            if (tensors->shapes[i] != NULL)
                free(tensors->shapes[i]);
        }
        free(tensors->shapes);
        tensors->shapes = NULL;
    }

    if (tensors->names != NULL) {
        for (size_t i = 0; i < tensors->num_tensors; i++) {
            if (tensors->names[i] != NULL)
                free(tensors->names[i]);
        }
        free(tensors->names);
        tensors->names = NULL;
    }
}