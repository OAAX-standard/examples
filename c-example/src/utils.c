#include "utils.h"
#include "string.h"
#include <malloc.h>
#include <math.h>

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

tensors_struct *build_tensors_struct(uint8_t *data, size_t height, size_t width, size_t channels) {
    tensors_struct *input_tensors = (tensors_struct *) malloc(sizeof(tensors_struct));

    if (input_tensors == NULL) {
        printf("Failed to allocate memory for input tensors.\n");
        return NULL;
    }
    input_tensors->num_tensors = 2;
    input_tensors->names = NULL;
    input_tensors->data_types = (tensor_data_type *) malloc(input_tensors->num_tensors * sizeof(tensor_data_type));
    input_tensors->ranks = (size_t *) malloc(input_tensors->num_tensors * sizeof(size_t));
    input_tensors->shapes = (size_t **) malloc(input_tensors->num_tensors * sizeof(size_t *));
    input_tensors->data = (void **) malloc(input_tensors->num_tensors * sizeof(void *));

    // First tensor: input image
    input_tensors->data_types[0] = DATA_TYPE_FLOAT;
    input_tensors->ranks[0] = 4;
    input_tensors->shapes[0] = (size_t *) malloc(input_tensors->ranks[0] * sizeof(size_t));
    input_tensors->shapes[0][0] = 1;
    input_tensors->shapes[0][1] = channels;
    input_tensors->shapes[0][2] = height;
    input_tensors->shapes[0][3] = width;
    input_tensors->data[0] = (void *) data;

    // Second tensor: NMS threshold
    input_tensors->data_types[1] = DATA_TYPE_FLOAT;
    input_tensors->ranks[1] = 1;
    input_tensors->shapes[1] = (size_t *) malloc(input_tensors->ranks[1] * sizeof(size_t));
    input_tensors->shapes[1][0] = 1;
    input_tensors->data[1] = (void *) malloc(input_tensors->shapes[1][0] * sizeof(float));
    ((float *) input_tensors->data[1])[0] = 0.5f;

    return input_tensors;
}

void print_tensors(tensors_struct *tensors) {
    // Extract the output tensors
    printf("Number of output tensors: %zu\n", tensors->num_tensors);

    for (size_t i = 0; i < tensors->num_tensors; i++) {
        if(tensors->names != NULL)
            printf("Tensor name: %s\n", tensors->names[i]);
        printf("Tensor data type: %d\n", tensors->data_types[i]);
        printf("Tensor rank: %zu\n", tensors->ranks[i]);
        printf("Tensor shape: [");
        for (size_t j = 0; j < tensors->ranks[i]; j++) {
            printf("%zu, ", tensors->shapes[i][j]);
        }
        printf("]\n");
        printf("Tensor data: \n");
        float *data = (float *) tensors->data[i];
        size_t size = 1;
        for (size_t j = 0; j < tensors->ranks[i]; j++) {
            size *= tensors->shapes[i][j];
        }
        int size_to_print = size < 10 ? size : 10;
        for (size_t j = 0; j < size_to_print; j++) {
            printf("%f, ", data[j]);
            if(j % tensors->shapes[0][1] == 5)
                printf("\n");
        }
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
            if (tensors->data[i] != NULL) {
                free(tensors->data[i]);
                tensors->data[i] = NULL;
            }
        }
        free(tensors->data);
        tensors->data = NULL;
    }

    if (tensors->shapes != NULL) {
        for (size_t i = 0; i < tensors->num_tensors; i++) {
            if (tensors->shapes[i] != NULL) {
                free(tensors->shapes[i]);
                tensors->shapes[i] = NULL;
            }
        }
        free(tensors->shapes);
        tensors->shapes = NULL;
    }

    if (tensors->names != NULL) {
        for (size_t i = 0; i < tensors->num_tensors; i++) {
            if (tensors->names[i] != NULL) {
                free(tensors->names[i]);
                tensors->names[i] = NULL;
            }
        }
        free(tensors->names);
        tensors->names = NULL;
    }
    free(tensors);
    tensors = NULL;
}

tensors_struct* deep_copy_tensors_struct(tensors_struct* tensors) {
    tensors_struct* new_tensors = (tensors_struct*)malloc(sizeof(tensors_struct));
    new_tensors->num_tensors = tensors->num_tensors;
    if (tensors->names == NULL) {
        new_tensors->names = NULL;
    } else {
        new_tensors->names = (char**)malloc(new_tensors->num_tensors * sizeof(char*));
    }
    new_tensors->data_types = (tensor_data_type*)malloc(new_tensors->num_tensors * sizeof(tensor_data_type));
    new_tensors->ranks = (size_t*)malloc(new_tensors->num_tensors * sizeof(size_t));
    new_tensors->shapes = (size_t**)malloc(new_tensors->num_tensors * sizeof(size_t*));
    new_tensors->data = (void**)malloc(new_tensors->num_tensors * sizeof(void*));

    for (size_t i = 0; i < new_tensors->num_tensors; i++) {
        // Copy the names
        if (new_tensors->names != NULL) {
            new_tensors->names[i] = (char*)malloc(strlen(tensors->names[i]) + 1);
            strcpy(new_tensors->names[i], tensors->names[i]);
        }
        // Copy the data types
        new_tensors->data_types[i] = tensors->data_types[i];
        // Copy the ranks
        new_tensors->ranks[i] = tensors->ranks[i];
        // Copy the shapes
        new_tensors->shapes[i] = (size_t*)malloc(new_tensors->ranks[i] * sizeof(size_t));
        long size = 1;
        for (size_t j = 0; j < new_tensors->ranks[i]; j++) {
            new_tensors->shapes[i][j] = tensors->shapes[i][j];
            size *= new_tensors->shapes[i][j];
        }
        // Copy the data
        long bytes = size * get_sizeof_onnx_type(new_tensors->data_types[i]);
        new_tensors->data[i] = (void*) malloc(bytes);
        memcpy(new_tensors->data[i], tensors->data[i], bytes);
    }

    return new_tensors;
}

int64_t get_sizeof_onnx_type(int32_t datatype) {
    if (datatype == DATA_TYPE_INT8)
        return sizeof(int8_t);
    if (datatype == DATA_TYPE_UINT8)
        return sizeof(uint8_t);
    if (datatype == DATA_TYPE_BOOL)
        return sizeof(bool);
    if (datatype == DATA_TYPE_INT16)
        return sizeof(int16_t);
    if (datatype == DATA_TYPE_INT16)
        return sizeof(int16_t);
    if (datatype == DATA_TYPE_UINT16)
        return sizeof(uint16_t);
    if (datatype == DATA_TYPE_INT32)
        return sizeof(int32_t);
    if (datatype == DATA_TYPE_INT64)
        return sizeof(int64_t);
    if (datatype == DATA_TYPE_FLOAT)
        return sizeof(float);
    if (datatype == DATA_TYPE_DOUBLE)
        return sizeof(double);
    return 0;
}