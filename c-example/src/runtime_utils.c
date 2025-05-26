#include "runtime_utils.h"

#include "lib_loader.h"
#include "logger.h"

#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef _WIN32
#include <windows.h>
#include <stdio.h>
static char win32_dl_error_msg[512];
static const char *get_dl_error()
{
    DWORD err = GetLastError();
    if (err == 0)
        return NULL;
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                   win32_dl_error_msg, sizeof(win32_dl_error_msg), NULL);
    return win32_dl_error_msg;
}
#define DL_ERROR get_dl_error()
#else
#include <dlfcn.h>
#define DL_ERROR dlerror()
#endif

extern Logger *logger;

static char *copy_string(const char *str)
{
    if (str == NULL)
        return NULL;

    size_t len = strlen(str);
    char *copy = (char *)malloc(len + 1);
    if (copy == NULL)
        return NULL;

    strcpy(copy, str);
    return copy;
}

void destroy_runtime(Runtime *runtime)
{
    if (runtime == NULL)
        return;

    if (runtime->runtime_destruction != NULL)
    {
        runtime->runtime_destruction();
    }

    if (runtime->_library_path != NULL)
    {
        free(runtime->_library_path);
        runtime->_library_path = NULL;
    }

    if (runtime->_handle != NULL)
    {
        close_dynamic_library(runtime->_handle);
        runtime->_handle = NULL;
    }

    free(runtime);
    runtime = NULL;
}

Runtime *initialize_runtime(const char *library_path)
{
    Runtime *runtime = (Runtime *)calloc(1, sizeof(Runtime));
    if (runtime == NULL)
    {
        log_error(logger, "Failed to allocate memory for Runtime.");
        return NULL;
    }

    log_debug(logger, "Initializing runtime with library: %s", library_path);

    runtime->_library_path = NULL;
    runtime->_handle = NULL;

    // Copy the library path
    runtime->_library_path = malloc(strlen(library_path) + 1);
    if (runtime->_library_path == NULL)
    {
        destroy_runtime(runtime);
        log_error(logger, "Failed to load library: %s", DL_ERROR ? DL_ERROR : "Unknown error");
        return NULL;
    }
    strcpy(runtime->_library_path, library_path);

    // Load the shared library
    runtime->_handle = load_dynamic_library(library_path);
    if (runtime->_handle == NULL)
    {
        destroy_runtime(runtime);
        log_error(logger, "Failed to load library: %s", DL_ERROR);
        return NULL;
    }
    log_debug(logger, "Loaded library handle: %p", runtime->_handle);

    runtime->runtime_initialization = get_symbol_address(runtime->_handle, "runtime_initialization");
    if (runtime->runtime_initialization == NULL)
    {
        log_error(logger, "`runtime_initialization` not implemented: %s.", DL_ERROR);
    }
    log_debug(logger, "Loaded `runtime_initialization` function: %p", runtime->runtime_initialization);
    runtime->runtime_initialization_with_args = get_symbol_address(runtime->_handle, "runtime_initialization_with_args");
    if (runtime->runtime_initialization_with_args == NULL)
    {
        log_error(logger, "`runtime_initialization_with_args` not implemented: %s.", DL_ERROR);
    }
    log_debug(logger, "Loaded `runtime_initialization_with_args` function: %p", runtime->runtime_initialization_with_args);
    runtime->runtime_model_loading = get_symbol_address(runtime->_handle, "runtime_model_loading");
    if (runtime->runtime_model_loading == NULL)
    {
        destroy_runtime(runtime);
        log_error(logger, "Failed to load `runtime_model_loading` function: %s.", DL_ERROR);
        return NULL;
    }
    log_debug(logger, "Loaded `runtime_model_loading` function: %p", runtime->runtime_model_loading);
    runtime->send_input = get_symbol_address(runtime->_handle, "send_input");
    if (runtime->send_input == NULL)
    {
        destroy_runtime(runtime);
        log_error(logger, "Failed to load `send_input` function: %s.", DL_ERROR);
        return NULL;
    }
    runtime->receive_output = get_symbol_address(runtime->_handle, "receive_output");
    if (runtime->receive_output == NULL)
    {
        destroy_runtime(runtime);
        log_error(logger, "Failed to load `receive_output` function: %s.", DL_ERROR);
        return NULL;
    }
    runtime->runtime_destruction = get_symbol_address(runtime->_handle, "runtime_destruction");
    if (runtime->runtime_destruction == NULL)
    {
        destroy_runtime(runtime);
        log_error(logger, "Failed to load `runtime_destruction` function: %s.", DL_ERROR);
        return NULL;
    }
    log_debug(logger, "Loaded `runtime_destruction` function: %p", runtime->runtime_destruction);
    runtime->runtime_error_message = get_symbol_address(runtime->_handle, "runtime_error_message");
    if (runtime->runtime_error_message == NULL)
    {
        destroy_runtime(runtime);
        log_error(logger, "Failed to load `runtime_error_message` function: %s.", DL_ERROR);
        return NULL;
    }
    log_debug(logger, "Loaded `runtime_error_message` function: %p", runtime->runtime_error_message);
    runtime->runtime_version = get_symbol_address(runtime->_handle, "runtime_version");
    if (runtime->runtime_version == NULL)
    {
        destroy_runtime(runtime);
        log_error(logger, "Failed to load `runtime_version` function: %s.", DL_ERROR);
        return NULL;
    }
    log_debug(logger, "Loaded `runtime_version` function: %p", runtime->runtime_version);
    runtime->runtime_name = get_symbol_address(runtime->_handle, "runtime_name");
    if (runtime->runtime_name == NULL)
    {
        destroy_runtime(runtime);
        log_error(logger, "Failed to load `runtime_name` function: %s.", DL_ERROR);
        return NULL;
    }
    log_debug(logger, "Loaded `runtime_name` function: %p", runtime->runtime_name);

    return runtime;
}

void resize_image(const unsigned char *image,
                  int width,
                  int height,
                  int new_width,
                  int new_height,
                  float *resized_image)
{
    if (image == NULL)
        return;
    if (resized_image == NULL)
        return;

    // Simple resizing algorithm: nearest neighbor interpolation
    double x_ratio = (double)width / new_width;
    double y_ratio = (double)height / new_height;

    for (int y = 0; y < new_height; y++)
    {
        for (int x = 0; x < new_width; x++)
        {
            int px = (int)(x * x_ratio);
            int py = (int)(y * y_ratio);
            resized_image[(y * new_width + x) * 3] = image[(py * width + px) * 3];
            resized_image[(y * new_width + x) * 3 + 1] = image[(py * width + px) * 3 + 1];
            resized_image[(y * new_width + x) * 3 + 2] = image[(py * width + px) * 3 + 2];
        }
    }
}

void *load_image(const char *image_path, int new_width, int new_height, float mean, float std, bool nchw)
{
    FILE *input_file = fopen(image_path, "rb");
    if (!input_file)
    {
        log_error(logger, "Error: Couldn't open the image file.");
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

    unsigned char *image = (unsigned char *)malloc(width * height * num_channels * sizeof(unsigned char));
    float *resized_image = (float *)malloc(new_height * new_width * num_channels * sizeof(float));

    while (cinfo.output_scanline < cinfo.output_height)
    {
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
    if (nchw)
    {
        float *transposed_image = (float *)malloc(new_height * new_width * num_channels * sizeof(float));
        for (int c = 0; c < num_channels; c++)
        {
            for (int y = 0; y < new_height; y++)
            {
                for (int x = 0; x < new_width; x++)
                {
                    transposed_image[c * new_height * new_width + y * new_width + x] =
                        resized_image[y * new_width * num_channels + x * num_channels + c];
                }
            }
        }
        free(resized_image);
        resized_image = transposed_image;
    }

    free(image);
    return (void *)resized_image;
}

tensors_struct *build_tensors_struct(uint8_t *data, size_t height, size_t width, size_t channels)
{
    tensors_struct *input_tensors = (tensors_struct *)malloc(sizeof(tensors_struct));

    if (input_tensors == NULL)
    {
        log_error(logger, "Failed to allocate memory for input tensors.");
        return NULL;
    }
    input_tensors->num_tensors = 2;
    input_tensors->names = (char **)malloc(input_tensors->num_tensors * sizeof(char *));
    input_tensors->data_types = (tensor_data_type *)malloc(input_tensors->num_tensors * sizeof(tensor_data_type));
    input_tensors->ranks = (size_t *)malloc(input_tensors->num_tensors * sizeof(size_t));
    input_tensors->shapes = (size_t **)malloc(input_tensors->num_tensors * sizeof(size_t *));
    input_tensors->data = (void **)malloc(input_tensors->num_tensors * sizeof(void *));

    // First tensor: input image
    input_tensors->names[0] = copy_string("image-");
    input_tensors->data_types[0] = DATA_TYPE_FLOAT;
    input_tensors->ranks[0] = 4;
    input_tensors->shapes[0] = (size_t *)malloc(input_tensors->ranks[0] * sizeof(size_t));
    input_tensors->shapes[0][0] = 1;
    input_tensors->shapes[0][1] = channels;
    input_tensors->shapes[0][2] = height;
    input_tensors->shapes[0][3] = width;
    input_tensors->data[0] = (void *)data;

    // Second tensor: NMS threshold
    input_tensors->names[1] = copy_string("nms_sensitivity-");
    input_tensors->data_types[1] = DATA_TYPE_FLOAT;
    input_tensors->ranks[1] = 1;
    input_tensors->shapes[1] = (size_t *)malloc(input_tensors->ranks[1] * sizeof(size_t));
    input_tensors->shapes[1][0] = 1;
    input_tensors->data[1] = (void *)malloc(input_tensors->shapes[1][0] * sizeof(float));
    ((float *)input_tensors->data[1])[0] = 0.5f;

    return input_tensors;
}
