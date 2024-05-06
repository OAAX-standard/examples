# C example

This example demonstrates the usage of the OAX runtime library to load an optimized ONNX model and run it on CPU.

## Overview

This folder contains the source code of a C program that will load the runtime library, load the model, and run it on a
sample image.
The program will print the output of the model to the console. However, you can confirm the output by comparing it to
the output of the Python example (which is printed and visualized).

The `artifacts/` directory contains the following files:

- `model.onnx`: the optimized ONNX model, which is a face locator model, ie. it will detect faces in an image.
- `libRuntimeLibrary.so`: the runtime library shared object file.
- `image.jpg`: an image that contains faces, which will be used as input to the model.

> The source code is available in the `src/` directory.

## Requirements

This example is expected to run on an Ubuntu 20.04 (or higher) machine with an X86_64 architecture (since the runtime
library provided is for X86_64).

Also, make sure that libjpeg is installed on your machine. You can install it using the following command:

```bash
sudo apt-get install libjpeg-dev
```

## Getting started

Assuming you have the required dependencies installed, you can build the example using the following commands:

```bash
cd c-example
mkdir build
cd build
cmake ..
make
```

After building the example, you can run it using the following command (from the `build/` directory):

```bash
./c_example ./artifacts/libRuntimeLibrary.so ./artifacts/model.onnx  ./artifacts/image.jpg
```

The program will print the output of the model to the console, along with some additional information.

## Indepth explanation

The `main.c` file contains the main function of the program. It will load the runtime library, load the model, and run
it on the input image. The script expects three arguments:

- The path to the runtime library shared object file.
- The path to the optimized model.
- The path to the input image.

For the sake of example, we've included a sample runtime library, runtime and image in the `artifacts/` directory.
You can replace it with your own files according to the
guidelines [below](#adapting-the-example-to-your-own-runtime---model---image-combination).

The `main.c` function uses two functions (available in the `utils.c` file) to preprocess the input image and build the
input tensors as specified by the
OAX standard.

- The `preprocess_image` function reads the input image, resizes it to the model's input size, and normalizes it.

```c
/**
 * @brief Load the image from the file path, resize it, normalize it and convert it to float.
 * @param [in] image_path Path to the image file
 * @param [in] new_width Desired width of the image
 * @param [in] new_height Desired height of the image
 * @param [in] mean Float value to subtract from the image pixel values
 * @param [in] std Float value to divide the image pixel values (after mean subtraction)
 * @param [in] nchw Boolean flag to indicate if the image should be in NCHW or NHWC format
 * @return Pointer to the resized image in float format
 */
void *load_image(const char *image_path, int new_width, int new_height, float mean, float std, bool nchw);
```

- The `build_input_tensors` function populates the input tensors with the preprocessed image data along with the NMS
  threshold.

```c
/**
 * @brief Build the tensors struct from the input data
 * @param [in] data Pointer to the preprocessed image data
 * @param [in] height Preprocessed image height
 * @param [in] width Preprocessed image width
 * @param [in] channels Number of channels in the image
 * @param [out] tensors Pointer to the tensors struct
 */
void build_tensors_struct(uint8_t *data, size_t height, size_t width, size_t channels, tensors_struct *tensors);
```

These two functions are called in the `main.c` file to preprocess the input image and build the input tensors, as shown
below:

```c
// Load the image
// TODO: Depending on the model inputs, you may need to change the image size, mean, std and the tensors struct
// Also, make sure to adapt the `resize_image` and `build_tensors_struct` function to your needs
uint8_t *data = load_image(image_path, 320, 240, 127, 128, true);
build_tensors_struct(data, 240, 320, 3, &input_tensors);
```

### Adapting the example to your own runtime - model - image combination

When using your own runtime library, optimized model, and/or input image, make sure that:

- The runtime library shared object file is compiled for the same architecture as your machine.
- The optimized model file is compatible with the runtime library.
- The `main.c` file is updated to build the input tensor according to the model's input tensors format. You can find a
  TODO
  comment in the `main.c` file, which indicates where you should update the code.
- You may need to remove/update the last line in the `CMakeLists.txt` file, since it copies the `artifacts/` directory
  to the build directory.