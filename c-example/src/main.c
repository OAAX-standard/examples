// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

// Example OAAX 2.0 host application: loads a runtime library and an optimized
// face detection model, sends a few inference requests with the same image,
// matches every output to its request and prints the detections.
//
// Usage: ./c_example <runtime library> <model path> <image path>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "image.h"    // NOLINT[build/include]
#include "runtime.h"  // NOLINT[build/include]
#include "tensors.h"  // NOLINT[build/include]

// Number of inference requests to send
// NOTE: Adjust this as you see fit
#define NUM_REQUESTS 10

// Max time to wait for each output (the runtime may be busy with a backlog)
#define RETRIEVE_TIMEOUT_MS 5000

// Model-specific values (the face locator in artifacts/model.onnx)
// NOTE: Adjust these to the input tensors of your own model
#define MODEL_ID 0  // Index of the model in the array given to load_models
#define IMAGE_WIDTH 320
#define IMAGE_HEIGHT 240
#define IMAGE_MEAN 127.0f
#define IMAGE_STD 128.0f
#define NMS_SENSITIVITY 0.5f

static double now_ms(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// Builds the input tensors of request `id`: the image (NCHW float) and the NMS
// sensitivity. The tensor names are those declared in the model's ONNX graph.
// The image is copied: once enqueued, the runtime owns and frees the tensors.
static Tensors *build_input_tensors(int id, const float *image) {
  const int image_shape[] = {1, 3, IMAGE_HEIGHT, IMAGE_WIDTH};
  const size_t image_size =
      3 * IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(float);  // In bytes
  const int nms_shape[] = {1};
  const float nms_sensitivity = NMS_SENSITIVITY;

  Tensors *input = alloc_tensors(id, 2);
  if (input == NULL) return NULL;
  if (fill_tensor(&input->tensors[0], "image-", DATA_TYPE_FLOAT, 4, image_shape,
                  image, image_size) != 0 ||
      fill_tensor(&input->tensors[1], "nms_sensitivity-", DATA_TYPE_FLOAT, 1,
                  nms_shape, &nms_sensitivity, sizeof(float)) != 0) {
    free_tensors(input);
    return NULL;
  }
  return input;
}

// Prints the metadata of an output tensor, and its detections if it holds
// `bboxes-format:xyxysc` rows (x1, y1, x2, y2, score, class), where the
// coordinates are relative to the image size (0 to 1).
static void print_output(const Tensors *output) {
  for (int i = 0; i < output->num_tensors; i++) {
    const TensorDescriptor *tensor = &output->tensors[i];
    printf("Output tensor '%s': type %d, shape [", tensor->name,
           (int)tensor->data_type);
    for (int d = 0; d < tensor->rank; d++) {
      printf(d ? ", %d" : "%d", tensor->shape[d]);
    }
    printf("]\n");

    if (tensor->data_type != DATA_TYPE_FLOAT || tensor->rank != 2 ||
        tensor->shape[1] != 6 ||
        tensor->data_size < (size_t)tensor->shape[0] * 6 * sizeof(float)) {
      continue;
    }
    const float *rows = (const float *)tensor->data;
    for (int r = 0; r < tensor->shape[0]; r++) {
      const float *box = rows + r * 6;
      printf("  face %.3f at (%.3f, %.3f) - (%.3f, %.3f)\n", box[4], box[0],
             box[1], box[2], box[3]);
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <runtime library> <model path> <image path>\n",
            argv[0]);
    return 1;
  }
  const char *library_path = argv[1];
  const char *model_path = argv[2];
  const char *image_path = argv[3];

  Runtime runtime;
  if (load_runtime_library(library_path, &runtime) != 0) return 1;
  printf("Runtime: %s %s\n", runtime.get_name(), runtime.get_version());

  int exit_code = 1;
  int received[NUM_REQUESTS] = {0};
  RuntimeStatus status;
  float *image = NULL;

  // 1. Initialize the runtime. Keys and values are always strings; the keys a
  // runtime understands are vendor-specific and unknown keys are ignored.
  // NOTE: Adjust these as you see fit (see your runtime's documentation)
  const char *init_keys[] = {"log_level", "log_file"};
  const char *init_values[] = {"3", "runtime.log"};
  Config init_config = {2, init_keys, init_values};
  if ((status = runtime.init(init_config)) != RUNTIME_STATUS_SUCCESS) {
    report_error(&runtime, "runtime_init", status);
    goto unload;
  }

  // 2. Load the model(s). A model's id is its index in the array.
  ModelConfig model = {.file_path = model_path,
                       .model_data = NULL,
                       .model_size = 0,
                       .config = {0, NULL, NULL}};
  if ((status = runtime.load_models(1, &model)) != RUNTIME_STATUS_SUCCESS) {
    report_error(&runtime, "runtime_load_models", status);
    goto cleanup;
  }
  printf("Runtime info: %s\n", runtime.get_info());

  // 3. Preprocess the image once; every request sends a copy of it.
  image = load_image(image_path, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_MEAN,
                     IMAGE_STD, true);
  if (image == NULL) goto cleanup;

  // 4. Enqueue the requests. enqueue_input doesn't block: the runtime runs the
  // inferences asynchronously. On success the runtime owns the input tensors,
  // on failure we still do and must free them.
  double start = now_ms();
  for (int id = 0; id < NUM_REQUESTS; id++) {
    Tensors *input = build_input_tensors(id, image);
    if (input == NULL) {
      fprintf(stderr, "Error: out of memory\n");
      goto cleanup;
    }
    status = runtime.enqueue_input(MODEL_ID, input);
    if (status != RUNTIME_STATUS_SUCCESS) {
      report_error(&runtime, "runtime_enqueue_input", status);
      free_tensors(input);
      goto cleanup;
    }
  }

  // 5. Retrieve the outputs. They may come back in any order: `id` ties each
  // one to its request. We own the outputs and must free them.
  for (int count = 0; count < NUM_REQUESTS; count++) {
    int model_id = -1;
    Tensors *output = NULL;
    status = runtime.retrieve_output(&model_id, &output, RETRIEVE_TIMEOUT_MS);
    if (status != RUNTIME_STATUS_SUCCESS) {
      report_error(&runtime, "runtime_retrieve_output", status);
      goto cleanup;
    }
    int id = output->id;
    if (model_id != MODEL_ID || id < 0 || id >= NUM_REQUESTS || received[id]) {
      fprintf(stderr, "Error: unexpected output (model %d, request %d)\n",
              model_id, id);
      free_tensors(output);
      goto cleanup;
    }
    received[id] = 1;
    if (count == NUM_REQUESTS - 1) print_output(output);  // Show the last one
    free_tensors(output);
  }

  double elapsed = now_ms() - start;
  printf("OK: %d/%d requests answered in %.1f ms (%.1f ms per request)\n",
         NUM_REQUESTS, NUM_REQUESTS, elapsed, elapsed / NUM_REQUESTS);
  exit_code = 0;

cleanup:
  // Stops the runtime's threads and drops pending inputs and outputs
  if ((status = runtime.cleanup()) != RUNTIME_STATUS_SUCCESS) {
    report_error(&runtime, "runtime_cleanup", status);
    exit_code = 1;
  }
unload:
  free(image);
  unload_runtime_library(&runtime);
  return exit_code;
}
