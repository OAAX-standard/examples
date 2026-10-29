// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

// Example OAAX 2.0 host application for YOLOX: loads any OAAX runtime library
// and a model converted by that runtime's toolchain, runs a YOLOX detector on
// an image and prints (and draws) the detections.
//
// Usage: yolox_host <runtime library> <model> <image> [options]  (--help)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "image.h"    // NOLINT[build/include]
#include "runtime.h"  // NOLINT[build/include]
#include "tensors.h"  // NOLINT[build/include]
#include "yolox.h"    // NOLINT[build/include]

#define MODEL_ID 0  // Index of the model in the array given to load_models
#define MAX_CONFIG 32

static const char* const kUsage =
    "Usage: %s <runtime library> <model> <image> [options]\n"
    "\n"
    "Runtime options:\n"
    "  --config KEY=VALUE   Option for runtime_init, repeatable (see your\n"
    "                       runtime's page, e.g. --config device_type=GPU).\n"
    "                       Defaults: log_level=3, log_file=runtime.log\n"
    "  --runs N             Number of inference requests (default 10)\n"
    "  --timeout MS         Max wait for each output (default 30000)\n"
    "What the runtime's converted model expects (the defaults suit the CPU,\n"
    "Intel and NVIDIA runtimes):\n"
    "  --input-name NAME    Input tensor name (default images)\n"
    "  --layout nchw|nhwc   Input layout (default nchw)\n"
    "  --dtype float32|float16|uint8\n"
    "                       Input type (default float32), values 0-255\n"
    "  --size N             Model input size, a multiple of 32 (default 640;\n"
    "                       416 for yolox_nano and yolox_tiny)\n"
    "Detections:\n"
    "  --conf X             Score threshold (default 0.3)\n"
    "  --nms X              NMS IoU threshold (default 0.45)\n"
    "  --output FILE        Image with the detections drawn, .png or .jpg\n"
    "                       (default detections.png; empty = none)\n";

typedef struct Options {
  const char *runtime_path, *model_path, *image_path;
  const char *config_keys[MAX_CONFIG], *config_values[MAX_CONFIG];
  int num_config;
  int runs, timeout_ms;
  const char* input_name;
  int nhwc, size;
  TensorElementType dtype;  // float32, float16 or uint8
  float conf, nms;
  const char* output_path;
} Options;

static double now_ms(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// Adds `key=value` to the runtime options unless the user gave that key.
static void add_default_config(Options* o, const char* key, const char* value) {
  for (int i = 0; i < o->num_config; i++) {
    if (strcmp(o->config_keys[i], key) == 0) return;
  }
  if (o->num_config == MAX_CONFIG) return;
  o->config_keys[o->num_config] = key;
  o->config_values[o->num_config++] = value;
}

// Parses the command line. Returns 0 on success, -1 on error, 1 for --help.
static int parse_args(int argc, char** argv, Options* o) {
  *o = (Options){.runs = 10,
                 .timeout_ms = 30000,
                 .input_name = "images",
                 .dtype = DATA_TYPE_FLOAT,
                 .size = 640,
                 .conf = 0.3f,
                 .nms = 0.45f,
                 .output_path = "detections.png"};
  int positional = 0;
  for (int i = 1; i < argc; i++) {
    const char* arg = argv[i];
    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) return 1;
    if (strncmp(arg, "--", 2) != 0) {
      const char** slot[] = {&o->runtime_path, &o->model_path, &o->image_path};
      if (positional == 3) {
        fprintf(stderr, "Error: unexpected argument '%s'\n", arg);
        return -1;
      }
      *slot[positional++] = arg;
      continue;
    }
    if (i + 1 >= argc) {
      fprintf(stderr, "Error: %s needs a value\n", arg);
      return -1;
    }
    char* value = argv[++i];
    if (strcmp(arg, "--config") == 0) {
      char* equals = strchr(value, '=');
      if (equals == NULL || equals == value || o->num_config == MAX_CONFIG) {
        fprintf(stderr, "Error: --config needs KEY=VALUE (got '%s')\n", value);
        return -1;
      }
      *equals = '\0';
      o->config_keys[o->num_config] = value;
      o->config_values[o->num_config++] = equals + 1;
    } else if (strcmp(arg, "--runs") == 0) {
      o->runs = atoi(value);
    } else if (strcmp(arg, "--timeout") == 0) {
      o->timeout_ms = atoi(value);
    } else if (strcmp(arg, "--input-name") == 0) {
      o->input_name = value;
    } else if (strcmp(arg, "--layout") == 0) {
      if (strcmp(value, "nchw") != 0 && strcmp(value, "nhwc") != 0) {
        fprintf(stderr, "Error: --layout is nchw or nhwc\n");
        return -1;
      }
      o->nhwc = strcmp(value, "nhwc") == 0;
    } else if (strcmp(arg, "--dtype") == 0) {
      if (strcmp(value, "float32") == 0) {
        o->dtype = DATA_TYPE_FLOAT;
      } else if (strcmp(value, "float16") == 0) {
        o->dtype = DATA_TYPE_FLOAT16;
      } else if (strcmp(value, "uint8") == 0) {
        o->dtype = DATA_TYPE_UINT8;
      } else {
        fprintf(stderr, "Error: --dtype is float32, float16 or uint8\n");
        return -1;
      }
    } else if (strcmp(arg, "--size") == 0) {
      o->size = atoi(value);
    } else if (strcmp(arg, "--conf") == 0) {
      o->conf = (float)atof(value);
    } else if (strcmp(arg, "--nms") == 0) {
      o->nms = (float)atof(value);
    } else if (strcmp(arg, "--output") == 0) {
      o->output_path = value;
    } else {
      fprintf(stderr, "Error: unknown option '%s'\n", arg);
      return -1;
    }
  }
  if (positional != 3) {
    fprintf(stderr, "Error: runtime library, model and image are required\n");
    return -1;
  }
  // Quiet runtime unless told otherwise (log_level is the usual key)
  add_default_config(o, "log_level", "3");
  add_default_config(o, "log_file", "runtime.log");
  if (o->runs < 1 || o->size < 32 || o->size % 32 != 0) {
    fprintf(stderr, "Error: --runs must be >= 1 and --size a multiple of 32\n");
    return -1;
  }
  return 0;
}

// The IEEE half-precision bits of an integer 0-255 (exact: no rounding).
static uint16_t half_from_uint8(uint8_t value) {
  if (value == 0) return 0;
  int exponent = 0;
  while ((2 << exponent) <= value) exponent++;  // 2^exponent <= value
  const int mantissa = (value - (1 << exponent)) << (10 - exponent);
  return (uint16_t)(((exponent + 15) << 10) | mantissa);
}

// Converts the BGR letterboxed image (HWC, uint8) to the layout and type the
// model takes. Returns a buffer to free, `*bytes` long, or NULL.
static void* make_input_data(const uint8_t* bgr, const Options* o,
                             size_t* bytes) {
  const size_t pixels = (size_t)o->size * o->size;
  const size_t element = o->dtype == DATA_TYPE_FLOAT     ? sizeof(float)
                         : o->dtype == DATA_TYPE_FLOAT16 ? sizeof(uint16_t)
                                                         : sizeof(uint8_t);
  *bytes = pixels * 3 * element;
  uint8_t* data = (uint8_t*)malloc(*bytes);
  if (data == NULL) return NULL;
  for (size_t p = 0; p < pixels; p++) {
    for (size_t c = 0; c < 3; c++) {
      const size_t target = o->nhwc ? p * 3 + c : c * pixels + p;
      const uint8_t value = bgr[p * 3 + c];  // Raw 0-255, no mean or std
      if (o->dtype == DATA_TYPE_FLOAT) {
        ((float*)data)[target] = (float)value;
      } else if (o->dtype == DATA_TYPE_FLOAT16) {
        ((uint16_t*)data)[target] = half_from_uint8(value);
      } else {
        data[target] = value;
      }
    }
  }
  return data;
}

// The tensor names are those of the model's ONNX graph; an optimized model may
// rename them (see --input-name). The data is copied: once enqueued, the
// runtime owns and frees the tensors.
static Tensors* build_input(int id, const Options* o, const void* data,
                            size_t bytes) {
  const int chw[] = {1, 3, o->size, o->size};
  const int hwc[] = {1, o->size, o->size, 3};
  Tensors* input = alloc_tensors(id, 1);
  if (input == NULL) return NULL;
  if (fill_tensor(&input->tensors[0], o->input_name, o->dtype, 4,
                  o->nhwc ? hwc : chw, data, bytes) != 0) {
    free_tensors(input);
    return NULL;
  }
  return input;
}

static const char* dtype_name(TensorElementType dtype) {
  return dtype == DATA_TYPE_FLOAT     ? "float32"
         : dtype == DATA_TYPE_FLOAT16 ? "float16"
                                      : "uint8";
}

// Reports a failed call and, when the cause can be the input we sent, the
// flags that describe it.
static void print_failure(const Runtime* runtime, const char* what,
                          RuntimeStatus status, const Options* o) {
  report_error(runtime, what, status);
  const int no_output = status == RUNTIME_STATUS_NO_OUTPUT_AVAILABLE ||
                        status == RUNTIME_STATUS_TIMEOUT;
  if (no_output) {
    // A request the runtime cannot make sense of (e.g. a wrong input name) is
    // accepted by runtime_enqueue_input and simply never answered
    fprintf(stderr,
            "No output for the request. If the runtime is busy, raise "
            "--timeout (now %d ms).\nOtherwise it may not accept the input.\n",
            o->timeout_ms);
  } else if (status != RUNTIME_STATUS_INVALID_ARGUMENT &&
             status != RUNTIME_STATUS_INVALID_TENSOR &&
             status != RUNTIME_STATUS_TENSOR_SHAPE_MISMATCH &&
             status != RUNTIME_STATUS_TENSOR_TYPE_MISMATCH) {
    return;
  }
  fprintf(stderr,
          "Check what the model expects (see `Runtime info` above and the "
          "runtime's log) and pass it with\n  --input-name (now '%s'), "
          "--layout (%s), --dtype (%s), --size (%d)\n",
          o->input_name, o->nhwc ? "nhwc" : "nchw", dtype_name(o->dtype),
          o->size);
}

static void print_shape(const TensorDescriptor* t) {
  printf("[");
  for (int d = 0; d < t->rank; d++) printf(d ? ", %d" : "%d", t->shape[d]);
  printf("]");
}

// Finds the YOLOX output: float32, [1, N, 85] or [N, 85].
static const TensorDescriptor* find_output(const Tensors* output) {
  for (int i = 0; i < output->num_tensors; i++) {
    const TensorDescriptor* t = &output->tensors[i];
    if (t->data_type == DATA_TYPE_FLOAT && t->rank >= 2 &&
        t->shape[t->rank - 1] == YOLOX_ROW_SIZE &&
        t->data_size % (YOLOX_ROW_SIZE * sizeof(float)) == 0) {
      return t;
    }
  }
  return NULL;
}

static int print_detections(const Tensors* output, const Options* o,
                            uint8_t* rgb, int width, int height, float scale) {
  for (int i = 0; i < output->num_tensors; i++) {
    const TensorDescriptor* t = &output->tensors[i];
    printf("Output tensor '%s': type %d, shape ", t->name, (int)t->data_type);
    print_shape(t);
    printf("\n");
  }
  const TensorDescriptor* t = find_output(output);
  if (t == NULL) {
    fprintf(stderr,
            "Error: no float32 [1, N, %d] output (the raw YOLOX head). Is "
            "this an official YOLOX model without NMS?\n",
            YOLOX_ROW_SIZE);
    return -1;
  }
  const int rows = (int)(t->data_size / (YOLOX_ROW_SIZE * sizeof(float)));
  Detection* detections = (Detection*)malloc(sizeof(Detection) * rows);
  if (detections == NULL) return -1;
  const int count = yolox_decode((const float*)t->data, rows, o->size, scale,
                                 width, height, o->conf, o->nms, detections);
  if (count < 0) {
    fprintf(stderr, "Error: the output has %d rows but --size %d needs %d\n",
            rows, o->size, yolox_num_anchors(o->size));
    free(detections);
    return -1;
  }
  printf("Detections (score >= %.2f, NMS IoU %.2f): %d\n", o->conf, o->nms,
         count);
  for (int i = 0; i < count; i++) {
    const Detection* d = &detections[i];
    printf("  %-14s %.3f  (%4.0f, %4.0f) - (%4.0f, %4.0f)\n",
           YOLOX_CLASS_NAMES[d->class_id], d->score, d->x1, d->y1, d->x2,
           d->y2);
  }
  int status = 0;
  if (o->output_path[0] != '\0' &&
      save_detections_png(o->output_path, rgb, width, height, detections,
                          count) == 0) {
    printf("Wrote %s\n", o->output_path);
  } else if (o->output_path[0] != '\0') {
    status = -1;
  }
  free(detections);
  return status;
}

int main(int argc, char** argv) {
  Options o;
  const int parsed = parse_args(argc, argv, &o);
  if (parsed != 0) {
    fprintf(parsed > 0 ? stdout : stderr, kUsage, argv[0]);
    return parsed > 0 ? 0 : 1;
  }

  // 1. Preprocess the image once, like the official YOLOX ONNX demo: letterbox
  // to the model size (gray padding, BGR), values stay 0-255. Every request
  // sends a copy of it.
  int width, height;
  uint8_t* rgb = load_image(o.image_path, &width, &height);
  if (rgb == NULL) return 1;
  uint8_t* canvas = (uint8_t*)malloc((size_t)o.size * o.size * 3);
  size_t input_bytes = 0;
  void* input_data = NULL;
  float scale = 1;
  if (canvas != NULL) {
    scale = yolox_letterbox(rgb, width, height, o.size, canvas);
    input_data = make_input_data(canvas, &o, &input_bytes);
  }
  if (input_data == NULL) {
    fprintf(stderr, "Error: out of memory\n");
    free(canvas);
    free(rgb);
    return 1;
  }

  Runtime runtime;
  int exit_code = 1;
  Tensors* last_output = NULL;
  RuntimeStatus status;
  if (load_runtime_library(o.runtime_path, &runtime) != 0) goto free_buffers;
  printf("Runtime: %s %s\n", runtime.get_name(), runtime.get_version());

  // 2. Initialize the runtime. Keys and values are strings; the keys a runtime
  // understands are vendor-specific and unknown keys are ignored.
  Config init_config = {o.num_config, o.config_keys, o.config_values};
  if ((status = runtime.init(init_config)) != RUNTIME_STATUS_SUCCESS) {
    report_error(&runtime, "runtime_init", status);
    goto unload;
  }

  // 3. Load the model. A model's id is its index in the array.
  ModelConfig model = {.file_path = o.model_path,
                       .model_data = NULL,
                       .model_size = 0,
                       .config = {0, NULL, NULL}};
  if ((status = runtime.load_models(1, &model)) != RUNTIME_STATUS_SUCCESS) {
    report_error(&runtime, "runtime_load_models", status);
    goto cleanup;
  }
  printf("Runtime info: %s\n", runtime.get_info());
  printf("Input: '%s' %s %s, %dx%d image letterboxed to %d\n", o.input_name,
         dtype_name(o.dtype), o.nhwc ? "NHWC" : "NCHW", width, height, o.size);

  // 4. Send the requests one by one: enqueue_input doesn't block, and
  // retrieve_output waits for the result. Outputs of concurrent requests may
  // come back in any order: `id` ties each one to its request.
  double first_ms = 0, rest_ms = 0, rest_min = 1e300;
  for (int id = 0; id < o.runs; id++) {
    Tensors* input = build_input(id, &o, input_data, input_bytes);
    if (input == NULL) {
      fprintf(stderr, "Error: out of memory\n");
      goto cleanup;
    }
    const double start = now_ms();
    status = runtime.enqueue_input(MODEL_ID, input);
    if (status != RUNTIME_STATUS_SUCCESS) {
      print_failure(&runtime, "runtime_enqueue_input", status, &o);
      free_tensors(input);  // Not accepted: still ours
      goto cleanup;
    }
    int model_id = -1;
    Tensors* output = NULL;
    status = runtime.retrieve_output(&model_id, &output, o.timeout_ms);
    if (status != RUNTIME_STATUS_SUCCESS) {
      print_failure(&runtime, "runtime_retrieve_output", status, &o);
      goto cleanup;
    }
    const double elapsed = now_ms() - start;
    if (model_id != MODEL_ID || output->id != id) {
      fprintf(stderr, "Error: unexpected output (model %d, request %d)\n",
              model_id, output->id);
      free_tensors(output);
      goto cleanup;
    }
    if (id == 0) first_ms = elapsed;
    if (id > 0) rest_ms += elapsed;
    if (id > 0 && elapsed < rest_min) rest_min = elapsed;
    free_tensors(last_output);
    last_output = output;  // Keep the last one, we own it
  }

  // 5. Decode the last output into boxes
  if (print_detections(last_output, &o, rgb, width, height, scale) != 0) {
    goto cleanup;
  }
  if (o.runs > 1) {
    printf(
        "OK: %d requests, first %.1f ms (warm-up), then %.1f ms on average "
        "(min %.1f ms)\n",
        o.runs, first_ms, rest_ms / (o.runs - 1), rest_min);
  } else {
    printf("OK: 1 request, %.1f ms (includes warm-up)\n", first_ms);
  }
  exit_code = 0;

cleanup:
  // Stops the runtime's threads and drops pending inputs and outputs
  if ((status = runtime.cleanup()) != RUNTIME_STATUS_SUCCESS) {
    report_error(&runtime, "runtime_cleanup", status);
    exit_code = 1;
  }
unload:
  free_tensors(last_output);
  unload_runtime_library(&runtime);
free_buffers:
  free(input_data);
  free(canvas);
  free(rgb);
  return exit_code;
}
