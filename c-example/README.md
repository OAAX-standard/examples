# C example

A C program that uses an OAAX 2.0 runtime to run a model on an image: it loads the runtime library, loads the model,
sends 10 inference requests, matches every output to its request and prints the detections.

The `artifacts/` directory contains:

- `model.onnx`: a face locator model, i.e. it detects faces in an image.
- `image.jpg`: an image that contains faces.
- `libRuntimeLibrary_x86_64.so`, `libRuntimeLibrary_aarch64.so`, `RuntimeLibrary_x86_64.dll`: the OAAX 2.0 CPU runtime
  (ONNX Runtime based, v2.0.0) for Linux x86_64, Linux aarch64 and Windows x86_64.

## Requirements

- A C compiler, CMake 3.13+ and libjpeg: `sudo apt-get install cmake build-essential libjpeg-dev` (Ubuntu 20.04+).
- On Windows, a C compiler, CMake and Ninja. libjpeg is bundled in `deps/jpeg`.

## Build and run

```bash
bash build-run.sh        # Linux (picks the runtime matching your architecture)
.\build-run.bat          # Windows
```

Or step by step:

```bash
mkdir build && cd build
cmake .. && make
./c_example ./artifacts/libRuntimeLibrary_x86_64.so ./artifacts/model.onnx ./artifacts/image.jpg
```

Expected output (the runtime also writes `runtime.log`):

```
Runtime: OAAX CPU Runtime 2.0.0
Runtime info: {"loaded_models":1,"requests_in_flight":0,"backend_version":"1.17.3"}
Output tensor 'bboxes-format:xyxysc;0:face': type 1, shape [12, 6]
  face 1.000 at (0.651, 0.045) - (0.771, 0.297)
  ...
OK: 10/10 requests answered in 7.0 ms (0.7 ms per request)
```

The program exits with a non-zero code if anything fails, e.g. if an output doesn't match a request.

## How it works

| File | Role |
| --- | --- |
| `deps/oaax/interface.h` | The OAAX 2.0 runtime interface (copied from the specification). |
| `src/runtime.c` | Loads the runtime library and resolves the OAAX functions (`dlopen` / `LoadLibrary`). |
| `src/tensors.c` | Allocates, fills and frees `Tensors`. |
| `src/image.c` | Loads a JPEG, resizes and normalizes it. |
| `src/main.c` | The OAAX lifecycle, below. |

The call sequence in `main.c`:

1. `runtime_init(config)`: starts the runtime. Config keys and values are strings; the keys a runtime understands are
   vendor-specific and unknown keys are ignored.
2. `runtime_load_models(n, configs)`: loads the models, from a file path or from memory. A model's id is its index in
   the array.
3. `runtime_enqueue_input(model_id, tensors)`: queues a request and returns immediately. Set a unique `Tensors.id` on
   each request.
4. `runtime_retrieve_output(&model_id, &tensors, timeout_ms)`: waits for a result (`0` doesn't wait, a negative value
   waits indefinitely). Outputs can come back in any order: the runtime copies the request's `Tensors.id` onto the
   output, which is how you match them.
5. `runtime_cleanup()`: stops the runtime's threads and drops pending work. Always call it before unloading the library.

Enqueueing and retrieving are thread-safe, so they can also be done from different threads.

### Who frees what

| Memory | Freed by |
| --- | --- |
| Input `Tensors` after a successful `runtime_enqueue_input` | The runtime |
| Input `Tensors` after a failed `runtime_enqueue_input` | You |
| Output `Tensors` from `runtime_retrieve_output` | You: every `name`, `shape` and `data`, then `tensors`, then the struct (`free_tensors`) |
| `Config`, `ModelConfig` and what they point to | You (the runtime only borrows them during the call) |
| Strings from `runtime_get_error` / `runtime_get_info` | The runtime (valid until the next runtime call) |

On any status other than `RUNTIME_STATUS_SUCCESS`, `runtime_get_error()` describes what went wrong.

## Adapting the example

- **Another runtime**: pass its library path as the first argument. It must be built for your OS and architecture, and
  the model must be optimized for it. Runtime-specific options go in the `init_keys` / `init_values` arrays in `main.c`.
- **Another model or image**: update the constants at the top of `main.c` (input size, normalization, NMS sensitivity)
  and `build_input_tensors`, so the tensor names, types and shapes match the model's inputs. `print_output` shows how to
  read the output tensors.
- The last line of `CMakeLists.txt` copies `artifacts/` to the build directory; remove or change it for your own files.
