# Parallel models

This Python example is used to demonstrate how two OAX runtimes can be loaded at the same time, and used to run
inference simultaneously.

## Overview

This example shows how to run two ONNX models at the same time using two different runtimes. The first model is a face
locator model, which will detect faces in an image, and the second model is an object detector model, which will detect
objects in an image.

The program will load the two runtimes, load the two models, and run them on the same input image. The output of the two
models will be visualized on the input image.

The `artifacts/` directory contains the following files:

- `face-locator.onnx`: an optimized ONNX model, which is a face locator model, ie. it will detect faces in an image.
- `object-detecot.onnx`: another optimized ONNX model, which is an object detector model, ie. it will detect objects (80
  different classes) in an image.
- `libRuntimeLibrary.so`: the runtime library shared object file.
- `image.jpg`: an image that contains faces, which will be used as input to the two models.

> The source code is available in the `src/` directory.

## Getting started

Before delving into this example, make sure that you have looked at the [Python example](../python-example/README.md)
and
have the necessary requirements installed.

To run the parallel models example, follow these steps:

```bash
pip install -r requirements.txt
python src/main.py --lib artifacts/libRuntimeLibrary.so --onnx1 artifacts/face-locator.onnx --onnx2 artifacts/object-detector.onnx --image artifacts/image.jpg
```

## Indepth explanation

The Python code is capable of running the two models simultaneously, by creating two `ThreadedRuntime` objects,
where each of them runs inference on a separate thread. The code snippet below shows how the two runtimes are
used to run inference on the input image:

```python
runtime1.inference(model1_tensors)
runtime2.inference(model2_tensors)
model1_outputs = runtime1.get_last_result()
model2_outputs = runtime2.get_last_result()
```

The Python code instructs the two OAX runtimes to run inference on the input image, and then waits for the inference
results by calling `get_last_result()` on each runtime to get the output tensors when the inference is done.
