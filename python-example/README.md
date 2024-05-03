# Python example

The Python example is used to demonstrate how an OAX runtime can be loaded during runtime, and used to run inference on
any OAX-compliant AI Accelerator.

## Requirements

Before running the example, make sure you have the following installed:

- Python 3.8 or higher
- A compatible OAX runtime shared library (libRuntimeLibrary.so) built for your target architecture.
- An optimized ONNX model file that can be run on the OAX runtime.
- The OAX runtime shared library and the optimized model file should be placed in the `artifacts/` directory.

## Getting started

To run the Python example, follow these steps:

```bash
pip install -r requirements.txt
python src/main.py
```

The Python code will load the OAX runtime shared library and create a runtime variable that can be used to interact with
the low-level runtime API.

The runtime loads an ONNX model from the `artifacts/` directory, and runs inference on a test image available in the
same directory.