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
python src/main.py --lib artifacts/libRuntimeLibrary.so --onnx artifacts/model.onnx --image artifacts/image.jpg

```

The Python code will load the OAX runtime shared library and create a runtime variable that can be used to interact with
the low-level runtime API.

The runtime loads an ONNX model from the `artifacts/model.onnx` directory, and runs inference on a test image available
in the same directory.

The output of the model will be printed to the console, and the image will be displayed with the detected faces.

## Indepth explanation

The `main.py` file contains the main function of the program. It will load the runtime library, load the model, and run
it on the input image. The script expects three arguments:
- The path to the runtime library shared object file.
- The path to the optimized model.
- The path to the input image.

For the sake of example, we've included a sample runtime library, runtime and image in the `artifacts/` directory.


### Adapting the example

To adapt the example to your use case, you can modify the `main.py` file to load a different model, or run inference on 
a different image. You can also modify the `requirements.txt` file to include any additional dependencies that your 
Python code may require.

