# Python example

The Python example is used to demonstrate how an OAAX runtime can be loaded during runtime, and used to run inference on
any OAAX-compliant AI Accelerator.

## Requirements

Before running the example, make sure you have the following installed:

- An Ubuntu 18.04 or later x86_64 machine
- Python 3.8 or higher

For the sake of example, we've included a sample runtime library, runtime and image in the `artifacts/` directory.

## Getting started

To run the Python example, follow these steps:

```bash
pip install -r requirements.txt
python src/main.py --lib artifacts/libRuntimeLibrary.so --onnx artifacts/model.onnx --image artifacts/image.jpg

```

The Python code will load the OAAX runtime shared library and create a runtime variable that can be used to interact with
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

Those arguments are used to create a `Runtime` object, which is used to run inference on the input image, as shown in the 
code snippet below:
```python
# Load the runtime library from `lib_path`
runtime = OAAXRuntime(lib_path)

rt_name = runtime.name
print(f'Runtime name: {rt_name}')

rt_version = runtime.version
print(f'Runtime version: {rt_version}')

# Initialize the runtime
runtime.initialize()
# Load the model
runtime.load_model(onnx_path)
# prepare the input_tensors
image = preprocess_image(image_path, 320, 240, 127, 128)
input_tensors = {'image-': image, 'nms_sensitivity': np.array([0.5], dtype='float32')}

# Run the inference
output_data = runtime.inference(input_tensors)
```

The `src/runtime.py` file contains the `Runtime` class, which is used to interact with the OAAX runtime and manage the
data structure so that it's compatible for the C runtime and Python based on who's using it. 
For example, the `Runtime` class will convert the input image to a format that the runtime can understand, and convert 
the output of the model to a format that can be used by Python as shown in the code snippet below:
```python
def inference(self, input_data: Dict[str, np.ndarray]):
    input_tensors = numpy_to_c_struct(input_data)
    output_tensors = tensors_struct()
    exit_code = self.lib.runtime_inference_execution(input_tensors, output_tensors)
    output_data = c_struct_to_numpy(output_tensors)
    return output_data
```

The `src/runtime_utils.py` file contains utility functions that are used to convert data between Python and C data structures:
`numpy_to_c_struct` and `c_struct_to_numpy`.


### Adapting the example

To adapt the example to your use case, you can modify the `main.py` file to load a different model, or run inference on
a different image. You can also modify the `requirements.txt` file to include any additional dependencies that your
Python code may require.

