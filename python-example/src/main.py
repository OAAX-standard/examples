import argparse

import numpy as np

from runtime import OAAXRuntime
from utils import visualize_bboxes, preprocess_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OAX Python Runtime Example')
    parser.add_argument('--lib', type=str, required=True, help='Path to the shared library')
    parser.add_argument('--onnx', type=str, required=True, help='Path to the ONNX model')
    parser.add_argument('--image', type=str, required=True, help='Path to the image')
    args = parser.parse_args()
    lib_path = args.lib
    onnx_path = args.onnx
    image_path = args.image

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

    bboxes = list(output_data.values())[0]
    print(bboxes)
    visualize_bboxes(image_path, bboxes, 320, 240)

    # Free memory
    runtime.inference_cleanup()
    # Destroy the runtime session
    runtime.destroy()
    print('Exiting.')
