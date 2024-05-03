from os.path import dirname, join

import numpy as np

from runtime import OAXRuntime
from utils import visualize_bboxes, preprocess_image

if __name__ == '__main__':
    _here = dirname(__file__)
    artifacts_path = join(_here, '..', 'artifacts')

    lib_path = join(artifacts_path, 'libRuntimeLibrary.so')
    onnx_path = join(artifacts_path, 'model.onnx')
    image_path = join(artifacts_path, 'image.jpg')

    runtime = OAXRuntime(lib_path)
    rt_name = runtime.name()
    print(f'Runtime name: {rt_name}')
    rt_version = runtime.version()
    print(f'Runtime version: {rt_version}')
    runtime.initialize()
    runtime.load_model(onnx_path)
    # prepare the input_tensors
    image = preprocess_image(image_path, 320, 240, 127, 128)
    input_tensors = {'image-': image, 'nms_sensitivity': np.array([0.5], dtype='float32')}

    # Run the inference
    output_data = runtime.inference(input_tensors)

    bboxes = list(output_data.values())[0]
    visualize_bboxes(image_path, bboxes, 320, 240)
    runtime.inference_cleanup()
    runtime.destroy()
    print('Exiting.')
