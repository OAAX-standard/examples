import struct
import time
from os.path import dirname, abspath, join

import cv2
import msgpack
import numpy as np
from PIL import Image

_here = dirname(abspath(__file__))

# Initialize vars
engine_pipe_name = join(_here, 'engine_pipe')
module_pipe_name = join(_here, 'module_pipe')
shm_path = 1

input_image_path = join(_here, 'image.jpg')  # TODO: Change this to the path of your image
output_image_path = join(_here, 'output.jpg')
height, width, means, stds, nchw = 240, 320, 127, 128, True  # TODO: Change these values to match your model


def get_input_image():
    image = Image.open(input_image_path).convert('RGB')
    image = image.resize((width, height))
    image = np.array(image)
    image = (image - means) / stds
    image = image.astype('float32')
    image = np.expand_dims(image, axis=0)
    if nchw:
        image = np.transpose(image, (0, 3, 1, 2))  # NHWC -> NCHW
    return image


def build_input_message():
    image = get_input_image()  # NCHW
    height, width = image.shape[2], image.shape[3]
    nms_sensitivity = np.array([0.5], dtype='float32')
    mask = np.ones((height, width), dtype='bool')

    num_tensors = 2
    input_shapes = [image.shape, nms_sensitivity.shape]
    input_ranks = [len(shape) for shape in input_shapes]
    input_shapes = [*image.shape, *nms_sensitivity.shape, *mask.shape]

    # image to bytes
    image_list = image.flatten().tolist()
    image_bytes = struct.pack("f" * len(image_list), *image_list)
    # nms to bytes
    nms_list = nms_sensitivity.flatten().tolist()
    nms_bytes = struct.pack("f" * len(nms_list), *nms_list)

    input_data = [image_bytes, nms_bytes]
    # msgpack data: [number of tensors, tensors, 'json', True, tensor ranks, tensor shapes]
    input_message = [num_tensors]
    for input_d in input_data:
        input_message.append(input_d)
    input_message.extend(['json', True, *input_ranks, *input_shapes])
    packer: msgpack.Packer = msgpack.Packer(use_bin_type=True)
    msgpack_data: bytearray = bytearray()
    for val in input_message:
        packed_val = packer.pack(val)
        msgpack_data.extend(packed_val)
    return msgpack_data


def parse_output_message(byte_data):
    output = msgpack.unpackb(byte_data)
    outputs = output['Outputs']  # dict
    output_ranks = output['OutputRanks']
    output_shapes = output['OutputShapes']
    output_dtypes = output['OutputDataTypes']

    output_sizes = [np.prod(shape) for shape in output_shapes]
    arrays = [struct.unpack("f" * size, output) for size, output in zip(output_sizes, outputs.values())]
    arrays = [np.array(array).reshape(shape) for array, shape in zip(arrays, output_shapes)]
    return arrays


def visualize_bboxes(bboxes, img_path, width, height):
    if not (bboxes.shape[1] == 6 and len(bboxes.shape) == 2):
        raise ValueError('Invalid bboxes shape. Expected (N, 6) where N is the number of bboxes.')
    # check if bboxes are in [-0.1, 1.1] range
    if np.all(bboxes[:, :4] < 1.1) and np.all(bboxes[:, :4] > -0.1):
        bboxes[:, :4] = bboxes[:, :4] * np.array([width, height, width, height])

    img = cv2.imread(img_path)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    for bbox in bboxes:
        x1, y1, x2, y2, score, class_id = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{int(class_id)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # save the image
    cv2.imwrite(output_image_path, img)


# Create the pipes
import os

if os.path.exists(engine_pipe_name):
    os.unlink(engine_pipe_name)
os.mkfifo(engine_pipe_name, mode=0o666)

if os.path.exists(module_pipe_name):
    os.unlink(module_pipe_name)
os.mkfifo(module_pipe_name, mode=0o666)

# Create the shared memory
import sysv_ipc

shm_key = sysv_ipc.ftok('/tmp', shm_path)
shm = sysv_ipc.SharedMemory(shm_key, flags=sysv_ipc.IPC_CREAT | 0o666, size=1024 * 1024 * 10)
shm_id = shm.id

# Print engine connection info and wait for the engine to connect
print(
    f'Engine paramters (engine_pipe_name, module_pipe_name, shm_id, shm_key): {engine_pipe_name} {module_pipe_name} {shm_id} {shm_key}')

# Wait for the engine to connect
print("Opening pipe to engine...")
with open(engine_pipe_name, 'wb') as pipe:
    pipe.write(bytes(1))
    pipe.flush()

# Wait for the engine to connect
print("Waiting for the engine to connect...")
with open(module_pipe_name, 'rb') as pipe:
    pipe.read(1)

# Wait for the engine to connect
try:
    print("Engine connected. Starting the module...")
    while True:
        # print('Creating input message...')
        # Send data to the engine through shm
        msg = build_input_message()
        msg_len = struct.pack("<I", len(msg))
        # Send through shared memory
        shm.write(msg_len)
        shm.write(msg, offset=4)
        print(f'Sent {len(msg)} bytes to shared memory')

        # Signal the engine to start
        # print("Signaling the engine to start...")
        with open(engine_pipe_name, 'wb') as f:
            f.write(b'a')
            f.flush()

        # Start wait for signal from engine
        # print("Waiting for the engine to finish...")
        with open(module_pipe_name, 'rb') as f:
            f.read(1)

        # Get the output from the shared memory
        # print("Reading output from shared memory...")
        output_len = shm.read(4)
        output_len = struct.unpack("<I", output_len)[0]
        output = shm.read(output_len, offset=4)
        print(f'Received {output_len} bytes from shared memory')
        arrays = parse_output_message(output)

        # Visualize the output
        if not (arrays is None or len(arrays) == 0):
            bboxes = arrays[0]
            visualize_bboxes(bboxes, input_image_path, width, height)
        else:
            print('No valid output received')

        print('Sleeping for 10 seconds...')
        time.sleep(10)

except KeyboardInterrupt:
    pass

# Clean up
os.remove(engine_pipe_name)
os.remove(module_pipe_name)
shm.detach()
