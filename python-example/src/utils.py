import cv2
import numpy as np
from PIL import Image


def preprocess_image(image_path, width, height, means, stds):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((width, height))
    image = np.array(image).astype('float32')
    image = (image - means) / stds
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))  # NHWC -> NCHW
    return image


def visualize_bboxes(image_path, bboxes, width, height, color=(0, 255, 0), thickness=2):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (width, height))
    bboxes[:, :4] = bboxes[:, :4] * [width, height, width, height]
    for bbox in bboxes:
        x1, y1, x2, y2, score, label = bbox
        x1, y1, x2, y2 = map(lambda x: int(x), [x1, y1, x2, y2])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow('image', image)
    cv2.waitKey(0)
