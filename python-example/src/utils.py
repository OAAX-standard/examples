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
    # check bboxes[:, :4} is in the range of [0, 1]
    if bboxes[:, :4].max() <= 1 and bboxes[:, :4].min() >= 0:
        bboxes[:, :4] = bboxes[:, :4] * [width, height, width, height]
    for bbox in bboxes:
        x1, y1, x2, y2, score, label = bbox
        x1, y1, x2, y2, label = map(lambda x: int(x), [x1, y1, x2, y2, label])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('image', image)
    cv2.waitKey(0)
