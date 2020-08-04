import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import sys
import matplotlib.pyplot as plt
from pathlib import Path

def LoadCV2Image(image_path):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, channels = image.shape

    assert channels == 3

    return image, height, width

def main():
    DATA_PATH = Path(r'C:/DS/data/Open Images Object Detection RVC 2020/')
    TRAIN_DATA_PATH = DATA_PATH / 'train_01 sample'
    image, height, width = LoadCV2Image(TRAIN_DATA_PATH / 'c0a0a84c9586b7f2.jpg')

    image_bytes = image.tobytes()

    image_decoded = tf.io.decode_raw(image_bytes, tf.uint8)

    #image_casted = tf.cast(image_decoded, tf.float32)

    image_numpy = image_decoded.numpy().reshape(width, height, 3)

    cv2.imwrite('saved_image.jpg', image_numpy, [cv2.IMWRITE_JPEG_QUALITY, 100])

if __name__ == "__main__":
    main()
