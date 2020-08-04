import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import os
import sys

from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

from ImagesTFRecordIO import LoadCV2Image

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

###################################################################################################
def LoadAndDecodeImage(tf_image_path):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    image = tf.io.read_file(tf_image_path)

    if image is None:
        return None

    image = tf.image.decode_jpeg(image, channels = 3)# compressed string -> 3D uint8 tensors
    image = tf.image.convert_image_dtype(image, tf.float32)# uint8 -> floats in [0, 1] range

    return tf.image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])

def ImageToBoundingBoxesGenerator(tf_filename, data):
    image = LoadAndDecodeImage(tf_filename)

    xmins = data[0::5]
    xmaxs = data[1::5]
    ymins = data[2::5]
    ymaxs = data[3::5]
    class_ids = data[4::5]

    bboxes = np.asarray(zip(xmins, xmaxs, ymins, ymaxs))

    return (image, bboxes)
    return (image, (bboxes, class_ids))

###################################################################################################
def GenerateImagesToBoundingBoxesDataset():
    bbox_files_list_df = tf.data.Dataset.list_files(str(INFO_DATA_PATH / "*.txt"))

    # file_path, [XMin, XMax, YMin, YMax, class_id]
    bboxes_df = tf.data.TextLineDataset(bbox_files_list_df, num_parallel_reads = 32)

    for line in bboxes_df:
        filename, data = line.numpy().decode('utf-8').split('.jpg ')

        filename += '.jpg'

        print(filename)

        bboxes_list = data.split(' ')

        list_items_count = len(bboxes_list)
        if list_items_count % 5 != 0:
            bboxes_list = bboxes_list[:(list_items_count // 5) * 5]

        yield ImageToBoundingBoxesGenerator(filename, bboxes_list)

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, images_path, config_file_path, debug = False):
        self.boxes = []
        self.debug = debug
        self.data_path = config_file_path

        if not os.path.isfile(config_file_path):
            print("File path {} does not exist. Existing...".format(config_file_path))
            sys.exit()

        if not os.path.isdir(images_path):
            print("Images folder path {} does not exist. Exiting...".format(images_path))
            sys.exit()

        with open(config_file_path) as f:
            image_info = f.readline()

            filename, data = image_info.split('.jpg ')

            filename += '.jpg'

            print(filename)

            bboxes_data = data.split(' ')

            list_items_count = len(bboxes_data)
            if list_items_count % 5 != 0:
                bboxes_data = bboxes_data[:(list_items_count // 5) * 5]

            image, bboxes_list = ImageToBoundingBoxesGenerator(filename, bboxes_data)

    def __len__(self):
        # Should return Sequence length

    def __getitem__(self, idx):
        # Returns preprocessed data

if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()

    DATA_PATH = Path(r'C:/DS/data/Open Images Object Detection RVC 2020/')
    TRAIN_DATA_PATH = DATA_PATH / 'train_01 sample'
    INFO_DATA_PATH = DATA_PATH / 'bounding_boxes'

    #dataset = tf.data.Dataset.from_generator(GenerateImagesToBoundingBoxesDataset, (tf.float32, tf.float32),#(tf.float32, tf.int32)),\
    #                                         (tf.TensorShape([224, 224, 3]), tf.TensorShape([None, 4])))#(tf.TensorShape([None, 4]), tf.TensorShape([None]))))

    base_model = tf.keras.applications.vgg19.VGG19(input_shape = (224, 224, 3), weights = 'imagenet', include_top = False)
    base_model.trainable = False

    print(base_model)

    intermediate_layer = tf.keras.layers.Dense(1000, activation = 'relu')(base_model.output)
    bboxes_output = tf.keras.layers.Dense(4, activation = 'relu')(intermediate_layer)
    #class_output = tf.keras.layers.Dense(599, activation = 'softmax')(intermediate_layer)

    model = tf.keras.Model(inputs = base_model.input, outputs = bboxes_output)#, class_output])
    #model.compile(loss = [tf.keras.losses.MSE, tf.keras.losses.SparseCategoricalCrossentropy], optimizer = tf.keras.optimizers.SGD, metrics = [tf.keras.metrics.Accuracy, tf.keras.metrics.SparseCategoricalAccuracy])
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    history = model.fit_generator(GenerateImagesToBoundingBoxesDataset, epochs = 1)
