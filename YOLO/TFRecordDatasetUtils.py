import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import datetime
import sys

###################################################################################################
def GetBytesFeature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

###################################################################################################
def GetFloatFeature(value):
    """Returns a float_list from a float / double."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

###################################################################################################
def GetIntFeature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

###################################################################################################
def LoadCV2Image(image_path):
    # cv2 load images as BGR, convert it to RGB
    image = cv2.imread(str(image_path))

    height, width, channels = image.shape

    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image, width, height, channels

###################################################################################################
def GenerateTFRecordsFromCSV(images_data_file_path, desctination_data_path, image_shape, images_per_record_file = 200, max_number_of_record_files = -1):
    Path(desctination_data_path).mkdir(parents=True, exist_ok=True)

    # image_path, class_name, class_id
    images_df = pd.read_csv(images_data_file_path)

    tf_record_file_index = 0
    images_count = len(images_df)

    for row_index, row_data in images_df.iterrows():
        if row_index % images_per_record_file == 0:
            current_time = datetime.datetime.now()
            writer = tf.io.TFRecordWriter('{}\{}.tfrecords'.format(desctination_data_path, tf_record_file_index))
            tf_record_file_index += 1

        image_path = row_data['image_path']
        image, image_width, image_height, image_channels = LoadCV2Image(image_path)

        if image_channels != image_shape[2]:
            print('Image channels number is {} instead of {}'.format(image_channels, image_shape[2]))

        if image_width != image_shape[0] or image_height != image_shape[1]:
            image = cv2.resize(image, image_shape[:2], interpolation=cv2.INTER_CUBIC)

        class_id = row_data['class_id']

        if image is None:
            print('Empty image: {}'.format(image_path))
            continue

        features =\
                {
                'image_data': GetBytesFeature(image.tobytes()),
                'image_width': GetIntFeature(image_shape[0]),
                'image_height': GetIntFeature(image_shape[1]),
                'image_channels': GetIntFeature(image_channels),
                'class_id': GetIntFeature(class_id),
                }

        example = tf.train.Example(features=tf.train.Features(feature=features))

        writer.write(example.SerializeToString())

        if row_index % images_per_record_file == images_per_record_file - 1 or row_index == images_count - 1:
            print('step number {} at {}'.format(row_index, datetime.datetime.now() - current_time))
            print('Saved data: {}/{}'.format(row_index + 1, images_count))
            sys.stdout.flush()
            writer.close()

            if max_number_of_record_files > 0 and tf_record_file_index >= max_number_of_record_files:
                break

###################################################################################################
def ParseExample(record):
    features =\
        {
        'image_data': tf.io.FixedLenFeature([], tf.string),
        'image_width': tf.io.FixedLenFeature([], tf.int64),
        'image_height': tf.io.FixedLenFeature([], tf.int64),
        'image_channels': tf.io.FixedLenFeature([], tf.int64),
        'class_id': tf.io.FixedLenFeature([], tf.int64),
        }

    parsed = tf.io.parse_single_example(serialized = record, features = features)
    image = tf.io.decode_raw(parsed['image_data'], tf.uint8)
    image = tf.cast(image, tf.float32)

    image_width = tf.cast(parsed['image_width'], tf.int64)
    image_height = tf.cast(parsed['image_height'], tf.int64)
    image_channels = tf.cast(parsed['image_channels'], tf.int64)
    class_id = tf.cast(parsed['class_id'], tf.int64)

    return tf.reshape(image, [image_width, image_height, image_channels]), class_id

###################################################################################################
def GetDatasetFromTFRecordsList(filenames, batch_size, repeat):
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=12)
    dataset = dataset.shuffle(256)

    dataset = dataset.repeat(repeat)

    dataset = dataset.map(ParseExample, 12)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=2)

    return dataset

###################################################################################################
def GetDatasetFromTFRecordsDirectory(data_path, batch_size, repeat = None):
    records_list = []
    records_list += (sorted(data_path.glob('*.tfrecords')))

    if len(records_list) == 0:
        print('Directory is empty.')
        return None

    records_series = pd.Series(records_list)
    records_series = records_series.apply(lambda x: str(x))

    return GetDatasetFromTFRecordsList(records_series.to_numpy(), batch_size, repeat)


###################################################################################################
def GenerateCSVFromImageFolder(images_path, csv_path):
    images_df = pd.DataFrame()
    images_df['image_path'] = pd.Series(images_path.glob("**/*.jpg")).apply(lambda x: str(x))
    images_df['class_name'] = images_df['image_path'].apply(lambda x: x.split('\\')[-2])
    images_df['class_id'] = images_df['class_name'].astype('category').cat.codes
    images_df = images_df.sample(frac = 1, random_state = 42).reset_index(drop = True)
    images_df.to_csv(csv_path, index=False)

    return images_df['class_id'].nunique()


###################################################################################################
def ShowOrSaveExampleAsPlot(example, image_shape, filename = ""):
    images_tensor = example[0]
    labels_tensor = example[1]

    plt.figure(figsize=(16, 9))

    images_count = len(images_tensor)
    size = np.ceil(np.sqrt(images_count))

    for index in range(images_count):
        ax = plt.subplot(size, size, index + 1)
        image = images_tensor[index].numpy().reshape(image_shape[0], image_shape[1], image_shape[2])
        plt.imshow((image).astype(np.uint8))
        plt.title(labels_tensor[index].numpy())
        plt.axis('off')

    if len(filename) > 0:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

###################################################################################################
def ShowOrSaveTFRecordAsPlot(source_path, image_shape, destination_path = ".", images_per_plot = 32, max_plots_number = 1):
    # repeat should be turned off when showing or saving plots
    train_df = GetDatasetFromTFRecordsDirectory(source_path, 32)
    filename = source_path.name
    batch_number = 0

    for example in train_df:
        ShowOrSaveExampleAsPlot(example, image_shape, '{}/{}{}.png'.format(destination_path, filename, batch_number))
        batch_number += 1

        if max_plots_number > 0 and batch_number >= max_plots_number:
            break
