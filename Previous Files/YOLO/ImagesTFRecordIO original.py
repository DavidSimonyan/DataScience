import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import sys
import matplotlib.pyplot as plt
from pathlib import Path

###################################################################################################
def GetBytesFeature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

###################################################################################################
def GetFloatFeature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

###################################################################################################
def GetIntFeature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

###################################################################################################
def LoadCV2Image(image_path):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(image_path)

    if img is None:
        return None

    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

###################################################################################################
def GenerateTFRecordsFromDirectory(source_data_path, desctination_data_path, images_per_record_file = 1000, max_number_of_record_files = -1):
    Path(source_data_path).mkdir(parents=True, exist_ok=True)
    Path(desctination_data_path).mkdir(parents=True, exist_ok=True)

    images_list = []

    for extension in ['jpg']:
        images_list += (sorted(source_data_path.glob('*.' + extension)))

    images_list_series = pd.Series(images_list)

    images = pd.DataFrame()
    images['image_path'] = images_list_series.apply(lambda x: str(x))
    images['label'] = images_list_series.apply(lambda x: str(x.stem))

    images_count = len(images['image_path'])
    tf_record_file_index = 0

    for image_index in range(images_count):
        if image_index % images_per_record_file == 0:
            writer = tf.io.TFRecordWriter('{}\{}_{}.tfrecords'.format(desctination_data_path, source_data_path.name, tf_record_file_index))
            tf_record_file_index += 1

        image = LoadCV2Image(images['image_path'][image_index])
        label = images['label'][image_index]

        if image is None:
            print('Empty image: {}'.format(image))
            continue

        feature = {'image_raw': GetBytesFeature(image.tobytes()),
                   'label': GetBytesFeature(bytes(str(label), 'ascii'))
                   }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

        if image_index % images_per_record_file == images_per_record_file - 1 or image_index == images_count - 1:
            print('Saved data: {}/{}'.format(image_index + 1, images_count))
            sys.stdout.flush()
            writer.close()

            if max_number_of_record_files > 0 and tf_record_file_index >= max_number_of_record_files:
                break

###################################################################################################
def ParseExample(record):
    keys_to_features =\
        {
        "image_raw": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string)
        }

    parsed = tf.io.parse_single_example(record, keys_to_features)
    image = tf.io.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed["label"], tf.string)

    return {'image': image}, label

###################################################################################################
def GetDatasetFromTFRecordsList(filenames, batch_size, repeat = True):
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=24)
    dataset = dataset.shuffle(1024)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.map(ParseExample, 24)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=2)

    return dataset

###################################################################################################
def GetDatasetFromTFRecordsDirectory(data_path, batch_size, repeat = True):
    records_list = []
    records_list += (sorted(data_path.glob('*.tfrecords')))

    if len(records_list) == 0:
        print('Directory is empty.')
        return None

    records_series = pd.Series(records_list)
    records_series = records_series.apply(lambda x: str(x))

    return GetDatasetFromTFRecordsList(records_series.to_numpy(), batch_size, repeat)

###################################################################################################
def ShowOrSaveExampleAsPlot(example, filename = ""):
    images_tensor = example[0]['image']
    labels_tensor = example[1]

    plt.figure(figsize=(16, 9))

    images_count = len(images_tensor)
    size = np.ceil(np.sqrt(images_count))

    for index in range(images_count):
        ax = plt.subplot(size, size, index + 1)
        image = images_tensor[index].numpy().reshape(224, 224, 3)
        plt.imshow((image).astype(np.uint8))
        plt.title(labels_tensor[index].numpy())
        plt.axis('off')

    if len(filename) > 0:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

###################################################################################################
def ShowOrSaveTFRecordAsPlot(source_path, destination_path = "", images_per_plot = 32, max_plots_number = -1):
    Path(source_path).mkdir(parents=True, exist_ok=True)
    if len(str(destination_path)) > 0:
        Path(destination_path).mkdir(parents=True, exist_ok=True)

    # repeat should be turned off when showing or saving plots
    train_df = GetDatasetFromTFRecordsDirectory(source_path, 256, False)
    filename = source_path.name
    batch_number = 0

    for example in train_df:
        ShowOrSaveExampleAsPlot(example, '{}/{}{}.png'.format(destination_path, filename, batch_number))
        batch_number += 1

        if max_plots_number > 0 and batch_number >= max_plots_number:
            break

###################################################################################################
def main():
    DATA_PATH = Path(r'C:/DS/data/Open Images Object Detection RVC 2020/')
    TRAIN_DATA_PATH = DATA_PATH / 'sample'

    GenerateTFRecordsFromDirectory(DATA_PATH / 'validation', TRAIN_DATA_PATH / 'tfrecords', 1000)
    ShowOrSaveTFRecordAsPlot(TRAIN_DATA_PATH / 'tfrecords', TRAIN_DATA_PATH / 'plots', 256)

if __name__ == "__main__":
    main()
