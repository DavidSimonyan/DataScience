import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import datetime

MAX_NUMBER_OF_BOUNDING_BOXES_PER_IMAGE = 100

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
def ExtendOrTruncateList(source_list, target_length = MAX_NUMBER_OF_BOUNDING_BOXES_PER_IMAGE):
    return source_list[:target_length] + [0] * (target_length - len(source_list))

###################################################################################################
def LoadCV2Image(image_path):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, channels = image.shape

    assert channels == 3

    return image, height, width

###################################################################################################
def GenerateTFRecordsFromDirectory(source_data_path, desctination_data_path, images_per_record_file = 1000, max_number_of_record_files = -1, max_number_of_bounding_rectangles_per_image = MAX_NUMBER_OF_BOUNDING_BOXES_PER_IMAGE):
    Path(source_data_path).mkdir(parents=True, exist_ok=True)
    Path(desctination_data_path).mkdir(parents=True, exist_ok=True)

    images_list = []

    for extension in ['jpg']:
        images_list += (sorted(source_data_path.glob('*.' + extension)))

    images_list_series = pd.Series(images_list)

    images = pd.DataFrame()
    images['image_path'] = images_list_series.apply(lambda x: str(x))
    images['image_id'] = images_list_series.apply(lambda x: str(x.stem))

    images_count = len(images['image_path'])
    tf_record_file_index = 0


    # ImageID	XMin	XMax	YMin	YMax	class_id	class_name
    DATA_PATH = Path(r'C:/DS/data/Open Images Object Detection RVC 2020/')
    train_df = pd.read_csv(DATA_PATH / 'train.csv')

    for image_index in range(images_count):
        current_time = datetime.datetime.now()

        if image_index % images_per_record_file == 0:
            writer = tf.io.TFRecordWriter('{}\{}_{}.tfrecords'.format(desctination_data_path, source_data_path.name, tf_record_file_index))
            tf_record_file_index += 1

        image_data, image_height, image_width = LoadCV2Image(images['image_path'][image_index])
        image_id = images['image_id'][image_index]

        if image_data is None:
            print('Empty image: {}'.format(image_id))
            continue

        image_data_df = train_df.loc[train_df['ImageID'] == image_id]

        original_number_of_bounding_rects = len(image_data_df)
        if original_number_of_bounding_rects > max_number_of_bounding_rectangles_per_image:
            image_data_df = image_data_df[:max_number_of_bounding_rectangles_per_image]

        xmins = list(image_data_df['XMin'])
        xmaxs = list(image_data_df['XMax'])
        ymins = list(image_data_df['YMin'])
        ymaxs = list(image_data_df['YMax'])
        class_ids = list(image_data_df['class_id'])

        if original_number_of_bounding_rects < max_number_of_bounding_rectangles_per_image:
            xmins = xmins + [0] * (max_number_of_bounding_rectangles_per_image - original_number_of_bounding_rects)
            xmaxs = xmaxs + [0] * (max_number_of_bounding_rectangles_per_image - original_number_of_bounding_rects)
            ymins = ymins + [0] * (max_number_of_bounding_rectangles_per_image - original_number_of_bounding_rects)
            ymaxs = ymaxs + [0] * (max_number_of_bounding_rectangles_per_image - original_number_of_bounding_rects)
            class_ids = class_ids + [0] * (max_number_of_bounding_rectangles_per_image - original_number_of_bounding_rects)

        features =\
                {
                'image_data': GetBytesFeature(image_data.tobytes()),
                'image_height': GetIntFeature(image_height),
                'image_width': GetIntFeature(image_width),
                'image_id': GetBytesFeature(bytes(str(image_id), 'ascii')),
                'bounding_boxes/count': GetIntFeature(original_number_of_bounding_rects if original_number_of_bounding_rects < max_number_of_bounding_rectangles_per_image else max_number_of_bounding_rectangles_per_image),
                'bounding_boxes/xmins': GetFloatFeature(xmins),
                'bounding_boxes/xmaxs': GetFloatFeature(xmaxs),
                'bounding_boxes/ymins': GetFloatFeature(ymins),
                'bounding_boxes/ymaxs': GetFloatFeature(ymaxs),
                'bounding_boxes/class_ids': GetIntFeature(class_ids),
                }

        example = tf.train.Example(features=tf.train.Features(feature=features))

        writer.write(example.SerializeToString())

        print("step number {} at {}".format(image_index, datetime.datetime.now() - current_time))

        if image_index % images_per_record_file == images_per_record_file - 1 or image_index == images_count - 1:
            print('Saved data: {}/{}'.format(image_index + 1, images_count))
            sys.stdout.flush()
            writer.close()

            if max_number_of_record_files > 0 and tf_record_file_index >= max_number_of_record_files:
                break

###################################################################################################
def ParseExample(record):
    features =\
        {
        'image_data': tf.io.FixedLenFeature([], tf.string),
        'image_height': tf.io.FixedLenFeature([], tf.int64),
        'image_width': tf.io.FixedLenFeature([], tf.int64),
        'image_id': tf.io.FixedLenFeature([], tf.string),
        'bounding_boxes/count': tf.io.FixedLenFeature([], tf.int64),
        'bounding_boxes/xmins': tf.io.VarLenFeature(tf.float32),
        'bounding_boxes/xmaxs': tf.io.VarLenFeature(tf.float32),
        'bounding_boxes/ymins': tf.io.VarLenFeature(tf.float32),
        'bounding_boxes/ymaxs': tf.io.VarLenFeature(tf.float32),
        'bounding_boxes/class_ids': tf.io.VarLenFeature(tf.int64),
        }

    parsed = tf.io.parse_single_example(serialized = record, features = features)

    #image = tf.reshape(parsed['image_data'], shape=[])

    image = tf.io.decode_raw(parsed['image_data'], tf.uint8)
    #image = tf.cast(image, tf.float32)

    image_height = tf.cast(parsed['image_height'], tf.int64)
    image_width = tf.cast(parsed['image_width'], tf.int64)
    image_id = tf.cast(parsed["image_id"], tf.string)
    count = tf.cast(parsed['bounding_boxes/count'], tf.int64)

    xmins = parsed['bounding_boxes/xmins'].values
    ymins = parsed['bounding_boxes/ymins'].values
    xmaxs = parsed['bounding_boxes/xmaxs'].values
    ymaxs = parsed['bounding_boxes/ymaxs'].values
    class_ids = parsed['bounding_boxes/class_ids'].values

    return {'image': image,
            'image_height': image_height,
            'image_width': image_width,
            'image_id' : image_id,
            'count': count,
            'xmins': xmins,
            'xmaxs': xmaxs,
            'ymins': ymins,
            'ymaxs': ymaxs,
            'class_ids': class_ids
            }

###################################################################################################
def GetDatasetFromTFRecordsList(filenames, batch_size, repeat = True):
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=12)
    dataset = dataset.shuffle(1024)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.map(ParseExample, 12)
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
    images = example['image'].numpy()
    images_height = example['image_height'].numpy()
    images_width = example['image_width'].numpy()
    images_id = example['image_id'].numpy()
    bounding_boxes_count = example['count'].numpy()
    bounding_boxes_xmins = example['xmins'].numpy()
    bounding_boxes_xmaxs = example['xmaxs'].numpy()
    bounding_boxes_ymins = example['ymins'].numpy()
    bounding_boxes_ymaxs = example['ymaxs'].numpy()
    bounding_boxes_class_ids = example['class_ids'].numpy()

    plt.figure(figsize=(16, 9))

    images_count = len(images)
    size = np.ceil(np.sqrt(images_count))

    for image_index in range(images_count):
        image_numpy = images[image_index].reshape(224, 224, 3)

        image_width = images_width[image_index]
        image_height = images_height[image_index]
        count = bounding_boxes_count[image_index]
        xmins = bounding_boxes_xmins[image_index][:count] * image_width
        xmaxs = bounding_boxes_xmaxs[image_index][:count] * image_width
        ymins = bounding_boxes_ymins[image_index][:count] * image_height
        ymaxs = bounding_boxes_ymaxs[image_index][:count] * image_height
        class_ids = bounding_boxes_class_ids[image_index][:count]

        for rectangle_index in range(count):
            image = cv2.rectangle(image_numpy,
                                  (xmins[rectangle_index], ymins[rectangle_index]),
                                  (xmaxs[rectangle_index], ymaxs[rectangle_index]),
                                  (0, 255, 0), 1)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite('{}.jpg'.format(images_id[image_index].decode('utf-8')), image, [cv2.IMWRITE_JPEG_QUALITY, 100])

        fig, ax = plt.subplots(1)
        ax.imshow(image_numpy)
        plt.text(xmins[rectangle_index] / image_width, ymins[rectangle_index] / image_height, 'class_id: {}'.format(class_ids[image_index]))

        bounding_box = patches.Rectangle((xmins[rectangle_index], ymaxs[rectangle_index]),
                                         (xmaxs[rectangle_index] - xmins[rectangle_index]),
                                         (ymaxs[rectangle_index] - ymins[rectangle_index]))

        ax.add_patch(bounding_box)

        plt.title(images_id[image_index].decode('utf-8'))
        plt.axis('off')

        if len(filename) > 0:
            plt.savefig('{} {}.png'.format(filename, image_index))
            plt.close()
        else:
            plt.show()

###################################################################################################
def ShowOrSaveTFRecordAsPlot(source_path, destination_path = "", images_per_plot = 32, max_plots_number = -1):
    Path(source_path).mkdir(parents=True, exist_ok=True)
    if len(str(destination_path)) > 0:
        Path(destination_path).mkdir(parents=True, exist_ok=True)

    # repeat should be turned off when showing or saving plots
    train_df = GetDatasetFromTFRecordsDirectory(source_path, images_per_plot, False)
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
    TRAIN_DATA_PATH = DATA_PATH / 'train_01 sample'

    GenerateTFRecordsFromDirectory(TRAIN_DATA_PATH, TRAIN_DATA_PATH / 'tfrecords', 4, 1)
    ShowOrSaveTFRecordAsPlot(TRAIN_DATA_PATH / 'tfrecords', TRAIN_DATA_PATH / 'plots', 4, 1)

if __name__ == "__main__":
    main()
