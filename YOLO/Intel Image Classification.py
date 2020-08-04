import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import datetime
import sys

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
CHANNELS = 3
CLASSES = 6

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
    # read an image and resize to (IMAGE_WIDTH, IMAGE_HEIGHT)
    # cv2 load images as BGR, convert it to RGB
    image = cv2.imread(str(image_path))

    width, height, channels = image.shape

    if image is None:
        return None

    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image, width, height, channels

###################################################################################################
def GenerateTFRecordsFromCSV(images_data_file_path, desctination_data_path, images_per_record_file = 200, max_number_of_record_files = -1):
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
        image_data, image_width, image_height, image_channels = LoadCV2Image(image_path)
        class_id = row_data['class_id']

        if image_data is None:
            print('Empty image: {}'.format(image_path))
            continue

        features =\
                {
                'image_data': GetBytesFeature(image_data.tobytes()),
                'class_id': GetIntFeature(class_id),
                }

        example = tf.train.Example(features=tf.train.Features(feature=features))

        writer.write(example.SerializeToString())

        if row_index % images_per_record_file == images_per_record_file - 1 or row_index == images_count - 1:
            print("step number {} at {}".format(row_index, datetime.datetime.now() - current_time))
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
        'class_id': tf.io.FixedLenFeature([], tf.int64),
        }

    parsed = tf.io.parse_single_example(serialized = record, features = features)
    image = tf.io.decode_raw(parsed['image_data'], tf.uint8)
    image = tf.cast(image, tf.float32)

    class_id = tf.cast(parsed['class_id'], tf.int64)

    return tf.reshape(image, [IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS]), class_id

###################################################################################################
def GetDatasetFromTFRecordsList(filenames, batch_size, repeat):
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=12)
    dataset = dataset.shuffle(1024)

    dataset = dataset.repeat(repeat)

    dataset = dataset.map(ParseExample, 12)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=2)

    return dataset

###################################################################################################
def GetDatasetFromTFRecordsDirectory(data_path, batch_size, repeat = 1):
    records_list = []
    records_list += (sorted(data_path.glob('*.tfrecords')))

    if len(records_list) == 0:
        print('Directory is empty.')
        return None

    records_series = pd.Series(records_list)
    records_series = records_series.apply(lambda x: str(x))

    return GetDatasetFromTFRecordsList(records_series.to_numpy(), batch_size, repeat)

###################################################################################################
def GetImageDataFromExample(example):
    image_numpy = example[0].numpy()
    class_id = example[1].numpy()

    return image_numpy, class_id

###################################################################################################
def build_model():
    base_model = tf.keras.applications\
        .VGG19(\
        #.ResNet101(\
            weights='imagenet', include_top=False, \
            input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS), classes = CLASSES)

    input = tf.keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))
    output = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(input)
    output = base_model(output)
    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dense(32, activation='relu')(output)
    output = tf.keras.layers.Dense(16, activation='relu')(output)
    #output = tf.keras.layers.Dense(1, activation = 'sigmoid')(output)
    output = tf.keras.layers.Dense(CLASSES, activation='softmax')(output)

    #output = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)(input)
    """output = tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(output)
    output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(output)
    output = tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(output)
    output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(output)"""
    """output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dense(128, activation='relu')(output)
    output = tf.keras.layers.Dense(64, activation='relu')(output)
    output = tf.keras.layers.Dense(32, activation='relu')(output)
    output = tf.keras.layers.Dense(6, activation='softmax')(output)"""

    model = tf.keras.Model(inputs=input, outputs=output)
    opt = tf.keras.optimizers.Adam(lr = 0.001)

    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opt, metrics=['accuracy'])

    return model

def GenerateCSVFromImageFolder(images_path, csv_path):
    images_df = pd.DataFrame()
    images_df['image_path'] = pd.Series(images_path.glob("**/*.jpg")).apply(lambda x: str(x))
    images_df['class_name'] = images_df['image_path'].apply(lambda x: x.split('\\')[-2])
    images_df['class_id'] = images_df['class_name'].astype('category').cat.codes + 1
    images_df = images_df.sample(frac = 1, random_state = 42).reset_index(drop = True)
    images_df.to_csv(csv_path, index=False)

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    DATA_PATH = Path("C:\DS\data\Intel Image Classification")

    TRAIN_PATH = DATA_PATH / 'seg_train/seg_train'
    TEST_PATH = DATA_PATH / 'seg_test/seg_test'
    TRAIN_CSV_PATH = DATA_PATH / 'train_image_path_to_class.csv'
    TEST_CSV_PATH = DATA_PATH / 'test_image_path_to_class.csv'

    TRAIN_TFRECORDS_PATH = DATA_PATH / 'TFRecords/train'
    TEST_TFRECORDS_PATH = DATA_PATH / 'TFRecords/test'

    GenerateCSVFromImageFolder(TRAIN_PATH, TRAIN_CSV_PATH)
    GenerateCSVFromImageFolder(TEST_PATH, TEST_CSV_PATH)
    GenerateTFRecordsFromCSV(TRAIN_CSV_PATH, TRAIN_TFRECORDS_PATH)
    GenerateTFRecordsFromCSV(TEST_CSV_PATH, TEST_TFRECORDS_PATH)

    train_df = GetDatasetFromTFRecordsDirectory(TRAIN_TFRECORDS_PATH, 32)
    validation_df = GetDatasetFromTFRecordsDirectory(TEST_TFRECORDS_PATH, 32)

    model = build_model()
    model.fit( train_df,\
               epochs = 10,\
               verbose = 1,\
               #steps_per_epoch = 10,\
               validation_data = validation_df,\
               #validation_steps = 10\
               )

    model.evaluate(validation_df)
