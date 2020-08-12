import tensorflow as tf
from pathlib import Path

import TFRecordDatasetUtils

###################################################################################################
def build_model(input_shape, num_classes):
    base_model = tf.keras.applications\
        .ResNet101(\
            weights='imagenet', include_top=False, \
            input_shape=(input_shape[0], input_shape[1], input_shape[2]), classes = num_classes)

    input = tf.keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    output = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)(input)
    output = base_model(output)
    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dense(256, activation='relu')(output)
    output = tf.keras.layers.Dense(128, activation='relu')(output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(output)

    """output = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)(input)
    output = tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(output)
    output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(output)
    output = tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(output)
    output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(output)
    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dense(128, activation='relu')(output)
    output = tf.keras.layers.Dense(64, activation='relu')(output)
    output = tf.keras.layers.Dense(32, activation='relu')(output)
    output = tf.keras.layers.Dense(6, activation='softmax')(output)"""

    model = tf.keras.Model(inputs=input, outputs=output)
    opt = tf.keras.optimizers.Adam(lr = 0.001)

    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opt, metrics=['sparse_categorical_accuracy'])

    return model

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

    IMAGE_SHAPE = (150, 150, 3)

    num_train_classes = TFRecordDatasetUtils.GenerateCSVFromImageFolder(TRAIN_PATH, TRAIN_CSV_PATH)
    num_test_classes = TFRecordDatasetUtils.GenerateCSVFromImageFolder(TEST_PATH, TEST_CSV_PATH)
    TFRecordDatasetUtils.GenerateTFRecordsFromCSV(TRAIN_CSV_PATH, TRAIN_TFRECORDS_PATH, IMAGE_SHAPE)
    TFRecordDatasetUtils.GenerateTFRecordsFromCSV(TEST_CSV_PATH, TEST_TFRECORDS_PATH, IMAGE_SHAPE)

    #TFRecordDatasetUtils.ShowOrSaveTFRecordAsPlot(TRAIN_TFRECORDS_PATH, IMAGE_SHAPE)

    train_df = TFRecordDatasetUtils.GetDatasetFromTFRecordsDirectory(TRAIN_TFRECORDS_PATH, 32)
    validation_df = TFRecordDatasetUtils.GetDatasetFromTFRecordsDirectory(TEST_TFRECORDS_PATH, 32)

    model = build_model(IMAGE_SHAPE, max(num_train_classes, num_test_classes))
    model.fit( train_df,\
               epochs = 10,\
               verbose = 1,\
               steps_per_epoch = 10,\
               validation_data = validation_df,\
               validation_steps = 10\
               )
