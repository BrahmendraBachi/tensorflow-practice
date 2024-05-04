import os
import tensorflow as tf
import zipfile
from IDLE.commons.commonUtils import plot_for_one_model, MyCallback
from IDLE.commons.commonUtils import download_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# print(tf.__version__)
isTraining = False

DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"

callbacks = MyCallback()

final_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(16, (3, 3), padding="same", strides=2, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", strides=2, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", strides=2, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), padding="same", strides=2, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), padding="same", strides=2, activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

final_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])
final_model.summary()

eff_model = tf.keras.models.clone_model(final_model)
eff_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['acc'])

try:
    if isTraining:
        raise FileNotFoundError()
    final_model.load_weights('cats_vs_dogs.keras')
    final_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])

    # eff_model.load_weights('eff_cats_vs_dogs.keras')
    # eff_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])
    # print("efficient weights exists")
except FileNotFoundError:

    base_path = "C:/Users/Brahmendra Bachi/PycharmProjects/tensorflow-practice/Data/Cats_Vs_Dogs"
    data_zip_dir = os.path.join(base_path, "cats_and_dogs.zip")
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    if not os.path.exists(data_zip_dir):
        download_dataset(DATA_URL, data_zip_dir)
    data_dir = os.path.join(base_path, "cats_and_dogs_filtered")
    train_dir = os.path.join(data_dir, "train")
    validation_dir = os.path.join(data_dir, "validation")
    if not os.path.exists(data_dir):
        local_zip = data_zip_dir
        zip_ref = zipfile.ZipFile(local_zip, 'r')
        zip_ref.extractall(base_path)
        zip_ref.close()

    train_data_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    train_data_generator = train_data_gen.flow_from_directory(
        train_dir,
        class_mode='binary',
        batch_size=128,
        target_size=(150, 150)
    )

    validation_data_gen = ImageDataGenerator(rescale=1. / 255)
    validation_data_generator = validation_data_gen.flow_from_directory(
        validation_dir,
        class_mode='binary',
        batch_size=32,
        target_size=(150, 150)
    )
    #
    # print()
    print("\n************************************Training Started************************************\n")
    # print()

    final_history = final_model.fit(
        train_data_generator,
        epochs=1,
        validation_data=validation_data_generator,
        callbacks=[callbacks]
    )

    # Get efficient model weights and save
    final_model.save('horses_vs_humans.keras')

    # plot_for_one_model(final_history)
    plot_for_one_model(history=final_history, isValidation=True, is_save=True,
                       location="history.png")
