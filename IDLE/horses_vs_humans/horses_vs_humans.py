import os
import tensorflow as tf
import zipfile
from IDLE.commons.commonUtils import plot_for_one_model
from IDLE.commons.commonUtils import download_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# print(tf.__version__)
isTraining = True

TRAIN_DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
VALIDATION_DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"

final_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

final_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

final_model.summary()

try:
    if isTraining:
        raise FileNotFoundError()
    final_model.load_weights('horses_vs_humans.keras')
except:
    base_path = "C:/Users/Brahmendra Bachi/PycharmProjects/tensorflow-practice/Data/Horses_Vs_Humans"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    train_path = os.path.join(base_path, "train")
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    train_data_zip = os.path.join(train_path, "horse_or_human.zip")

    validation_path = os.path.join(base_path, "validation")
    if not os.path.exists(validation_path):
        os.mkdir(validation_path)
    validation_data_zip = os.path.join(validation_path, "validation-horse-or-human.zip")

    train_data_path = os.path.join(train_path, "horse-or-human")
    validation_data_path = os.path.join(validation_path, "validation-horse-or-human")


    try:
        if not os.path.exists(train_data_path):
            if not os.path.exists(train_data_zip):
                print("Downloading training dataset......", end=" ")
                download_dataset(TRAIN_DATA_URL, train_data_zip)
                print("Completed")
            local_zip = train_data_zip
            zip_ref = zipfile.ZipFile(local_zip, 'r')
            zip_ref.extractall(train_data_path)
            zip_ref.close()

        if not os.path.exists(validation_data_path):
            if not os.path.exists(validation_data_zip):
                print("Downloading Validation dataset......", end=" ")
                download_dataset(VALIDATION_DATA_URL, validation_data_zip)
                print("Completed")
            local_zip = validation_data_zip
            zip_ref = zipfile.ZipFile(local_zip, 'r')
            zip_ref.extractall(validation_data_path)
            zip_ref.close()


        # Extract dataset

        train_data_gen = ImageDataGenerator(rescale=1./255)
        train_data_generator = train_data_gen.flow_from_directory(
            train_data_path,
            class_mode='binary',
            batch_size=32,
            target_size=(150, 150)
        )

        validation_data_gen = ImageDataGenerator(rescale=1./255)
        validation_data_generator = validation_data_gen.flow_from_directory(
            validation_data_path,
            class_mode='binary',
            batch_size=32,
            target_size=(150, 150)
        )

        # print()
        print("\n************************************Training Started************************************\n")
        # print()

        final_history = final_model.fit(
            train_data_generator,
            epochs=20,
            validation_data=validation_data_generator,
        )

        # Get efficient model weights and save
        final_model.save('horses_vs_humans.keras')

        # plot_for_one_model(final_history)
        plot_for_one_model(history=final_history, isValidation=True, is_save=True,
                           location="history.png")
    except Exception as e:
        print("An error occurred while downloading the dataset:", e)
