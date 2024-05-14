import os
import tensorflow as tf
import zipfile

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from Commons.commonUtils import download_dataset, plot_for_one_model

TRAIN_DATA_URL = "https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps.zip"
VALIDATION_DATA_URL = "https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps-test-set.zip"

base_path = "/Data/Rock_Paper_Scissor"
is_Training = False


def main():
    global is_Training
    final_model = get_model()
    final_model.summary()

    # # Preparing the data
    train_generator, validation_generator = prepare_dataset()

    if not (os.path.exists("rps_weights.keras")):
        is_Training = True

    if not is_Training:
        return

    print("\n************************************Training Started************************************\n")

    final_history = final_model.fit(train_generator, epochs=1, validation_data=validation_generator)

    # Get efficient model weights and save
    final_model.save('rps_weights.keras')

    # plot_for_one_model(final_history)
    plot_for_one_model(history=final_history, isValidation=True, is_save=True,
                       location="history.png")


def prepare_dataset():
    if not is_Training:
        return

    if not os.path.exists(base_path):
        os.mkdir(base_path)
    train_path = os.path.join(base_path, "train")
    if not os.path.exists(train_path):
        os.mkdir(train_path)

    validation_path = os.path.join(base_path, "validation")
    if not os.path.exists(validation_path):
        os.mkdir(validation_path)

    train_data_path = os.path.join(train_path, "rps")
    validation_data_path = os.path.join(validation_path, "rps")

    train_data_zip = os.path.join(train_path, "rps.zip")
    validation_data_zip = os.path.join(validation_path, "rps-test-set.zip")

    try:
        if not os.path.exists(train_data_zip):
            download_dataset(TRAIN_DATA_URL, train_data_zip)
        if not os.path.exists(validation_data_zip):
            download_dataset(TRAIN_DATA_URL, validation_data_zip)
    except OSError:
        print("Error")
        return
    if not os.path.exists(train_data_path):
        unzip_data(train_data_zip, train_path)
    if not os.path.exists(validation_data_path):
        unzip_data(validation_data_zip, validation_path)

    return get_data_image_generator(train_data_path, validation_data_path)


def unzip_data(data_zip, data_path):
    zip_ref = zipfile.ZipFile(data_zip, 'r')
    zip_ref.extractall(data_path)

    zip_ref.close()


def get_model():
    final_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", strides=2, activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    #
    final_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                        loss="sparse_categorical_crossentropy", metrics=["acc"])

    return final_model


def get_data_image_generator(train_path, validation_path):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # train_datagen = ImageDataGenerator(
    #     rescale=1./255
    # )
    validation_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(150, 150),
        batch_size=128,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_path,
        target_size=(150, 150),
        batch_size=128,
        class_mode='binary'
    )

    return train_generator, validation_generator


if __name__ == '__main__':
    main()
