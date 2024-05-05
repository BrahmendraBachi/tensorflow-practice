import os
import tensorflow as tf
import zipfile
from Commons.commonUtils import plot_for_one_model, MyCallback
from Commons.commonUtils import download_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Flatten, Dense

# print(tf.__version__)
isTraining = False

DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
WEIGHTS_URL = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

inception_v3_weights_path = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.keras"

def load_pretrained_model():
    pretrained_model = InceptionV3(
        input_shape=(150, 150, 3),
        include_top=False,
        weights=None
    )
    pretrained_model.load_weights(inception_v3_weights_path)
    layers = pretrained_model.layers
    for layer in layers:
        layer.trainable = False

    last_layer = pretrained_model.get_layer('mixed10')
    last_output = last_layer.output
    x = Flatten()(last_output)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(pretrained_model.input, x)

if not os.path.exists(inception_v3_weights_path):
    download_dataset(WEIGHTS_URL, inception_v3_weights_path)

final_model = load_pretrained_model()
final_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss="binary_crossentropy", metrics=["acc"])

final_model.summary()

eff_model = tf.keras.models.clone_model(final_model)
eff_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['acc'])

callbacks = MyCallback()

try:
    if isTraining:
        raise FileNotFoundError()
    final_model.load_weights('cats_vs_dogs.keras')
    final_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])
    print("weights exists")

    eff_model.load_weights('eff_cats_vs_dogs.keras')
    eff_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])
    print("efficient weights exists")
except:
    base_path = "C:/Users/haor1122/PycharmProjects/tensorflow-practice/Data/Cats_Vs_Dogs"
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
        epochs=20,
        validation_data=validation_data_generator,
        callbacks=[callbacks]
    )

    # Get efficient model weights and save
    final_model.save('cats_vs_dogs.keras')

    efficient_model_weights = callbacks.get_efficient_model_weights()
    eff_model.set_weights(efficient_model_weights)
    eff_model.save('eff_cats_vs_dogs.keras')

    plot_for_one_model(history=final_history, isValidation=True, is_save=True,
                       location="history.png")