import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from Commons.commonUtils import MyCallback

# print(tf.__version__)

isTraining = False

final_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, kernel_size=5, strides=2, padding="same", activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="same", activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation='softmax')
])

final_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

final_model.summary()

eff_model = tf.keras.models.clone_model(final_model)
eff_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

callbacks = MyCallback()

try:
    if isTraining:
        raise FileNotFoundError()
    final_model.load_weights('mnist_model.keras')
    final_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    print("weights exists")

    eff_model.load_weights('eff_mnist_model.keras')
    eff_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    print("efficient weights exists")

except FileNotFoundError:
    df = pd.read_csv('../../Data/Mnist/train/train.csv')

    label_column = 'label'
    if label_column in df.columns:
        labels = df.pop(label_column)

    train_images, val_images, train_labels, val_labels = train_test_split(df, labels, test_size=0.2, random_state=42)

    train_images = train_images.to_numpy().reshape((train_images.shape[0], 28, 28, 1))
    train_labels = train_labels.to_numpy().reshape((train_labels.shape[0], 1))

    val_images = val_images.to_numpy().reshape((val_images.shape[0], 28, 28, 1))
    val_labels = val_labels.to_numpy().reshape((val_labels.shape[0], 1))

    # one hot encoding
    y_train = to_categorical(train_labels, num_classes=10)
    y_valid = to_categorical(val_labels, num_classes=10)

    dataGen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    dataGenerator = dataGen.flow(train_images, y_train, batch_size=64)
    # print()
    print("\n************************************Training Started************************************\n")
    # print()

    final_history = final_model.fit(
        dataGenerator,
        epochs=20,
        steps_per_epoch=train_images.shape[0] // 64,
        validation_data=(val_images, y_valid),
        callbacks=[callbacks]
    )

    # plot_for_one_model(final_history)
    final_model.save('mnist_model.keras')

    # Get efficient model weights and save
    efficient_model_weights = callbacks.get_efficient_model_weights()
    eff_model.set_weights(efficient_model_weights)
    eff_model.save('eff_mnist_model.keras')

# Evaluation

test_df = pd.read_csv('../../Data/Mnist/test/test.csv')
results_df = pd.read_csv('../../Data/Mnist/MNIST-CNN.csv')

x_test = test_df.to_numpy()
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
if "ImageId" in results_df.columns:
    results_df.pop("ImageId")
y_test = results_df.to_numpy()
y_test = to_categorical(y_test, num_classes=10)

print("\n# Evaluation with trained model")
final_model.evaluate(x_test, y_test)

print("\n# Evaluation with maximum efficient model")
eff_model.evaluate(x_test, y_test)