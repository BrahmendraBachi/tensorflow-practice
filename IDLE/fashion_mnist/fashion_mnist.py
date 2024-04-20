import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from IDLE.commons.commonUtils import MyCallback
from tensorflow.keras.datasets import fashion_mnist

print(tf.__version__)

isTraining = False

(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

Y_train = to_categorical(Y_train, num_classes=10)
y_test = to_categorical(Y_test, num_classes=10)

split_size = 50000

X_train = (X_train / 255.0).reshape((X_train.shape[0], 28, 28, 1))
x_test = (X_test / 255.0).reshape((X_test.shape[0], 28, 28, 1))

x_train = X_train[:split_size]
y_train = Y_train[:split_size]

x_valid = X_train[split_size:]
y_valid = Y_train[split_size:]

assert len(x_train) == len(y_train)
assert len(x_valid) == len(y_valid)
assert len(x_train) + len(x_valid) == len(X_train)
assert len(y_train) + len(y_valid) == len(Y_train)

assert x_train.shape == (split_size, 28, 28, 1)
assert y_train.shape == (split_size, 10)

final_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', strides=2, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', strides=2, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

final_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["acc"])

final_model.summary()

final_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
final_model.summary()

eff_model = tf.keras.models.clone_model(final_model)
eff_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

callbacks = MyCallback()

try:
    if isTraining:
        raise FileNotFoundError()
    final_model.load_weights('fashion_mnist_model.keras')
    final_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    print("weights exists")

    eff_model.load_weights('eff_fashion_mnist_model.keras')
    eff_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    print("efficient weights exists")

except FileNotFoundError:

    # print()
    print("\n************************************Training Started************************************\n")
    # print()

    final_history = final_model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=128,
        validation_data=(x_valid, y_valid),
        callbacks=[callbacks]
    )

    # plot_for_one_model(final_history)
    final_model.save('fashion_mnist_model.keras')

    # Get efficient model weights and save
    efficient_model_weights = callbacks.get_efficient_model_weights()
    eff_model.set_weights(efficient_model_weights)
    eff_model.save('eff_fashion_mnist_model.keras')

# Evaluation

print("\n# Evaluation with trained model")
final_model.evaluate(x_test, y_test)

print("\n# Evaluation with maximum efficient model")
eff_model.evaluate(x_test, y_test)
