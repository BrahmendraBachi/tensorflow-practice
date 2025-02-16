import json

import tensorflow as tf
import os
import numpy as np

from Commons.commonUtils import download_dataset, plot_for_one_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Commons.commonUtils import MyCallback

tf.random.set_seed(42)

DATA_URL = "https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json"

base_url = "C:/Users/Brahmendra Bachi/PycharmProjects/tensorflow-practice/Data/Sarcasm"
is_Training = False

# constants
TRUNCATE_TYPE = "post"
PADDING_TYPE = "post"
MAX_LEN = 120
OOV_TOKEN = "<OOV>"
VOCAB_SIZE = 10000
EMBEDDING_DIM = 16
DROPOUT = 0.1
DENSE_LAYERS = 64


def main():
    global is_Training

    callbacks = MyCallback()
    all_data = prepare_dataset()

    x_train, y_train = all_data["train"]["train_X"], all_data["train"]["train_Y"]
    x_valid, y_valid = all_data["valid"]["valid_X"], all_data["valid"]["valid_Y"]
    x_test, y_test = all_data["test"]["test_X"], all_data["test"]["test_Y"]

    final_model, eff_model = get_model()

    if not os.path.exists('sarcasm_weights.keras') or not os.path.exists("eff_sarcasm_weights.keras"):
        is_Training = True

    if not is_Training:
        final_model.load_weights("sarcasm_weights.keras")
        eff_model.load_weights("eff_sarcasm_weights.keras")
        print("Evaluating Model: ")
        result = evaluate_model(final_model, x_test, y_test)
        print("Evaluating Efficient Model: ")
        eff_result = evaluate_model(eff_model, x_test, y_test)
        return

    final_history = final_model.fit(
        x_train,
        y_train,
        epochs=10,
        validation_data=(x_valid, y_valid),
        callbacks=[callbacks]
    )

    eff_model.set_weights(callbacks.get_efficient_model_weights())

    final_model.save('sarcasm_weights.keras')
    eff_model.save('eff_sarcasm_weights.keras')

    print("Evaluating Model: ")
    result = evaluate_model(final_model, x_test, y_test)
    print("Evaluating Efficient Model: ")
    eff_result = evaluate_model(eff_model, x_test, y_test)

    plot_for_one_model(final_history, isValidation=True, is_save=True, location="history.png")


def prepare_dataset():
    sarcasm_data_file = os.path.join(base_url, 'sarcasm.json')
    if not os.path.exists(base_url):
        os.mkdir(base_url)
    if not os.path.exists(sarcasm_data_file):
        download_dataset(DATA_URL, sarcasm_data_file)

    with open(sarcasm_data_file, 'r') as f:
        datasource = json.load(f)

    data, labels = [], []
    for item in datasource:
        data.append(item['headline'])
        labels.append(item["is_sarcastic"])

    train_split = 18000
    test_split = 24000

    train_data = data[:train_split]
    train_labels = labels[:train_split]

    valid_data = data[train_split:test_split]
    valid_labels = labels[train_split:test_split]

    test_data = data[test_split:]
    test_labels = labels[test_split:]

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(train_data)
    sequences = tokenizer.texts_to_sequences(train_data)
    padded = pad_sequences(sequences, truncating=TRUNCATE_TYPE, padding=PADDING_TYPE, maxlen=MAX_LEN)

    padded = np.array(padded)
    train_labels = np.array(train_labels)

    valid_sequences = tokenizer.texts_to_sequences(valid_data)
    padded_valid = pad_sequences(valid_sequences, truncating="post", padding="post", maxlen=120)

    test_sequences = tokenizer.texts_to_sequences(test_data)
    padded_test = pad_sequences(test_sequences, truncating="post", padding="post", maxlen=120)

    padded = np.array(padded)
    padded_valid = np.array(padded_valid)
    padded_test = np.array(padded_test)

    train_labels = np.array(train_labels)
    valid_labels = np.array(valid_labels)
    test_labels = np.array(test_labels)

    return {
        "train": {"train_X": padded, "train_Y": train_labels},
        "valid": {"valid_X": padded_valid, "valid_Y": valid_labels},
        "test": {"test_X": padded_test, "test_Y": test_labels}
    }


def get_model():
    final_model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='relu')
    ])

    final_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

    final_model.summary()

    eff_model = tf.keras.models.clone_model(final_model)
    eff_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

    return final_model, eff_model


def evaluate_model(model, x_test, y_test):
    return model.evaluate(x_test, y_test)


if __name__ == "__main__":
    main()
