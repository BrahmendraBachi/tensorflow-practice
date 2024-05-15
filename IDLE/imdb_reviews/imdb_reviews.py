import os
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from Commons.commonUtils import MyCallback, plot_for_one_model

is_Training = False

# Download the plain text dataset

VOCAB_SIZE = 20000
TRUNC_TYPE = "post"
PADDING = "post"
MAX_LEN = 120
OOV_TOKEN = "<OOV>"

# After Experiment
EMBEDDING_DIM = 64

def main():
    global is_Training
    callbacks = MyCallback()
    if not (os.path.exists("imdb_reviews_model.h5") or os.path.exists("eff_imdb_reviews_model.h5")):
        is_Training = True

    all_data = prepare_dataset()

    x_train, y_train = all_data["train"]["train_X"], all_data["train"]["train_Y"]
    x_valid, y_valid = all_data["valid"]["valid_X"], all_data["valid"]["valid_Y"]
    x_test, y_test = all_data["test"]["test_X"], all_data["test"]["test_Y"]

    final_model, eff_model = build_model()

    if not is_Training:
        final_model.load_weights('imdb_reviews_model.h5')
        eff_model.load_weights('eff_imdb_reviews_model.h5')

        print("Evaluating Model: ")
        result1 = final_model.evaluate(x_test, y_test)
        print("Evaluating Efficient Model: ")
        result2 = final_model.evaluate(x_valid, y_valid)
        return

    final_history = final_model.fit(
        x_train,
        y_train,
        epochs=10,
        validation_data=(x_valid, y_valid),
        callbacks=[callbacks]
    )

    plot_for_one_model(final_history, isValidation=True, is_save=True, location='history.png')

    final_model.save('imdb_reviews_model.keras')
    eff_model.set_weights(callbacks.get_efficient_model_weights())
    eff_model.save('eff_imdb_reviews_model.keras')

    print("Evaluating Model: ")
    result1 = final_model.evaluate(x_test, y_test)
    print("Evaluating Efficient Model: ")
    result2 = final_model.evaluate(x_valid, y_valid)






def build_model():
    final_model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=["acc"])
    final_model.summary()
    eff_model = tf.keras.models.clone_model(final_model)
    eff_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=["acc"])

    return final_model, eff_model

def prepare_dataset():
    print("Downloading Dataset......")
    imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
    print("Completed......")

    train, test = imdb['train'], imdb['test']
    train_sentences, testing_sentences, train_labels, testing_labels = [], [], [], []
    for s, l in train:
        train_sentences.append(s.numpy().decode('utf8'))
        train_labels.append(l.numpy())
    for s, l in test:
        testing_sentences.append(s.numpy().decode('utf8'))
        testing_labels.append(l.numpy())
    val_split = 15000
    test_sentences = testing_sentences[:val_split]
    test_labels = testing_labels[:val_split]
    valid_sentences = testing_sentences[val_split:]
    valid_labels = testing_labels[val_split:]

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(train_sentences)
    sequences_train = tokenizer.texts_to_sequences(train_sentences)
    sequences_valid = tokenizer.texts_to_sequences(valid_sentences)
    sequences_test = tokenizer.texts_to_sequences(test_sentences)
    padded_train = pad_sequences(sequences_train, truncating=TRUNC_TYPE, padding=PADDING, maxlen=MAX_LEN)
    padded_train = np.array(padded_train)
    padded_valid = pad_sequences(sequences_valid, truncating=TRUNC_TYPE, padding=PADDING, maxlen=MAX_LEN)
    padded_valid = np.array(padded_valid)
    padded_test = pad_sequences(sequences_test, truncating=TRUNC_TYPE, padding=PADDING, maxlen=MAX_LEN)
    padded_test = np.array(padded_test)
    train_labels = np.array(train_labels)
    valid_labels = np.array(valid_labels)
    test_labels = np.array(test_labels)

    return {
        "train": {"train_X": padded_train, "train_Y": train_labels},
        "valid": {"valid_X": padded_valid, "valid_Y": valid_labels},
        "test": {"test_X": padded_test, "test_Y": test_labels}
    }

if __name__ == "__main__":
    main()

