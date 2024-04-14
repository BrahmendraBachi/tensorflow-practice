import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def plotForOneResult(x_labelData, y_labelData, x_label='', y_label='', labelName='', title='', symbolToPlot='b', isShow=False):
    plt.plot(x_labelData, y_labelData, symbolToPlot, label=labelName)
    plt.xlabel(x_label) 

    plt.title(title)
    plt.ylabel(y_label)

    plt.grid(True)
    if isShow:
        plt.show()

def plotForTwoResults(x_data, y1_data, y2_data, x_label='', y_label='', y1_label='', y2_label='', title='', isLegend=True, isShow=False, isGrid = True):
    plt.plot(x_data, y1_data, 'bo', label=y1_label)
    plt.plot(x_data, y2_data, 'b', label=y2_label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if isLegend:
        plt.legend()
    if isShow:
        plt.show()

def plot_for_one_model(history, isValidation=False):

    epochs = range(1, len(history.history['loss']) + 1)

    training_loss = history.history['loss']
    training_accuracy = history.history["acc"]
    plt.figure(figsize=(12, 4))


    if isValidation:
        validation_loss = history.history["val_loss"]
        validation_accuracy = history.history["val_acc"]

        plt.subplot(1, 2, 1)

        plotForTwoResults(
            epochs,
            training_loss,
            validation_loss,
            x_label="Epochs",
            y_label="Losses",
            y1_label="Training Loss",
            y2_label="Validation_Loss",
            title="Training and Validation Loss",
            isShow=False
        )

        plt.subplot(1, 2, 2)

        plotForTwoResults(
            epochs,
            training_accuracy,
            validation_accuracy,
            x_label="Epochs",
            y_label="Accuracies",
            y1_label="Training_Accuracy",
            y2_label="Validation_Accuracy",
            title="Training and Validation Accuracy",
            isShow=False
        )

    else:

        plt.subplot(1, 2, 1)

        plotForOneResult(
            epochs,
            training_loss,
            x_label="Epochs",
            y_label="Training_Loss",
            title="Training_Loss",
            isShow=False
        )

        plt.subplot(1, 2, 2)

        plotForOneResult(
            epochs,
            training_accuracy,
            x_label="Epochs",
            y_label="Training_Accuracy",
            title="Training_Accuracy",
            isShow=False
        )

    plt.show()

def plot_for_two_model(history1, history2, isValidation=False, with_combined=False):

    training_loss1 = history1.history['loss']
    training_accuracy1 = history1.history['acc']

    training_loss2 = history2.history['loss']
    training_accuracy2 = history2.history['acc']

    plt.figure(figsize=(12, 10))

    if with_combined:

        epochs = range(1, len(history1.history['loss']) + 1)

        plt.subplot(2, 2, 1)

        plotForTwoResults(
            epochs,
            training_loss1,
            training_loss2,
            x_label="Epochs",
            y_label="Losses",
            y1_label="Training Loss_1",
            y2_label="Training Loss_2",
            title="Training Losses for Model_1 and Model_2"
        )

        plt.subplot(2, 2, 2)

        plotForTwoResults(
            epochs,
            training_accuracy1,
            training_accuracy2,
            x_label="Epochs",
            y_label="Accuracies",
            y1_label="Training Accuracy_1",
            y2_label="Training Accuracy_2",
            title="Training Accuracies for Model_1 and Model_2"
        )

        if isValidation:
            validation_loss1 = history1.history['val_loss']
            validation_accuracy1 = history1.history['val_acc']

            validation_loss2 = history2.history['val_loss']
            validation_accuracy2 = history2.history['val_acc']

            plt.subplot(2, 2, 3)

            plotForTwoResults(
                epochs,
                validation_loss1,
                validation_loss2,
                x_label="Epochs",
                y_label="Losses",
                y1_label="Validation Loss_1",
                y2_label="Validation Loss_2",
                title="Validation Losses for Model_1 and Model_2"
            )

            plt.subplot(2, 2, 4)

            plotForTwoResults(
                epochs,
                validation_accuracy1,
                validation_accuracy2,
                x_label="Epochs",
                y_label="Accuracies",
                y1_label="Validation Accuracy_1",
                y2_label="Validation Accuracy_2",
                title="Validation Accuracies for Model_1 and Model_2")
    else:

        epochs_1 = range(1, len(history1.history['val_loss']) + 1)
        epochs_2 =range(1, len(history2.history['val_loss']) + 1)

        if isValidation:

            validation_loss1 = history1.history['val_loss']
            validation_accuracy1 = history1.history['val_acc']

            validation_loss2 = history2.history['val_loss']
            validation_accuracy2 = history2.history['val_acc']

            plt.subplot(2, 2, 1)

            plotForTwoResults(
                epochs_1,
                training_loss1,
                validation_loss1,
                x_label="Epochs",
                y_label="Losses",
                y1_label="Training Loss_1",
                y2_label="Validation Loss_1",
                title="Training and Validation Losses for Model_1"
            )


            plt.subplot(2, 2, 2)

            plotForTwoResults(
                epochs_1,
                training_accuracy1,
                validation_accuracy1,
                x_label="Epochs",
                y_label="Accuracies",
                y1_label="Training Accuracy_1",
                y2_label="Validation Accuracy_1",
                title="Training and Validation Accuracies for Model_1"
            )

            plt.subplot(2, 2, 3)

            plotForTwoResults(
                epochs_1,
                training_loss2,
                validation_loss2,
                x_label="Epochs",
                y_label="Losses",
                y1_label="Training Loss_2",
                y2_label="Validation Loss_2",
                title="Training and Validation Losses for Model_2"
            )

            plt.subplot(2, 2, 4)

            plotForTwoResults(
                epochs_1,
                training_accuracy2,
                validation_accuracy2,
                x_label="Epochs",
                y_label="Accuracies",
                y1_label="Training Accuracy_2",
                y2_label="Validation Accuracy_2",
                title="Training and Validation Accuracies for Model_2"
            )
    plt.show()

def plot_image(image, isShow=True):
    plt.gray()
    plt.axis('off')
    plt.grid(False)
    plt.imshow(image)
    if isShow:
        plt.show()

def visualize_model(model, img):
    successive_outputs = [layer.output for layer in model.layers[1:]]
    visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

    successive_feature_maps = visualization_model.predict(img)
    layer_names = [layer.name for layer in model.layers[1:]]

    for layer_name, feature_map in zip(layer_names, successive_feature_maps):

        if len(feature_map.shape) == 4:

            n_features = feature_map.shape[-1]
            size = feature_map.shape[1]
            display_grid = np.zeros((size, size * n_features))

            for i in range(n_features):

                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128

                x = np.clip(x, 0, 255).astype('uint8')

                display_grid[:, i * size : (i + 1) * size] = x

            scale = 20./n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')

def plot_for_multiple_models(histories):
    histories = [history.history for history in histories]

    fig_size = (12, 10)  # Width, Height
    epochs = range(1, len(histories[0]["loss"]) + 1)
    fig, axs = plt.subplots(2, 2, figsize=fig_size)  # Creating a 2x2 grid of subplots
    for (i, history) in enumerate(histories):
        training_loss = history['loss']
        training_acc = history['acc']
        validation_loss = history['val_loss']
        validation_acc = history['val_acc']
        label_name = "Model-" + str(i + 1)
        axs[0, 0].plot(epochs, training_loss, label=label_name)
        axs[0, 1].plot(epochs, training_acc, label=label_name)
        axs[1, 0].plot(epochs, validation_loss, label=label_name)
        axs[1, 1].plot(epochs, validation_acc, label=label_name)

    titleNames = [["Training Losses", "Traning Accuracies",], ["Validation Losses", "Validation Accuracies"]]
    for i in range(2):
        for j in range(2):
            axs[i, j].grid(True)
            axs[i, j].legend()
            axs[i, j].set_title(titleNames[i][j])
    plt.show()


class myCallback(tf.keras.callbacks.Callback):

    def __init__(self):
        self.all_model_weights = []
        self.all_logs = []
        self.eff_model_weights = None
        self.max_val_acc = 0
        self.logs = None
        self.epochs = None

    def on_epoch_end(self, epochs, logs={}):
        print()
        self.all_model_weights.append(self.model.get_weights())
        self.all_logs.append(logs)
        if logs.get("val_acc") > self.max_val_acc:
            self.max_val_acc = logs.get("val_acc")
            self.eff_model_weights = self.model.get_weights()
            self.logs = logs
            self.epochs = epochs
            print(f"max_val_acc changed at epoch: {epochs}")

        print(f"Val_Accuracy at epoch: {logs.get('val_acc')}")
        print(f"Maximum Val_Accuracy at epoch: {self.max_val_acc}")

    def get_efficient_model_weights(self):
        return self.eff_model_weights

    def max_efficient_logs(self):
        return self.logs

    def get_all_logs(self):
        return self.all_logs

    def get_all_model_weights(self):
        return self.all_models_weights


callbacks = myCallback()