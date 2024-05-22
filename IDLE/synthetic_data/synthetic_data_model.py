import os

from IDLE.synthetic_data.utils.common_functions import *

is_Training = False
split_time = 1000
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000


def main():
    global is_Training
    # Parameters
    time = np.arange(4 * 365 + 1, dtype="float32")
    baseline = 10
    amplitude = 40
    slope = 0.05
    noise_level = 5
    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    series += noise(time, noise_level, seed=42)

    train_data = series[:split_time]
    time_train = time[:split_time]
    valid_data = series[split_time:]
    time_valid = time[split_time:]

    model = build_model()

    train_dataset = windowed_dataset(train_data, window_size, batch_size, shuffle_buffer_size)
    valid_dataset = forecast_dataset(series[split_time - window_size:-1], window_size, batch_size)

    if not os.path.exists("synthetic_data_model.keras"):
        is_Training = True

    if not is_Training:
        model.load_weights('synthetic_data_model.keras')
        print("Forecasting.... ")
        forecast = model.predict(valid_dataset).squeeze()
        print("Completed")

        calculate_metrics(valid_data, forecast)
        return

    print("\n************************************ Training Started ************************************\n")
    history = model.fit(
        train_dataset,
        epochs=500
    )
    print("\n************************************ Completed ************************************\n")

    print("Forecasting.... ")
    forecast = model.predict(valid_dataset).squeeze()
    print("Completed")

    print("\nPlotting Series")
    plot_series(time_valid, (valid_data, forecast))
    print("Completed")

    model.save('synthetic_data_model.keras')

    calculate_metrics(valid_data, forecast)


def calculate_metrics(valid_data, predicted_forecast):
    print("\nCalculating metrics....")
    print(f"MSE: {tf.keras.metrics.mse(valid_data, predicted_forecast).numpy()}")
    print(f"MAE: {tf.keras.metrics.mae(valid_data, predicted_forecast).numpy()}")
    print("Completed")


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_shape=[window_size])
    ])

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)
    loss = tf.keras.losses.Huber()

    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
    return model


if __name__ == "__main__":
    main()