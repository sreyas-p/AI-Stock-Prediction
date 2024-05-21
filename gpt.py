import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
import tensorflow as tf

# Generate synthetic time series data
def generate_time_series(length=100):
    np.random.seed(42)
    return np.cumsum(np.random.randn(length))

# Load or generate your own time series data
# Replace this with your actual data
data = generate_time_series()

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Grid search for ARIMA parameters
def find_best_arima_params(train_data):
    stepwise_fit = auto_arima(train_data, start_p=1, start_q=1,
                              max_p=3, max_q=3, m=1,
                              seasonal=False,
                              trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)
    return stepwise_fit.order

# Find the best ARIMA parameters
best_p, best_d, best_q = find_best_arima_params(train_data)

# Build and train the ARIMA model using TensorFlow
def build_arima_model(train_data, p, d, q):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_shape=(p + d + q,))
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Prepare data for training
    X, y = [], []
    for i in range(len(train_data) - (p + d + q)):
        X.append(train_data[i:i+p+d+q])
        y.append(train_data[i+p+d+q])
    X, y = np.array(X), np.array(y)

    # Train the model
    model.fit(X, y, epochs=100, batch_size=1, verbose=0)
    return model

# Build the ARIMA model using the best parameters
arima_model = build_arima_model(train_data, best_p, best_d, best_q)

# Make predictions on the test data
def make_predictions(model, test_data, p, d, q):
    predictions = []
    for i in range(len(test_data) - (p + d + q)):
        X_test = test_data[i:i+p+d+q]
        prediction = model.predict(np.array([X_test]))[0][0]
        predictions.append(prediction)
    return predictions

# Make predictions using the ARIMA model
predictions = make_predictions(arima_model, test_data, best_p, best_d, best_q)

# Calculate the Root Mean Squared Error (RMSE)
def calculate_rmse(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    return np.sqrt(mse)

rmse = calculate_rmse(test_data[best_p+best_d+best_q:], predictions)
print("Best ARIMA Parameters (p, d, q):", (best_p, best_d, best_q))
print("RMSE on Test Data:", rmse)