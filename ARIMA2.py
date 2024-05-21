x

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

# Create ARIMA model
def create_arima_model(train_data, p, d, q):
    model = sm.tsa.ARIMA(train_data, order=(p, d, q))
    fitted_model = model.fit(disp=0)
    return fitted_model

# Find the best ARIMA parameters (p, d, q) using grid search, as shown in the previous example.

# Replace these values with your best parameters
best_p, best_d, best_q = 1, 1, 1

# Create and train the ARIMA model using the best parameters
arima_model = create_arima_model(train_data, best_p, best_d, best_q)

# Make predictions on the test data
predictions = arima_model.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(data, label='Original Data', color='blue')
plt.plot(train_data, label='Training Data', color='green')
plt.plot(np.arange(len(train_data), len(train_data) + len(test_data)), test_data, label='Test Data', color='orange')
plt.plot(np.arange(len(train_data), len(train_data) + len(test_data)), predictions, label='Predictions', color='red')
plt.legend()
plt.title('ARIMA Model')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
