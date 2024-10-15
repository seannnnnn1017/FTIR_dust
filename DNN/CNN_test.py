import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Generate synthetic data
def generate_data(n_samples=100):
    X = np.linspace(-10, 10, n_samples).reshape(-1, 1)
    y = 2 * X**2 + X + 13
    return X, y

# Prepare data
X, y = generate_data(100)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build 1D CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, kernel_size=1, activation='relu', input_shape=(1, 1)),
    tf.keras.layers.Conv1D(32, kernel_size=1, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Reshape X_train and X_test for CNN input
X_train = X_train.reshape(-1, 1, 1)
X_test = X_test.reshape(-1, 1, 1)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)


# 繪製損失曲線
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss')
plt.show()


# Plot true values vs predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Values')
plt.plot(y_pred, label='Predictions', linestyle='--')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.title('1D CNN Predictions vs True Values')
plt.show()
