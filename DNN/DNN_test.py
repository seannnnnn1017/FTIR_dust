import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 生成數據
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y = np.sin(x)

# 構建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_dim=1, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


# 編譯模型
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# 訓練模型
history = model.fit(x, y, epochs=100, batch_size=32, verbose=0)

# 預測
y_pred = model.predict(x)

# 繪製結果
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='True Function')
plt.plot(x, y_pred, label='DNN Prediction', linestyle='--')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('DNN Prediction of $y = \sin(x)$')
plt.show()
