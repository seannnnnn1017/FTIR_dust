import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# 讀取資料
data = pd.read_excel('dataset/FTIR_Data.xlsx')

# 提取特定列範圍作為特徵
features = pd.concat([data.iloc[:, 650:820], data.iloc[:, 850:1220], 
                      data.iloc[:, 1250:1750], data.iloc[:, 2800:3000]], axis=1)

# 假設目標變量在數據集中為最後一列
target = data.iloc[:, -1]

# 分割訓練集和測試集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2,random_state=42)

# 構建神經網絡模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1048, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(512, activation='relu'), # Dropout 正則化 
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # 输出层
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# 編譯模型
model.compile(optimizer='adam', loss='mse')

# 訓練模型
history = model.fit(X_train, y_train, epochs=100, batch_size=512, validation_split=0.2, verbose=1)

# 評估模型
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# 預測
y_pred = model.predict(X_test)

# 繪製結果
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='True Values')
plt.plot(y_pred, label='Predictions', linestyle='--')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.title('Model Predictions vs True Values')
plt.show()
