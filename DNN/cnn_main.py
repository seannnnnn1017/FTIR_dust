import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 讀取資料
data = pd.read_excel('dataset/FTIR(調基準線).xlsx')

# 提取特定列範圍作為特徵
features = pd.concat([data.iloc[:, 650:820], data.iloc[:, 850:1220], 
                      data.iloc[:, 1250:1750], data.iloc[:, 2800:3000]], axis=1)
#features= data.iloc[:,: -1]
target = data.iloc[:, -1]

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 特徵標準化
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_np = scaler.transform(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

# 定義輸入形狀 (樣本數, 特徵數, 通道數)
input_shape = (X_train.shape[1], 1)
print(f"Input shape: {input_shape}")

# 建立 CNN 模型
def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=7, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=3),
        layers.Conv1D(128, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),  # 添加 Flatten 層
        
        layers.Dense(128, activation='relu'),
        #layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        #layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        #layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        #layers.Dropout(0.3),
        layers.Dense(1)
    ])
    return model

# 創建模型
model = create_cnn_model(input_shape)

# 編譯模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mae')

# 顯示模型摘要
model.summary()

# 確保 y_train 和 y_test 也是 NumPy 陣列
y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
y_test_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test

# 訓練模型
history = model.fit(X_train_np, y_train_np, epochs=50, batch_size=128, validation_split=0.2, verbose=1)

# 評估模型
loss = model.evaluate(X_test_np, y_test_np)
print(f'Test loss: {loss}')

# 預測
y_pred = model.predict(X_test_np).flatten()
# 計算訓練集的預測
y_train_pred = model.predict(X_train_np).flatten()  # 使用訓練集進行預測
# 計算 MSE 和 R²
mse = mean_squared_error(y_test_np, y_pred)
r2 = r2_score(y_test_np, y_pred)
train_mse = mean_squared_error(y_train_np, y_train_pred)  # 修正這裡
train_r2 = r2_score(y_train_np, y_train_pred)  # 修正這裡
print(f'test MSE: {mse}, test R²: {r2}')
print(f'train MSE: {train_mse}, train R²: {train_r2}')
# 繪製訓練和驗證損失曲線
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# 繪製預測結果與真實值的比較
plt.figure(figsize=(10, 6))
plt.plot(y_test_np, label='True Values')
plt.plot(y_pred, label='Predictions', linestyle='--')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.title('Model Predictions vs True Values')
plt.show()

# 檢查目標變數的分佈
plt.figure(figsize=(8, 6))
plt.hist(y_train, bins=30)
plt.title('Target Variable Distribution')
plt.xlabel('Target Value')
plt.ylabel('Frequency')
plt.show()
