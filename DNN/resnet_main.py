import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# 讀取資料
data = pd.read_excel('dataset\Soil_Organic_Carbon_Data.xlsx')

# 提取特定列範圍作為特徵
features = pd.concat([data.iloc[:, 650:820], data.iloc[:, 850:1220], 
                      data.iloc[:, 1250:1750], data.iloc[:, 2800:3000]], axis=1)
features = data.iloc[:, 1:-1]
# 假設目標變量在數據集中為最後一列
target = data.iloc[:, -1]

# 分割訓練集和測試集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2,random_state=42)

# 創建模型
def residual_block(x, filters, kernel_size=3):
    y = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)
    y = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    
    if x.shape[-1] != filters:
        x = tf.keras.layers.Conv1D(filters, 1, padding='same')(x)
    
    out = tf.keras.layers.Add()([x, y])
    out = tf.keras.layers.Activation('relu')(out)
    return out

def create_resnet_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Reshape((-1, 1))(inputs)
    
    x = tf.keras.layers.Conv1D(64, 7, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(3, strides=2, padding='same')(x)
    
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(1024, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(256, activation='sigmoid')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_resnet_model((X_train.shape[1],))

# 編譯模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# 顯示模型摘要
model.summary()

# 訓練模型
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# 評估模型
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# 預測
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


# 繪製結果
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='True Values')
plt.plot(y_pred, label='Predictions', linestyle='--')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.title('Model Predictions vs True Values')
plt.show()
