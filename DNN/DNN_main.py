import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.decomposition import PCA
# 讀取資料
data = pd.read_excel('dataset/FTIR(調基準線).xlsx')
selected_features = data.iloc[:,:-1]
print(selected_features.head())
pca = PCA(n_components=10)
X_pca_5 = pca.fit_transform(selected_features)
features = pd.DataFrame(X_pca_5, columns=[f'PC{i+1}' for i in range(X_pca_5.shape[1])])
target = data.iloc[:, -1]
# 分割訓練集和測試集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2,random_state=42)

# 構建神經網絡模型
model = tf.keras.Sequential([
   tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),

    
    tf.keras.layers.Dense(1)  # 輸出層
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# 編譯模型
model.compile(optimizer='adam', loss='mse')

# 訓練模型
history = model.fit(X_train, y_train, epochs=200, batch_size=512, validation_split=0.2, verbose=1)

# 評估模型
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# 預測
y_pred = model.predict(X_test)

test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)
print(test_mse,test_r2)
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
