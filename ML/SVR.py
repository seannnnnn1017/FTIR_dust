#%%
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import matplotlib.pyplot as plt

# 讀取資料
data = pd.read_excel('C:/Users/fishd/Desktop/Github/FTIR_dust/dataset/FTIR(調基準線).xlsx')

# 提取除了特定列範圍以外的所有列作為特徵
features = data.drop(data.columns[np.r_[650:820, 850:1220, 1250:1750, 2800:3000]], axis=1)
features.columns = features.columns.astype(str)  # 將所有特徵名稱轉換為字符串
features=features.iloc[:, :-1]
features
#%%
# 目標變量
target = data.iloc[:, -1].values.reshape(-1, 1)
target
#%%


# 分割資料集為訓練集和測試集
random_state = 25
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=random_state)

# 記錄損失
losses = []

# 建立和訓練 SVR 模型
model = SVR(kernel='linear', C=0.097, epsilon=0.13)

# 訓練模型並記錄損失
for i in range(100):  # 假設進行100次訓練
    model.fit(X_train, y_train.ravel())
    y_train_pred = model.predict(X_train)
    loss = mean_squared_error(y_train, y_train_pred)
    losses.append(loss)

# 預測測試集的結果
y_test_pred = model.predict(X_test)

# 繪製損失圖
plt.figure(figsize=(12, 5))

# 繪製損失
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 繪製測試預測
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, label='Test Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # 理想預測線
plt.title('Test Predictions')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()

plt.tight_layout()
plt.show()  # 顯示圖形
