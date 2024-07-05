import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 讀取資料
data = pd.read_excel('dataset/FTIR_Data.xlsx')
features = pd.concat([data.iloc[:, 650:820], data.iloc[:, 850:1220], 
                      data.iloc[:, 1250:1750], data.iloc[:, 2800:3000]], axis=1)
target = data.iloc[:, -1]

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 構建隨機森林回歸模型
model = RandomForestRegressor(n_estimators=1600, max_depth=120,min_samples_leaf=14, min_samples_split=2, n_jobs=-1)
{'max_depth': 2, 'min_samples_leaf': 4, 'min_samples_split': 20, 'n_estimators': 600}
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估
mse = mean_squared_error(y_test, y_pred)
print(f'Test MSE: {mse}')

# 繪製結果
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='True Values')
plt.plot(y_pred, label='Predictions', linestyle='--')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.title('Random Forest Predictions vs True Values')
plt.show()
