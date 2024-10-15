import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 讀取資料
data = pd.read_excel('dataset/FTIR_Data.xlsx')
features = pd.concat( [data.iloc[:, 900:1000]], axis=1)
features = data.iloc[:,:-1]
target = data.iloc[:, -1]

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 構建隨機森林回歸模型
model = RandomForestRegressor(n_estimators=1600, max_depth=120, min_samples_leaf=14, min_samples_split=2, n_jobs=-1,)
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)  # 將所有負值變為0

# 評估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Test MSE: {mse}')
print(f'R-squared: {r2}')

# 繪製測試資料的實際值與預測值
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.legend()
plt.show()