import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 讀取數據
data = pd.read_excel('dataset/FTIR_Data.xlsx')

# 選取特定範圍的列作為特徵
features = pd.concat([data.iloc[:, 650:820], data.iloc[:, 850:1220], 
                      data.iloc[:, 1250:1750], data.iloc[:, 2800:3000]], axis=1)
features = data.iloc[:, :-1]
# 目標變量
target = data.iloc[:, -1]



# 分割資料集為訓練集和測試集
random_state = 25
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=random_state)

# 建立和訓練 SVR 模型
model = SVR(kernel='linear', C=0.097, epsilon=0.13)
model.fit(X_train, y_train)

# 預測訓練集和測試集的結果
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 計算 R² 和 MSE
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# 輸出結果
print(f"Train R²: {train_r2}")
print(f"Test R²: {test_r2}")
print(f"Train MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
