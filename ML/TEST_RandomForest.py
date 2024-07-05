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
from sklearn.model_selection import GridSearchCV


# 定義參數網格
param_grid = {
    'n_estimators': [600, 700, 800],
    'max_depth': [2, 3, 4],
    'min_samples_split': [20,25,30],
    'min_samples_leaf': [3, 4, 5]
}

# 創建 GridSearchCV 對象
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
                           param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)

# 執行網格搜索
grid_search.fit(X_train, y_train)

# 打印最佳參數和最佳模型的分數
print("Best parameters:", grid_search.best_params_)
print("Best score (MSE):", -grid_search.best_score_)
# {'max_depth': 3, 'min_samples_leaf': 4, 'min_samples_split': 20, 'n_estimators': 700}
# {'max_depth': 2, 'min_samples_leaf': 4, 'min_samples_split': 20, 'n_estimators': 600}