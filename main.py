#參數調整
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.model_selection import train_test_split

x = data_
y = data["醫療焚化爐"]

# 定义参数空间
param_dist = {
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'min_child_weight': [1, 2, 3, 4, 5, 6],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'n_estimators': [50,100, 200, 300, 400, 500],
    'alpha': [0, 0.1, 0.5, 1, 2],
    'lambda': [1, 1.5, 2, 2.5, 3]
}

# 初始化 XGBRegressor 模型
xgboost_model = xgb.XGBRegressor()


# 初始化 RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=xgboost_model, param_distributions=param_dist,
                                   scoring='r2', n_iter=200, cv=10, verbose=1, n_jobs=-1)

# 假设 X_train 和 y_train 是你的训练数据
# 训练模型


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=4513)

feature=(list(mark.loc[0]))[1:] #提取指紋庫特徵
processed_data_xtrain=[]
for i in x_train.values: #提取採樣資料
    new_value_list=[]
    for j in range(0,17):
        new_value=(i[j]*feature[j]) #將採樣資料乘以權重
        new_value_list.append(new_value)

    new_value_list=np.array(new_value_list)
    processed_data_xtrain.append(new_value_list)   #標記資料存起來

processed_data_xtest=[]
for i in x_test.values: #提取採樣資料
    new_value_list=[]
    for j in range(0,17):
        new_value=(i[j]*feature[j]) #將採樣資料乘以權重
        new_value_list.append(new_value)

    new_value_list=np.array(new_value_list)
    processed_data_xtest.append(new_value_list)   #標記資料存起來


random_search.fit(processed_data_xtrain, y_train)

# 查看最佳参数和最佳得分
print("Best parameters found: ", random_search.best_params_)
print("Best R^2 train ", random_search.score(processed_data_xtrain, y_train))
print("Best R^2 test: ", random_search.score(processed_data_xtest, y_test))