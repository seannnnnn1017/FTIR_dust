import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import gradio as gr
import os

# 構建隨機森林回歸模型
model = RandomForestRegressor(n_estimators=1600, max_depth=120, min_samples_leaf=14, min_samples_split=2, n_jobs=-1)

def train_model(data):
    # 讀取數據並進行處理
    features = pd.concat([data.iloc[:, 650:820], data.iloc[:, 850:1220], 
                          data.iloc[:, 1250:1750], data.iloc[:, 2800:3000]], axis=1)
    target = data.iloc[:, -1]
    
    # 確保所有列標題都是字符串類型
    features.columns = features.columns.astype(str)

    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # 訓練模型
    model.fit(X_train, y_train)
    
    return X_test, y_test

def predict(file):
    # 根據文件類型讀取數據
    if file.name.endswith('.csv'):
        data = pd.read_csv(file.name)
    else:
        data = pd.read_excel(file.name)
    
    X_test, y_test = train_model(data)
    
    # 預測
    y_pred = model.predict(X_test)
    
    # 評估
    mse = mean_squared_error(y_test, y_pred)
    
    # 繪製結果圖
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='True Values')
    plt.plot(y_pred, label='Predictions', linestyle='--')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title('Random Forest Predictions vs True Values')
    
    # 保存圖像到當前目錄
    plot_path = os.path.join(os.getcwd(), 'server/results_plot.png')
    plt.savefig(plot_path)
    
    return f"Test MSE: {mse}", plot_path

inputs = gr.File(label="Upload CSV or XLSX file")
outputs = [gr.Textbox(label="MSE"), gr.Image(label="Prediction Plot")]

iface = gr.Interface(fn=predict, inputs=inputs, outputs=outputs, live=True)
iface.launch(share=True)
