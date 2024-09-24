import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import gradio as gr

def all_models():
    mse_r2 = []
    models_name = ['Random Forest', 'SVR', 'XGBoost']
    models = [RandomForestRegressor(n_estimators=1600, max_depth=120, min_samples_leaf=14, min_samples_split=2, n_jobs=-1),
             SVR(kernel='linear', C=0.097, epsilon=0.13),
             XGBRegressor(n_estimators=9, random_state=8400)]
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='True Values')

    for i,m  in enumerate(models):
        # 訓練模型
        m.fit(X_train, y_train)
        # 預測
        y_pred = m.predict(X_test)
        # 評估
        mse_r2.append([mean_squared_error(y_test, y_pred),r2_score(y_test, y_pred)])    
        # 繪製結果圖

        plt.plot(y_pred, label=models_name[i], linestyle='--')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title(f'all models Predictions vs True Values - R² Score')

    # 保存圖像到暫存位置
    plt.savefig('prediction_plot.png')
    plt.close()       
    txt=''
    for i in mse_r2:
        txt+=f'{models_name[mse_r2.index(i)]}: MSE: {i[0]:.2f}, R² Score: {i[1]:.2f}\n'

    return txt, 'prediction_plot.png'

def train_and_predict(data, model_type):
    global X_train, X_test, y_train, y_test
    # 讀取數據並進行處理
    features = pd.concat([data.iloc[:, 650:820], data.iloc[:, 850:1220],
                          data.iloc[:, 1250:1750], data.iloc[:, 2800:3000]], axis=1)
    features = data.iloc[:, :-1]
    features = pd.concat( [data.iloc[:, 900:1000]], axis=1)
    target = data.iloc[:, -1]

    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    # 根據選擇的模型類型初始化模型
    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=1600, max_depth=120, min_samples_leaf=14, min_samples_split=2, n_jobs=-1)
    elif model_type == 'SVR':
        model = SVR(kernel='linear', C=0.097, epsilon=0.13)
    elif model_type == 'XGBoost':
        model = XGBRegressor(n_estimators=9, random_state=8400)
    elif model_type == 'all':
        return all_models()
    else:
        raise ValueError("Invalid model type selected.")

    # 訓練模型
    model.fit(X_train, y_train)
    # 預測
    y_pred = model.predict(X_test)
    # 評估
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 繪製結果圖
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='True Values')
    plt.plot(y_pred, label='Predictions', linestyle='--')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title(f'{model_type} Predictions vs True Values - R² Score: {r2:.2f}')

    # 保存圖像到暫存位置
    plt.savefig('prediction_plot.png')
    plt.close()
    
    return f"Test MSE: {mse}\nR² Score: {r2}", 'prediction_plot.png'


def predict(file, model_type):
    data = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    return train_and_predict(data, model_type)

# Gradio 介面設定
inputs = [gr.File(label="Upload CSV or XLSX file"), gr.Radio(['Random Forest', 'SVR', 'XGBoost', 'all'], label="Select Model Type")]
outputs = [gr.Textbox(label="MSE and R² Score"), gr.Image(label="Prediction Plot")]

with gr.Blocks() as demo:
    
    gr.Markdown("# Model Selection Interface")
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV or XLSX file")
            model_radio = gr.Radio(['Random Forest', 'SVR', 'XGBoost', 'all'], label="Select Model Type")
        with gr.Column():
            result_text = gr.Textbox(label="MSE and R² Score")
            result_image = gr.Image(label="Prediction Plot")
    gr.Button("Predict").click(predict, inputs=[file_input, model_radio], outputs=[result_text, result_image])

demo.launch(share=True)
