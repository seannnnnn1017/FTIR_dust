import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import gradio as gr
import joblib
import csv

# 全局變數來存儲訓練的模型及其評估結果
global_model = None
global_model_name = ''
global_mse = 0
global_r2 = 0
all_models_save={}

def save_model(model_name, model, mse, r2):
    if model_name == 'all':
        txt=''
        for i in all_models_save:
            i=all_models_save[i]
            save_model(i[0], i[1], i[2], i[3])
            txt += f'{i[0]}: MSE: {i[2]:.2f}, R² Score: {i[3]:.2f}\n'
        return txt


    # 讀取現有的 CSV 文件
    rows = []
    file_path = 'history/all_model_score.csv'
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # 讀取表頭
        for row in reader:
            rows.append(row)
    
    # 檢查是否已存在該模型名稱
    model_exists = False
    for row in rows:
        if row[0] == model_name:
            row[1] = mse
            row[2] = r2
            row[3] = r2/mse if mse != 0 else 'N/A'  # 避免除以零

            model_exists = True
            break

    # 如果不存在，添加新的行
    if not model_exists:
        rows.append([model_name, mse, r2, -r2/mse])

    # 寫回 CSV 文件
    with open(file_path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # 寫入表頭
        writer.writerows(rows)  # 寫入數據

    # 保存模型為 .pkl 文件
    joblib.dump(model, f'models/{model_name}.pkl')

def all_models():

    mse_r2 = []
    models_name = ['Random Forest', 'SVR', 'XGBoost']
    models = [RandomForestRegressor(n_estimators=1600, max_depth=120, min_samples_leaf=14, min_samples_split=2, n_jobs=-1),
             SVR(kernel='linear', C=0.097, epsilon=0.13),
             XGBRegressor(n_estimators=9, random_state=8400)]
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='True Values')

    for i, m in enumerate(models):
        # 訓練模型
        m.fit(X_train, y_train)
        # 預測
        y_pred = m.predict(X_test)
        # 評估
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse_r2.append([mse, r2])
        
        # 保存模型和評估結果
        all_models_save[models_name[i]]=[models_name[i], m, mse, r2]

        # 繪製結果圖
        plt.plot(y_pred, label=models_name[i], linestyle='--')

    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title(f'all models Predictions vs True Values - R² Score')

    # 保存圖像到暫存位置
    plt.savefig('prediction_plot.png')
    plt.close()       

    if auto_save.value:
        file_path = 'history/all_model_score.csv'

        print("Auto save all models.")
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)  # 讀取表頭
            for row in reader:
                if row==[]:
                    continue
                if row[0] in all_models_save:
                    #print(row)
                    if float(row[3]) < all_models_save[row[0]][3]/all_models_save[row[0]][2]: #r2/mse
                        save_model(row[0], all_models_save[row[0]][1], all_models_save[row[0]][2], all_models_save[row[0]][3])
                        print(f"Model {row[0]} updated.")

    txt = ''
    for i in mse_r2:
        txt += f'{models_name[mse_r2.index(i)]}: MSE: {i[0]:.2f}, R² Score: {i[1]:.2f}\n'

    return txt, 'prediction_plot.png'

def train_and_predict(data, model_type):
    global X_train, X_test, y_train, y_test, global_model, global_model_name, global_mse, global_r2
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
        global_model_name = model_type
        return all_models()
    else:
        raise ValueError("Invalid model type selected.")
    
    global_model_name = model_type
    # 訓練模型
    model.fit(X_train, y_train)
    # 預測
    y_pred = model.predict(X_test)
    # 評估
    global_mse = mean_squared_error(y_test, y_pred)
    global_r2 = r2_score(y_test, y_pred)
    global_model = model

    # 繪製結果圖
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='True Values')
    plt.plot(y_pred, label='Predictions', linestyle='--')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title(f'{model_type} Predictions vs True Values - R² Score: {global_r2:.2f}')

    # 保存圖像到暫存位置
    plt.savefig('prediction_plot.png')
    plt.close()

    if auto_save.value:
        file_path = 'history/all_model_score.csv'
        print(f"Auto save model {model_type}.")
        if model_type != 'all':
            model_score= global_r2/global_mse if global_mse != 0 else 0
            # 讀取現有的 CSV 文件
            rows = []
            
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                header = next(reader)  # 讀取表頭
                for row in reader:
                    if row==[]:
                        continue
                    if row[0] == model_type:
                        if float(row[3]) < model_score:
                            save_model(model_type, model, global_mse, global_r2)
                        else:
                            return f"MSE: {global_mse}\nR² Score: {global_r2}", 'prediction_plot.png'


                
            
    
    return f"MSE: {global_mse}\nR² Score: {global_r2}", 'prediction_plot.png'

def predict(file, model_type):
    data = pd.read_csv(file.name) if file.name.endswith('.csv') else pd.read_excel(file.name)
    return train_and_predict(data, model_type)

def checkboxes_changed():
    auto_save.value = not auto_save.value

def save_model_click():
    global global_model_name, global_model, global_mse, global_r2

    # 保存模型和更新CSV文件
    txt=save_model(global_model_name, global_model, global_mse, global_r2)
    return f"MSE: {global_mse}\nR² Score: {global_r2}\nSaved model file: models/{global_model_name}.pkl" if global_model_name != 'all' else txt

# Gradio 介面設定
inputs = [gr.File(label="Upload CSV or XLSX file"), gr.Radio(['Random Forest', 'SVR', 'XGBoost', 'all'], label="Select Model Type")]
outputs = [gr.Textbox(label="MSE and R² Score"), gr.Image(label="Prediction Plot")]

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Train model"):
            gr.Markdown("# Model Selection Interface")
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Upload CSV or XLSX file")
                    model_radio = gr.Radio(['Random Forest', 'SVR', 'XGBoost', 'all'], label="Select Model Type")
                    auto_save = gr.Checkbox(label="Auto Save Model")
                with gr.Column():
                    result_text = gr.Textbox(label="MSE and R² Score")
                    result_image = gr.Image(label="Prediction Plot")
                    with gr.Row():
                        predict_button = gr.Button("Predict")
                        save_model_button = gr.Button("Save Model")
        with gr.TabItem("Test model"):
            gr.Markdown("# Test Model Selection Interface")

    predict_button.click(predict, inputs=[file_input, model_radio], outputs=[result_text, result_image])
    save_model_button.click(save_model_click,inputs=None, outputs=result_text)
    auto_save.change(fn=checkboxes_changed,inputs=None, outputs=None)

demo.launch(share=True)
