import os

def change_data_type(data_root_path):
    # 獲取指定目錄下的所有檔案
    for filename in os.listdir(data_root_path):
        file_path = os.path.join(data_root_path, filename)
        
        # 檢查是否為檔案且副檔名為 .asc
        if os.path.isfile(file_path) and filename.endswith('.asc'):
            # 取得不含副檔名的檔案名稱
            name_without_ext = os.path.splitext(filename)[0]
            
            # 新的檔案名稱
            new_filename = f"{name_without_ext}.txt"
            new_file_path = os.path.join(data_root_path, new_filename)
            
            # 將 .asc 檔案重新命名為 .txt
            os.rename(file_path, new_file_path)
            
            # 修改檔案內容，刪除前 25 行
            with open(new_file_path, 'r') as f:
                lines = f.readlines()
            lines = lines[25:]
            # 刪除後 100 行
            lines = lines[:-100]
            with open(new_file_path, 'w') as f:
                f.writelines(lines)
            
            print(f"已處理檔案：{new_file_path}")

import pandas as pd

def collect_data_horizontal(data_root_path, output_excel_path):
    # 初始化一個空的 DataFrame，用於存放轉置後的穿透率數據
    transmittance_data = pd.DataFrame()

    # 遍歷指定目錄下的所有子目錄和檔案
    for root, dirs, files in os.walk(data_root_path):
        for filename in files:
            # 只處理以 .txt 結尾的檔案
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                sample_name = os.path.splitext(filename)[0]

                try:
                    df = pd.read_csv(file_path, sep='\\s+', header=None)
                    df.columns = ['波長', '穿透率']

                    df = df.sort_values(by='波長', ascending=False)

                    df.set_index('波長', inplace=True)

                    # 轉置 DataFrame，使波長成為列
                    df_transposed = df.T

                    # 設定索引為樣本名稱
                    df_transposed.index = [sample_name]

                    # 將數據追加到總的 DataFrame 中
                    transmittance_data = pd.concat([transmittance_data, df_transposed], axis=0, sort=False)

                    print(f"已處理檔案：{file_path}")
                except Exception as e:
                    print(f"處理檔案時發生錯誤：{file_path}，錯誤信息：{e}")

    # 按波長（列）降序排序
    transmittance_data = transmittance_data.reindex(sorted(transmittance_data.columns, reverse=True), axis=1)

    # 重置索引，將樣本名稱作為一列
    transmittance_data.reset_index(inplace=True)
    transmittance_data.rename(columns={'index': '樣本'}, inplace=True)

    # 將列名列表更新為 [樣本, 波長1, 波長2, ...]
    wavelengths = ['樣本'] + list(transmittance_data.columns[1:])
    transmittance_data.columns = wavelengths

    try:
        transmittance_data.to_excel(output_excel_path, index=False)
        print(f"所有數據已寫入 Excel 檔案：{output_excel_path}")
    except Exception as e:
        print(f"寫入 Excel 檔案時發生錯誤：{e}")

if __name__ == "__main__":
    data_root_path = 'C:\\Users\\User\\Desktop\\土壤DATA\\FTIR未處裡\\學思湖坡地果園FTIR'  # 請根據實際路徑修改
    change_data_type(data_root_path)
    output_excel_path = 'C:\\Users\\User\\Desktop\\土壤DATA\\FTIR未處裡\\學思湖坡地果園FTIR.xlsx'  # 輸出的 Excel 檔案名
    collect_data_horizontal(data_root_path, output_excel_path)