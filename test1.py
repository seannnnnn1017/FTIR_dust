import pandas as pd

# 讀取 Excel 文件
file_path = 'dataset/FTIR_Data.xlsx'
data = pd.read_excel(file_path, header=0)  # 確保 header=0 表示第一行是標題

# 顯示資料的前幾行和所有欄位名稱
print("資料前五行：")
print(data.head())

print("\n所有欄位名稱：")
print(data.columns.tolist())  # 顯示所有欄位名稱

# 檢查標識符欄位是否存在
identifier_column = '樣本'  # 請確認這裡是你的標識符欄位名稱
if identifier_column in data.columns:
    print(f"標識符欄位 '{identifier_column}' 存在。")
else:
    print(f"標識符欄位 '{identifier_column}' 不存在。")
