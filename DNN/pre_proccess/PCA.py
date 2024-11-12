import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
# 讀取資料
data = pd.read_excel('dataset/FTIR(調基準線).xlsx')



# 排除最後一列並對剩餘資料進行 PCA
selected_features = data.iloc[:,:-1]
print(selected_features.head())
pca = PCA(n_components=5)
X_pca_5 = pca.fit_transform(selected_features)
X_pca_5_df = pd.DataFrame(X_pca_5, columns=[f'PC{i+1}' for i in range(X_pca_5.shape[1])])
print(X_pca_5_df.head())
#PCA 後的 5 個主成分

#print("PCA 到 5 個主成分的資料:\n", X_pca_5)
