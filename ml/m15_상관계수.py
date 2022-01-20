import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datasets.data
y = datasets.target
# print(x)
# print(y)
# print(type(x))   # <class 'numpy.ndarray'>

# df = pd.DataFrame(x, columns=datasets.feature_names)
# df = pd.DataFrame(x, columns=datasets['feature_names'])

df = pd.DataFrame(x, columns=[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']])
print(df)

df['Target(Y)'] = y
print(df)

print("====================================== 상관계수 히트 맵 ==========================================")
print(df.corr())  # 여기서 상관관계는 linear로 결정된 것 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()