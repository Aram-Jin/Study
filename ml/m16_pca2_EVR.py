import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')
import sklearn as sk

#1. 데이터
# datasets = load_boston()
# datasets = fetch_california_housing()
datasets = load_breast_cancer()


x = datasets.data
y = datasets.target
print(x.shape)    # (506, 13)
# print(x.shape)    # (20640, 8)

pca = PCA(n_components=10)
x = pca.fit_transform(x)
# print(x)
print(x.shape)  

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)
print(cumsum)

import matplotlib.pyplot as plt
plt.plot(cumsum)
# plt.plot(pca_EVR)
plt.grid()
plt.show()

"""
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)

#2. 모델
from xgboost import XGBRegressor, XGBClassifier
# model = XGBRegressor()
model = XGBClassifier()


#3. 훈련
model.fit(x_train, y_train, eval_metric='error')

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)

print(sk.__version__)

"""
'''
load_boston   (506, 13)
(506, 8)
결과 :  0.7856968255504542
1.0.1

fetch_california_housing   (20640, 8)
(20640, 8)
결과 :  0.7839977478554678
1.0.1

load_breast_cancer   (569, 30)
(569, 8)
결과 :  0.9473684210526315
1.0.1
'''