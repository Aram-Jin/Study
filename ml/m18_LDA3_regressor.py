import numpy as np
from sklearn.datasets import load_boston, load_diabetes, fetch_california_housing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import warnings
warnings.filterwarnings(action='ignore')

# LDA는 비지도학습 ~~ Y(target)값을 넣어서 전처리해줌; label값에 맞추어 차원 축소

#1. 데이터
# datasets = load_boston()
# datasets = load_diabetes()
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
y = np.round(y,0)
print("LDA 전 : ", x.shape)    

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)   # stratify

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# pca = PCA(n_components=8)
lda = LinearDiscriminantAnalysis()  # n_components=29를 넣었더니 "ValueError: n_components cannot be larger than min(n_features, n_classes - 1)." 
# x = pca.fit_transform(x)          # => 1개 이상 쓸수 없음!!

# x_train = lda.fit_transform(x_train, y_train)
lda.fit(x_train, y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

print("LDA 후: ", x_train.shape) 

#2. 모델
from xgboost import XGBRegressor
model = XGBRegressor()

#3. 훈련
# model.fit(x_train, y_train, eval_metric='error')  # 이진분류일떄
# model.fit(x_train, y_train, eval_metric='merror')  # 다중분류일때
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)

'''
load_boston
LDA 전 :  (506, 13)
LDA 후:  (404, 13)
결과 :  0.867137027720706

load_diabetes
LDA 전 :  (442, 10)
LDA 후:  (353, 10)
결과 :  0.313354229055848

fetch_california_housing
LDA 전 :  (20640, 8)
LDA 후:  (16512, 5)
결과 :  0.6575931583547636
'''
