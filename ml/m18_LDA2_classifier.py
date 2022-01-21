import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import warnings
warnings.filterwarnings(action='ignore')

# LDA는 비지도학습 ~~ Y(target)값을 넣어서 전처리해줌; label값에 맞추어 차원 축소

#1. 데이터
# datasets = load_iris()
datasets = load_breast_cancer()
# datasets = load_wine()
# datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print("LDA 전 : ", x.shape)    

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True, stratify=y)   # stratify

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
from xgboost import XGBRegressor, XGBClassifier
model = XGBClassifier()


#3. 훈련
# model.fit(x_train, y_train, eval_metric='error')  # 이진분류일떄
model.fit(x_train, y_train, eval_metric='merror')  # 다중분류일때
# model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)

'''
iris
LDA 전 :  (150, 4)
LDA 후:  (120, 2)
결과 :  1.0

cancer
LDA 전 :  (569, 30)
LDA 후:  (455, 1)
결과 :  0.9473684210526315

wine
LDA 전 :  (178, 13)
LDA 후:  (142, 2)
결과 :  1.0

fetch_covtype
LDA 전 :  (569, 30)
LDA 후:  (455, 1)
결과 :  0.9473684210526315
'''


'''
LinearDiscriminantAnalysis 사용
결과 :  0.9824561403508771  

=================================================
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