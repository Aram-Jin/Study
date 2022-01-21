from codecs import ignore_errors
from unittest import result
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer, fetch_covtype
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')
import sklearn as sk
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA는 비지도학습 ~~ Y(target)값을 넣어서 전처리해줌; label값에 맞추어 차원 축소

#1. 데이터
# datasets = load_boston()
# datasets = fetch_california_housing()
datasets = load_breast_cancer()
# datasets = fetch_covtype()

x = datasets.data
y = datasets.target
print(x.shape)    # (569, 30)

# pca = PCA(n_components=8)
lda = LinearDiscriminantAnalysis()  # n_components=29를 넣었더니 "ValueError: n_components cannot be larger than min(n_features, n_classes - 1)." 
# x = pca.fit_transform(x)          # => 1개 이상 쓸수 없음!!
x = lda.fit_transform(x, y)
print(x.shape)  # (569, 1) -> label이 한개로 줄어듬

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