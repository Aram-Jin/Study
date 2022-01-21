import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import warnings
warnings.filterwarnings(action='ignore')

# LDA는 비지도학습 ~~ Y(target)값을 넣어서 전처리해줌; label값에 맞추어 차원 축소

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("LDA 전 : ", x_train.shape)    

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# pca = PCA(n_components=8)
lda = LinearDiscriminantAnalysis()  
# x = pca.fit_transform(x)        

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
LDA 전 :  (60000, 28, 28)
LDA 후:  (60000, 9)
결과 :  0.9163
'''
