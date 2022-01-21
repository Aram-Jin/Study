import numpy as np
import time
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
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, KFold

parameters = [
    {"n_estimators":[100,200,300], "learning_rate":[0.3, 0.001, 0.01],"max_depth":[4,5]},
    {"n_estimators":[110], "learning_rate":[0.1, 0.01],"max_depth":[5], "colsample_bytree":[0.6]},
    {"n_estimators":[90], "learning_rate":[0.1, 0.5],"max_depth":[6], "colsample_bytree":[0.9, 1],"colsample_bylevel":[0.6,0.9]}
]
# param_grid = {
#     'n_estimators': [100, 150, 200, 250],
#     "learning_rate":[0.3, 0.001, 0.01],
#     'max_depth': [None, 6, 9, 12],
#     'min_samples_split': [0.01, 0.05, 0.1],
#     'max_features': ['auto', 'sqrt'],
# }

model = GridSearchCV(XGBClassifier(), parameters, cv=5, verbose=1, refit=True, n_jobs = -1)


#3. 훈련
start = time.time()
model.fit(x_train, y_train, eval_metric='merror')
end = time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)
print("time: ", end - start)

'''
LDA 전 :  (60000, 28, 28)
LDA 후:  (60000, 9)
결과 :  0.9177
time:  3674.9865612983704
'''
