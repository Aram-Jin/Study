import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler

# model : RandomForestClassifier

#1. 데이터
path = "../_data/kaggle/bike/"    

train = pd.read_csv(path + 'train.csv')
#print(train.shape)  # (10886, 12)
test_file = pd.read_csv(path + 'test.csv')
#print(test_file.shape)  # (6493, 9)
submit_file = pd.read_csv(path + 'sampleSubmission.csv')
#print(submit_file.shape)  # (6493, 2)

x = train.drop(['datetime', 'casual','registered','count'], axis=1)  
#print(x.shape)  # (10886, 8)

y = train['count']
#print(y.shape)  # (10886,)

test_file = test_file.drop(['datetime'], axis=1)  

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.pipeline import make_pipeline, Pipeline

#2.모델
# model = SVC()
model = make_pipeline(MinMaxScaler(), RandomForestRegressor())

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)    # score는 자동으로 맞춰서 반환해줌; 여기서 반환해주는건 'accuracy' (분류모델이기 때문에)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2_score = r2_score(y_test, y_predict)

print("model.score : ", result)

'''
model.score :  0.25560362416422955
'''