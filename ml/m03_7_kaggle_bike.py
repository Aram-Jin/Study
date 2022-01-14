from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     # Classifier :분류모델
from sklearn.linear_model import LogisticRegression,LinearRegression        # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

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

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#2.모델
# model = Perceptron()
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = LogisticRegression()
# model = LinearRegression()
# model = DecisionTreeClassifier()
model = DecisionTreeRegressor()
# model = RandomForestClassifier()
# model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)    # score는 자동으로 맞춰서 반환해줌; 여기서 반환해주는건 'accuracy' (분류모델이기 때문에)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# print("Perceptron : ", result)
# print("LinearSVC : ", result)
# print("SVC : ", result)
# print("KNeighborsClassifier : ", result)
# print("KNeighborsRegressor : ", result)
# print("LogisticRegression : ", result)
# print("LinearRegression : ", result)
# print("DecisionTreeClassifier : ", result)
print("DecisionTreeRegressor : ", result)
# print("RandomForestClassifier : ", result)
# print("RandomForestRegressor : ", result)
 
'''
# Perceptron :  0.0009182736455463728
# LinearSVC :  0.002295684113865932
# SVC :  0.018365472910927456   
KNeighborsRegressor :  0.1867313357142426
LinearRegression :  0.2494896826312223
DecisionTreeRegressor :  -0.21438741191357558
RandomForestRegressor :  0.26241193664149065
'''
