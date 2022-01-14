import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (150, 4) (150,)
#print(y)
#print(np.unique(y))  # [0 1 2]

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)  # (150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape, y_train.shape)  # (120, 4) (120, 3)
# print(x_test.shape, y_test.shape)  # (30, 4) (30, 3)


#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier   # Classifier :분류모델
from sklearn.linear_model import LogisticRegression, Perceptron  # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#2.모델
# model = Perceptron()
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)    # score는 자동으로 맞춰서 반환해줌; 여기서 반환해주는건 'accuracy' (분류모델이기 때문에)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

# print("Perceptron : ", result)
# print("LinearSVC : ", result)
# print("SVC : ", result)
# print("KNeighborsClassifier : ", result)
# print("LogisticRegression : ", result)
# print("DecisionTreeClassifier : ", result)
print("RandomForestClassifier : ", result)


# print("accuracy_score : ", acc)

'''
Perceptron :  0.9333333333333333
LinearSVC :  0.9666666666666667
SVC :  0.9666666666666667
KNeighborsClassifier :  0.9666666666666667
LogisticRegression :  1.0
DecisionTreeClassifier :  0.9333333333333333
RandomForestClassifier :  0.9333333333333333
'''
