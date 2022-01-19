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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape, y_train.shape)  # (120, 4) (120, 3)
# print(x_test.shape, y_test.shape)  # (30, 4) (30, 3)


#2. 모델구성   -> feature importance는 트리계열에만 있음
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#2.모델
model = DecisionTreeClassifier(max_depth=5, random_state=66)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)    # score는 자동으로 맞춰서 반환해줌; 여기서 반환해주는건 'accuracy' (분류모델이기 때문에)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("accuracy_score : ", acc)

print(model.feature_importances_)

'''
accuracy_score :  0.9666666666666667
[0.         0.0125026  0.53835801 0.44913938]  -> 합이 1
'''