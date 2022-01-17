import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier   # Classifier :분류모델
from sklearn.linear_model import LogisticRegression  # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = load_wine()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# model = Perceptron()
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = LogisticRegression()
# model = LinearRegression()
# model = DecisionTreeClassifier()
# model = DecisionTreeRegressor()
model = RandomForestClassifier()
# model = RandomForestRegressor()

scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("ACC : ", scores, "\n cross_val_score : ", np.mean(scores),4)

'''
Perceptron
ACC :  [0.55172414 0.62068966 0.53571429 0.35714286 0.60714286] 
 cross_val_score :  0.5344827586206897 4

LinearSVC
ACC :  [0.65517241 0.86206897 0.85714286 0.92857143 0.75      ]
cross_val_score :  0.8105911330049261 4

SVC
ACC :  [0.5862069  0.65517241 0.5        0.67857143 0.67857143] 
 cross_val_score :  0.619704433497537 4

KNeighborsClassifier 
ACC :  [0.65517241 0.79310345 0.57142857 0.71428571 0.5       ] 
 cross_val_score :  0.6467980295566502 4 

LogisticRegression 
ACC :  [0.89655172 1.         0.82142857 0.89285714 0.96428571]
 cross_val_score :  0.9150246305418719 4
 
DecisionTreeClassifier
ACC :  [0.93103448 0.86206897 0.89285714 0.89285714 0.89285714] 
 cross_val_score :  0.8943349753694582 
 
RandomForestClassifier
ACC :  [0.96551724 1.         0.96428571 0.89285714 0.96428571] 
 cross_val_score :  0.9573891625615764 4
'''