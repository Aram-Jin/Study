import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     # Classifier :분류모델
from sklearn.linear_model import LogisticRegression,LinearRegression        # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score

datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# model = Perceptron()
# model = LinearSVC()
model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = LogisticRegression()
# model = LinearRegression()
# model = DecisionTreeClassifier()
# model = DecisionTreeRegressor()
# model = RandomForestClassifier()
# model = RandomForestRegressor()

scores = cross_val_score(model, x_train, y_train, cv=kfold)
# print(scores)
print("r2 : ", scores, "\n cross_val_score : ", np.mean(scores),4)

'''
Perceptron
r2 :  [0.         0.01408451 0.         0.01428571 0.02857143] 
 cross_val_score :  0.011388329979879274 4

LinearSVC
r2 :  [0.         0.         0.         0.01428571 0.        ] 
 cross_val_score :  0.002857142857142857 4
 
SVC
r2 :  [0.         0.         0.         0.01428571 0.        ] 
 cross_val_score :  0.002857142857142857 4 

KNeighborsRegressor
r2 :  [0.37000683 0.35477108 0.32086338 0.51614896 0.41040527] 
 cross_val_score :  0.3944391027499309 4
 
LinearRegression
r2 :  [0.53550031 0.49362737 0.47105167 0.55090349 0.36810479] 
 cross_val_score :  0.4838375278967237 4
 
DecisionTreeRegressor 
r2 :  [-0.02201327  0.14021662  0.01696205  0.11680283 -0.09463731] 
 cross_val_score :  0.03146618587795071 4
 
RandomForestRegressor
r2 :  [0.47740851 0.55367133 0.39426981 0.56894184 0.41377125] 
 cross_val_score :  0.4816125489968341 4
'''