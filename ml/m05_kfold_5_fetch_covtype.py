import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_covtype

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier   # Classifier :분류모델
from sklearn.linear_model import LogisticRegression  # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = fetch_covtype()
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
print("ACC : ", scores, "\n cross_val_score : ", np.mean(scores),4)

'''
Perceptron
ACC :  [0.51103677 0.57477249 0.31403154 0.4990964  0.41989652] 
 cross_val_score :  0.4637667431778915 4
 
LinearSVC
ACC :  [0.4229255  0.42143026 0.55838945 0.49343818 0.55726595] 
 cross_val_score :  0.4906898673994755 4
 
SVC


'''