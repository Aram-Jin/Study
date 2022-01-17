import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier   # Classifier :분류모델
from sklearn.linear_model import LogisticRegression  # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = load_breast_cancer()
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
ACC :  [0.89010989 0.97802198 0.69230769 0.84615385 0.93406593] 
 cross_val_score :  0.868131868131868 4

LinearSVC
ACC :  [0.93406593 0.97802198 0.91208791 0.87912088 0.86813187]
 cross_val_score :  0.9142857142857143 4
 
SVC
ACC :  [0.87912088 0.97802198 0.92307692 0.89010989 0.9010989 ] 
 cross_val_score :  0.9142857142857143 4
 
KNeighborsClassifier
ACC :  [0.91208791 0.95604396 0.95604396 0.9010989  0.93406593] 
 cross_val_score :  0.9318681318681319 4
 
LogisticRegression
ACC :  [0.94505495 0.94505495 0.94505495 0.87912088 0.95604396]
 cross_val_score :  0.934065934065934 4
 
DecisionTreeClassifier
ACC :  [0.86813187 0.96703297 0.93406593 0.92307692 0.93406593] 
 cross_val_score :  0.9252747252747253 4
 
RandomForestClassifier
ACC :  [0.93406593 0.98901099 0.94505495 0.96703297 0.96703297] 
 cross_val_score :  0.9604395604395604 4
''' 