import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier   # Classifier :분류모델
from sklearn.linear_model import LogisticRegression  # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

model = SVC()

scores = cross_val_score(model, x, y, cv=kfold)

print("ACC : ", scores, "\n cross_val_score : ", np.mean(scores),4)
      
    #    5번 cross_val했던값, 값들의 평균

