import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     # Classifier :분류모델
from sklearn.linear_model import LogisticRegression,LinearRegression        # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score

datasets = load_boston()
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
KNeighborsRegressor
r2 :  [0.38689566 0.52994483 0.3434155  0.55325748 0.51995804]
 cross_val_score :  0.4666943025010436 4

LinearRegression
r2 :  [0.5815212  0.69885237 0.6537276  0.77449543 0.70223459]
 cross_val_score :  0.6821662390321913 4

DecisionTreeRegressor 
r2 :  [0.78712231 0.56220835 0.63261746 0.73552398 0.76800711] 
 cross_val_score :  0.6970958432513239 4
 
RandomForestRegressor 
r2 :  [0.87186954 0.73244108 0.79430554 0.86380272 0.89574326] 
 cross_val_score :  0.8316324288776276 4 
'''