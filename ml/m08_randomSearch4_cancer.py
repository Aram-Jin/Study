import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier   # Classifier :분류모델
from sklearn.linear_model import LogisticRegression  # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

# model : RandomForestClassifier

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [8, 10], 'min_samples_leaf' :[5, 7, 10], 'min_samples_split' : [2, 5, 10]},
    {'n_estimators' : [100], 'max_depth' : [6, 12], 'min_samples_leaf' :[7, 10], 'min_samples_split' : [2, 3]},
    {'n_estimators' : [200], 'max_depth' : [10, 12], 'min_samples_leaf' :[3, 5], 'min_samples_split' : [5, 10]},
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8], 'min_samples_leaf' :[3, 10], 'min_samples_split' : [2, 10]},
    {'n_jobs' : [-1, 2, 4]}
]                                                              

#2. 모델구성
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)   # -> 너무 중요해~~!!!!  *n_jobs: 속도에만 영향을 미칠뿐 성능향상에는 효과없음(data많을때만 사용하는게 좋음. '-1'은 모든 코어를 써주겠다는것)
# model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1, random_state=66, n_iter=20)   # 20 * 5 = 100 . n_iter는 랜덤으로 연산해 줄 갯수를 지정해줌
# model = SVC(C=1, kernel='linear', degree=3)

#3. 훈련
import time 
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측

# x_test = x_train  # 과적합 상황 보여주기
# y_test = y_train  # train데이터로 best_estimator_로 예측 뒤 점수를 내면 best_score_가 나온다

print("최적의 매개변수 : ", model.best_estimator_)  
print("최적의 파라미터 : ", model.best_params_)    

print("best_score_ : ", model.best_score_)            
print("model.score : ", model.score(x_test, y_test))  

y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))   # test까지 넣어서 나온 값 중 최고값(val_acc).  iris는 분류모델이므로 accuracy 값


y_pred_best = model.best_estimator_.predict(x_test)    # gridsearch 사용할떄 model.predict보다는 model.best_estimator_.predict 사용하길 권장함
print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))

print("걸린시간 : ", end - start)

'''
<RandomizedSearchCV>
Fitting 5 folds for each of 20 candidates, totalling 100 fits
최적의 매개변수 :  RandomForestClassifier(max_depth=8, min_samples_leaf=5, min_samples_split=5,
                       n_estimators=200)
최적의 파라미터 :  {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_depth': 8}
best_score_ :  0.9604395604395604
model.score :  0.9649122807017544
accuracy_score :  0.9649122807017544
최적 튠 ACC :  0.9649122807017544
걸린시간 :  3.772878408432007

<GridSearchCV>
Fitting 5 folds for each of 71 candidates, totalling 355 fits
최적의 매개변수 :  RandomForestClassifier(n_jobs=-1)
최적의 파라미터 :  {'n_jobs': -1}
best_score_ :  0.9648351648351647
model.score :  0.9736842105263158
accuracy_score :  0.9736842105263158
최적 튠 ACC :  0.9736842105263158
걸린시간 :  9.619811296463013
'''