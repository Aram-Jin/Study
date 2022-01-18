import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier   # Classifier :분류모델
from sklearn.linear_model import LogisticRegression  # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {"C":[1, 10, 100, 1000, 10000], "kernel":["linear", "sigmoid"], "degree":[3,4,5,6]},   # 40
    {"C":[1, 10, 100], "kernel":["rbf", "linear"], "gamma":[0.001, 0.0001], "degree":[5, 10]},   # 24
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"],                     
     "gamma":[0.01, 0.001, 0.0001], "degree":[0.1, 3, 4]}                   # 36
]                                                                 # 총 100개

#2. 모델구성
# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)   # -> 너무 중요해~~!!!!  *n_jobs: 속도에만 영향을 미칠뿐 성능향상에는 효과없음(data많을때만 사용하는게 좋음. '-1'은 모든 코어를 써주겠다는것)
# model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1, random_state=66, n_iter=20)   # 20 * 5 = 100 . n_iter는 랜덤으로 연산해 줄 갯수를 지정해줌
model = HalvingGridSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)   # 아직 완성 된 건 아니지만, GridSearch보다 연산량은 많지만(모든 파라미터를 돌리고 더 돌림) 데이터의 일부만 돌리기 때문에 속도가 훨씬 빠르다. 일부만 쓰면서 상위 몇퍼만 빼서 완전히 돌린다.
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
Fitting 5 folds for each of 42 candidates, totalling 210 fits
<GridSearchCV>
최적의 매개변수 :  SVC(C=1, kernel='linear')
최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
best_score_ :  0.9916666666666668
model.score :  0.9666666666666667
accuracy_score :  0.9666666666666667
최적 튠 ACC :  0.9666666666666667
걸린시간 :  1.1993284225463867


파라미터 튜닝 후 <RandomizedSearchCV>
Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 매개변수 :  SVC(C=1, degree=10, gamma=0.0001, kernel='linear')
최적의 파라미터 :  {'kernel': 'linear', 'gamma': 0.0001, 'degree': 10, 'C': 1}
best_score_ :  0.9916666666666668
model.score :  0.9666666666666667
accuracy_score :  0.9666666666666667
최적 튠 ACC :  0.9666666666666667
걸린시간 :  1.0255060195922852

<HalvingGridSearchCV>
n_iterations: 2
n_required_iterations: 5
n_possible_iterations: 2
min_resources_: 30
max_resources_: 120
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 100
n_resources: 30
Fitting 5 folds for each of 100 candidates, totalling 500 fits
----------
iter: 1
n_candidates: 34
n_resources: 90
Fitting 5 folds for each of 34 candidates, totalling 170 fits
최적의 매개변수 :  SVC(C=1, degree=4, kernel='linear')
최적의 파라미터 :  {'C': 1, 'degree': 4, 'kernel': 'linear'}
best_score_ :  0.9666666666666668
model.score :  0.9666666666666667
accuracy_score :  0.9666666666666667
최적 튠 ACC :  0.9666666666666667
걸린시간 :  1.2913944721221924

'''
################################################################################################################
# print(model.cv_results_)  #42번 훈련시킨 결과

# aaa = pd.DataFrame(model.cv_results_)
# print(aaa)

# bbb = aaa[['params', 'mean_test_score', 'rank_test_score','split0_test_score']] 
#         #    'split0_test_score', 'split1_test_score', 'split2_test_score', 
#         #    'split3_test_score', 'split4_test_score']]

# print(bbb)
