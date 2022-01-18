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

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

n_splits = 3
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {"C":[1, 10, 100, 1000, 10000], "kernel":["linear", "sigmoid"], "degree":[3,4,5,6]},   # 40
    {"C":[1, 10, 100], "kernel":["rbf", "linear"], "gamma":[0.001, 0.0001], "degree":[5, 10]},     # 24
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"],                     
     "gamma":[0.01, 0.001, 0.0001], "degree":[0.1, 3, 4]}                   # 36
]                                                                 # 총 100개

#2. 모델구성
# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)   # -> 너무 중요해~~!!!!  *n_jobs: 속도에만 영향을 미칠뿐 성능향상에는 효과없음(data많을때만 사용하는게 좋음. '-1'은 모든 코어를 써주겠다는것)
model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1, random_state=66)
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
'''
################################################################################################################
# print(model.cv_results_)  #42번 훈련시킨 결과

# aaa = pd.DataFrame(model.cv_results_)
# print(aaa)

# bbb = aaa[['params', 'mean_test_score', 'rank_test_score','split0_test_score']] 
#         #    'split0_test_score', 'split1_test_score', 'split2_test_score', 
#         #    'split3_test_score', 'split4_test_score']]

# print(bbb)

'''
                                               params  mean_test_score  rank_test_score  split0_test_score
0           {'C': 1, 'degree': 3, 'kernel': 'linear'}         0.991667                1           1.000000
1           {'C': 1, 'degree': 4, 'kernel': 'linear'}         0.991667                1           1.000000
2           {'C': 1, 'degree': 5, 'kernel': 'linear'}         0.991667                1           1.000000
3          {'C': 10, 'degree': 3, 'kernel': 'linear'}         0.950000               14           0.916667
4          {'C': 10, 'degree': 4, 'kernel': 'linear'}         0.950000               14           0.916667
5          {'C': 10, 'degree': 5, 'kernel': 'linear'}         0.950000               14           0.916667
6         {'C': 100, 'degree': 3, 'kernel': 'linear'}         0.950000               14           0.916667
7         {'C': 100, 'degree': 4, 'kernel': 'linear'}         0.950000               14           0.916667
8         {'C': 100, 'degree': 5, 'kernel': 'linear'}         0.950000               14           0.916667
9        {'C': 1000, 'degree': 3, 'kernel': 'linear'}         0.958333                7           0.916667
10       {'C': 1000, 'degree': 4, 'kernel': 'linear'}         0.958333                7           0.916667
11       {'C': 1000, 'degree': 5, 'kernel': 'linear'}         0.958333                7           0.916667
12          {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}         0.475000               34           0.250000
13         {'C': 1, 'gamma': 0.0001, 'kernel': 'rbf'}         0.316667               36           0.250000
14         {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}         0.908333               22           0.958333
15        {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}         0.475000               34           0.250000
16        {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}         0.966667                6           0.958333
17       {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}         0.908333               22           0.958333
18  {'C': 1, 'degree': 3, 'gamma': 0.01, 'kernel':...         0.616667               32           0.583333
19  {'C': 1, 'degree': 3, 'gamma': 0.001, 'kernel'...         0.316667               36           0.250000
20  {'C': 1, 'degree': 3, 'gamma': 0.0001, 'kernel...         0.316667               36           0.250000
21  {'C': 1, 'degree': 4, 'gamma': 0.01, 'kernel':...         0.616667               32           0.583333
22  {'C': 1, 'degree': 4, 'gamma': 0.001, 'kernel'...         0.316667               36           0.250000
23  {'C': 1, 'degree': 4, 'gamma': 0.0001, 'kernel...         0.316667               36           0.250000
24  {'C': 10, 'degree': 3, 'gamma': 0.01, 'kernel'...         0.916667               20           0.916667
25  {'C': 10, 'degree': 3, 'gamma': 0.001, 'kernel...         0.883333               24           0.833333
26  {'C': 10, 'degree': 3, 'gamma': 0.0001, 'kerne...         0.316667               36           0.250000
27  {'C': 10, 'degree': 4, 'gamma': 0.01, 'kernel'...         0.916667               20           0.916667
28  {'C': 10, 'degree': 4, 'gamma': 0.001, 'kernel...         0.883333               24           0.833333
29  {'C': 10, 'degree': 4, 'gamma': 0.0001, 'kerne...         0.316667               36           0.250000
30  {'C': 100, 'degree': 3, 'gamma': 0.01, 'kernel...         0.883333               24           0.791667
31  {'C': 100, 'degree': 3, 'gamma': 0.001, 'kerne...         0.958333                7           0.958333
32  {'C': 100, 'degree': 3, 'gamma': 0.0001, 'kern...         0.883333               24           0.833333
33  {'C': 100, 'degree': 4, 'gamma': 0.01, 'kernel...         0.883333               24           0.791667
34  {'C': 100, 'degree': 4, 'gamma': 0.001, 'kerne...         0.958333                7           0.958333
35  {'C': 100, 'degree': 4, 'gamma': 0.0001, 'kern...         0.883333               24           0.833333
36  {'C': 1000, 'degree': 3, 'gamma': 0.01, 'kerne...         0.841667               30           0.750000
37  {'C': 1000, 'degree': 3, 'gamma': 0.001, 'kern...         0.991667                1           1.000000
38  {'C': 1000, 'degree': 3, 'gamma': 0.0001, 'ker...         0.958333                7           0.958333
39  {'C': 1000, 'degree': 4, 'gamma': 0.01, 'kerne...         0.841667               30           0.750000
40  {'C': 1000, 'degree': 4, 'gamma': 0.001, 'kern...         0.991667                1           1.000000
41  {'C': 1000, 'degree': 4, 'gamma': 0.0001, 'ker...         0.958333                7           0.958333
'''