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

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3,4,5]},   # 12
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},     # 6
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"],                     
     "gamma":[0.01, 0.001, 0.0001], "degree":[3, 4]}                   # 24
]                                                                 # 총 42개

#2. 모델구성
model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True)
# model = SVC(C=1, kernel='linear', degree=3)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측

# x_test = x_train  # 과적합 상황 보여주기
# y_test = y_train  # train데이터로 best_estimator_로 예측 뒤 점수를 내면 best_score_가 나온다

print("최적의 매개변수 : ", model.best_estimator_)   # 최적의 매개변수 :  SVC(C=1, kernel='linear')
print("최적의 파라미터 : ", model.best_params_)      # 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

print("best_score_ : ", model.best_score_)            # best_score_ :  0.9916666666666668  -> train부분에서 훈련시킨 값 중 최고 값(acc)
print("model.score : ", model.score(x_test, y_test))  # model.score :  0.9666666666666667  -> test까지 넣어서 나온 값 중 최고값(val_acc).  iris는 분류모델이므로 accuracy 값

y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))   # accuracy_score :  0.9666666666666667  -> test까지 넣어서 나온 값 중 최고값(val_acc).  iris는 분류모델이므로 accuracy 값


y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))

################################################################################################################
# print(model.cv_results_)  42번 훈련시킨 결과
'''
{'mean_fit_time': array([0.00059867, 0.00099716, 0.00019937, 0.00039306, 0.00040436,
       0.00019941, 0.00099745, 0.        , 0.00039873, 0.00099621,
       0.00059838, 0.0003984 , 0.00021529, 0.00228105, 0.00059862,
       0.00079842, 0.00059838, 0.00079784, 0.00059772, 0.00081148,
       0.00079789, 0.00059838, 0.00079761, 0.00059862, 0.        ,
       0.00079794, 0.00059843, 0.        , 0.0008141 , 0.        ,
       0.        , 0.        , 0.00199337, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.0019989 ]), 'std_fit_time': array([4.89050801e-04, 9.48893964e-07, 3.98731232e-04, 4.81495694e-04,
       4.95320689e-04, 3.98826599e-04, 1.38693233e-06, 0.00000000e+00,
       4.88347383e-04, 1.74665653e-05, 4.88584427e-04, 4.87935281e-04,
       4.30583954e-04, 4.06847287e-03, 4.88775658e-04, 3.99209208e-04,
       4.88579890e-04, 3.98932026e-04, 4.88036903e-04, 4.06618832e-04,
       3.99345885e-04, 4.88577656e-04, 3.98809923e-04, 4.88778030e-04,
       0.00000000e+00, 3.98970006e-04, 4.88616597e-04, 0.00000000e+00,
       4.08281764e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       3.98674011e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 3.99780273e-03]), 'mean_score_time': array([0.00020466, 0.        , 0.00059843, 0.        , 0.00059834,
       0.        , 0.        , 0.00039878, 0.        , 0.00020056,
       0.00019956, 0.00019956, 0.        , 0.00020542, 0.00039902,
       0.00039873, 0.00039854, 0.00019932, 0.0003993 , 0.00019989,
       0.00019898, 0.00019951, 0.00020008, 0.00039859, 0.00099735,
       0.        , 0.00039883, 0.0009974 , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.00199895,
       0.        , 0.0019989 , 0.        , 0.        , 0.        ,
       0.        , 0.        ]), 'std_score_time': array([4.09317017e-04, 0.00000000e+00, 4.88954528e-04, 0.00000000e+00,
       4.88541523e-04, 0.00000000e+00, 0.00000000e+00, 4.88402484e-04,
       0.00000000e+00, 4.01115417e-04, 3.99112701e-04, 3.99112701e-04,
       0.00000000e+00, 4.10842896e-04, 4.88696760e-04, 4.88345917e-04,
       4.88110639e-04, 3.98635864e-04, 4.89047649e-04, 3.99780273e-04,
       3.97968292e-04, 3.99017334e-04, 4.00161743e-04, 4.88169534e-04,
       1.76107267e-06, 0.00000000e+00, 4.88460832e-04, 8.34124359e-07,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 3.99789810e-03, 0.00000000e+00,
       3.99780273e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00]), 'param_C': masked_array(data=[1, 1, 1, 10, 10, 10, 100, 100, 100, 1000, 1000, 1000,
                   1, 1, 10, 10, 100, 100, 1, 1, 1, 1, 1, 1, 10, 10, 10,
                   10, 10, 10, 100, 100, 100, 100, 100, 100, 1000, 1000,
                   1000, 1000, 1000, 1000],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_degree': masked_array(data=[3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, --, --, --, --, --,
                   --, 3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4,
                   4, 3, 3, 3, 4, 4, 4],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False,  True,  True,  True,  True,
                    True,  True, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_kernel': masked_array(data=['linear', 'linear', 'linear', 'linear', 'linear',
                   'linear', 'linear', 'linear', 'linear', 'linear',
                   'linear', 'linear', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
                   'rbf', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid',
                   'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid',
                   'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid',
                   'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid',
                   'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_gamma': masked_array(data=[--, --, --, --, --, --, --, --, --, --, --, --, 0.001,
                   0.0001, 0.001, 0.0001, 0.001, 0.0001, 0.01, 0.001,
                   0.0001, 0.01, 0.001, 0.0001, 0.01, 0.001, 0.0001, 0.01,
                   0.001, 0.0001, 0.01, 0.001, 0.0001, 0.01, 0.001,
                   0.0001, 0.01, 0.001, 0.0001, 0.01, 0.001, 0.0001],
             mask=[ True,  True,  True,  True,  True,  True,  True,  True,
                    True,  True,  True,  True, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'params': [{'C': 1, 'degree': 3, 'kernel': 'linear'}, {'C': 1, 'degree': 4, 'kernel': 'linear'}, {'C': 1, 'degree': 5, 'kernel': 'linear'}, {'C': 10, 'degree': 3, 'kernel': 'linear'}, {'C': 10, 'degree': 4, 'kernel': 'linear'}, {'C': 10, 'degree': 5, 'kernel': 'linear'}, {'C': 100, 'degree': 3, 'kernel': 'linear'}, {'C': 100, 'degree': 4, 'kernel': 'linear'}, {'C': 100, 'degree': 5, 'kernel': 'linear'}, {'C': 1000, 'degree': 3, 'kernel': 'linear'}, {'C': 1000, 'degree': 4, 'kernel': 'linear'}, {'C': 1000, 'degree': 5, 'kernel': 'linear'}, {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}, {'C': 1, 'gamma': 0.0001, 'kernel': 'rbf'}, {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}, {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}, {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}, {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}, {'C': 1, 'degree': 3, 'gamma': 0.01, 'kernel': 'sigmoid'}, {'C': 1, 'degree': 3, 'gamma': 0.001, 'kernel': 'sigmoid'}, {'C': 1, 'degree': 3, 'gamma': 0.0001, 'kernel': 'sigmoid'}, {'C': 1, 'degree': 4, 'gamma': 0.01, 'kernel': 'sigmoid'}, {'C': 1, 'degree': 4, 'gamma': 0.001, 'kernel': 'sigmoid'}, {'C': 1, 'degree': 4, 'gamma': 0.0001, 'kernel': 'sigmoid'}, {'C': 10, 'degree': 3, 'gamma': 0.01, 'kernel': 'sigmoid'}, {'C': 10, 'degree': 3, 'gamma': 0.001, 'kernel': 'sigmoid'}, {'C': 10, 'degree': 3, 'gamma': 0.0001, 'kernel': 'sigmoid'}, {'C': 10, 'degree': 4, 'gamma': 0.01, 'kernel': 'sigmoid'}, {'C': 10, 'degree': 4, 'gamma': 0.001, 'kernel': 'sigmoid'}, {'C': 10, 'degree': 4, 'gamma': 0.0001, 'kernel': 'sigmoid'}, {'C': 100, 'degree': 3, 'gamma': 0.01, 'kernel': 'sigmoid'}, {'C': 
100, 'degree': 3, 'gamma': 0.001, 'kernel': 'sigmoid'}, {'C': 100, 'degree': 3, 'gamma': 0.0001, 'kernel': 'sigmoid'}, {'C': 100, 'degree': 4, 'gamma': 0.01, 'kernel': 'sigmoid'}, {'C': 100, 'degree': 4, 
'gamma': 0.001, 'kernel': 'sigmoid'}, {'C': 100, 'degree': 4, 'gamma': 0.0001, 'kernel': 'sigmoid'}, {'C': 1000, 'degree': 3, 'gamma': 0.01, 'kernel': 'sigmoid'}, {'C': 1000, 'degree': 3, 'gamma': 0.001, 
'kernel': 'sigmoid'}, {'C': 1000, 'degree': 3, 'gamma': 0.0001, 'kernel': 'sigmoid'}, {'C': 1000, 'degree': 4, 'gamma': 0.01, 'kernel': 'sigmoid'}, {'C': 1000, 'degree': 4, 'gamma': 0.001, 'kernel': 'sigmoid'}, {'C': 1000, 'degree': 4, 'gamma': 0.0001, 'kernel': 'sigmoid'}], 'split0_test_score': array([1.        , 1.        , 1.        , 0.91666667, 0.91666667,
       0.91666667, 0.91666667, 0.91666667, 0.91666667, 0.91666667,
       0.91666667, 0.91666667, 0.25      , 0.25      , 0.95833333,
       0.25      , 0.95833333, 0.95833333, 0.58333333, 0.25      ,
       0.25      , 0.58333333, 0.25      , 0.25      , 0.91666667,
       0.83333333, 0.25      , 0.91666667, 0.83333333, 0.25      ,
       0.79166667, 0.95833333, 0.83333333, 0.79166667, 0.95833333,
       0.83333333, 0.75      , 1.        , 0.95833333, 0.75      ,
       1.        , 0.95833333]), 'split1_test_score': array([1.        , 1.        , 1.        , 0.91666667, 0.91666667,
       0.91666667, 0.95833333, 0.95833333, 0.95833333, 0.95833333,
       0.95833333, 0.95833333, 0.66666667, 0.25      , 0.95833333,
       0.66666667, 1.        , 0.95833333, 0.66666667, 0.25      ,
       0.25      , 0.66666667, 0.25      , 0.25      , 0.91666667,
       0.95833333, 0.25      , 0.91666667, 0.95833333, 0.25      ,
       0.95833333, 1.        , 0.95833333, 0.95833333, 1.        ,
       0.95833333, 0.875     , 1.        , 1.        , 0.875     ,
       1.        , 1.        ]), 'split2_test_score': array([0.95833333, 0.95833333, 0.95833333, 0.95833333, 0.95833333,
       0.95833333, 0.95833333, 0.95833333, 0.95833333, 0.95833333,
       0.95833333, 0.95833333, 0.58333333, 0.20833333, 0.875     ,
       0.58333333, 0.95833333, 0.875     , 0.58333333, 0.20833333,
       0.20833333, 0.58333333, 0.20833333, 0.20833333, 0.79166667,
       0.875     , 0.20833333, 0.79166667, 0.875     , 0.20833333,
       0.83333333, 0.95833333, 0.875     , 0.83333333, 0.95833333,
       0.875     , 0.79166667, 0.95833333, 0.95833333, 0.79166667,
       0.95833333, 0.95833333]), 'split3_test_score': array([1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 0.25      , 0.25      , 0.95833333,
       0.25      , 1.        , 0.95833333, 0.625     , 0.25      ,
       0.25      , 0.625     , 0.25      , 0.25      , 1.        ,
       0.95833333, 0.25      , 1.        , 0.95833333, 0.25      ,
       0.91666667, 1.        , 0.95833333, 0.91666667, 1.        ,
       0.95833333, 0.875     , 1.        , 1.        , 0.875     ,
       1.        , 1.        ]), 'split4_test_score': array([1.        , 1.        , 1.        , 0.95833333, 0.95833333,
       0.95833333, 0.91666667, 0.91666667, 0.91666667, 0.95833333,
       0.95833333, 0.95833333, 0.625     , 0.625     , 0.79166667,
       0.625     , 0.91666667, 0.79166667, 0.625     , 0.625     ,
       0.625     , 0.625     , 0.625     , 0.625     , 0.95833333,
       0.79166667, 0.625     , 0.95833333, 0.79166667, 0.625     ,
       0.91666667, 0.875     , 0.79166667, 0.91666667, 0.875     ,
       0.79166667, 0.91666667, 1.        , 0.875     , 0.91666667,
       1.        , 0.875     ]), 'mean_test_score': array([0.99166667, 0.99166667, 0.99166667, 0.95      , 0.95      ,
       0.95      , 0.95      , 0.95      , 0.95      , 0.95833333,
       0.95833333, 0.95833333, 0.475     , 0.31666667, 0.90833333,
       0.475     , 0.96666667, 0.90833333, 0.61666667, 0.31666667,
       0.31666667, 0.61666667, 0.31666667, 0.31666667, 0.91666667,
       0.88333333, 0.31666667, 0.91666667, 0.88333333, 0.31666667,
       0.88333333, 0.95833333, 0.88333333, 0.88333333, 0.95833333,
       0.88333333, 0.84166667, 0.99166667, 0.95833333, 0.84166667,
       0.99166667, 0.95833333]), 'std_test_score': array([0.01666667, 0.01666667, 0.01666667, 0.03118048, 0.03118048,
       0.03118048, 0.03118048, 0.03118048, 0.03118048, 0.02635231,
       0.02635231, 0.02635231, 0.18559215, 0.15500896, 0.06666667,
       0.18559215, 0.03118048, 0.06666667, 0.03118048, 0.15500896,
       0.15500896, 0.03118048, 0.15500896, 0.15500896, 0.06972167,
       0.06666667, 0.15500896, 0.06972167, 0.06666667, 0.15500896,
       0.06123724, 0.04564355, 0.06666667, 0.06123724, 0.04564355,
       0.06666667, 0.06123724, 0.01666667, 0.04564355, 0.06123724,
       0.01666667, 0.04564355]), 'rank_test_score': array([ 1,  1,  1, 14, 14, 14, 14, 14, 14,  7,  7,  7, 34, 36, 22, 34,  6,
       22, 32, 36, 36, 32, 36, 36, 20, 24, 36, 20, 24, 36, 24,  7, 24, 24,
        7, 24, 30,  1,  7, 30,  1,  7])}
'''

aaa = pd.DataFrame(model.cv_results_)
# print(aaa)
'''
    mean_fit_time  std_fit_time  mean_score_time  std_score_time param_C param_degree  ... split2_test_score split3_test_score split4_test_score  mean_test_score  std_test_score  rank_test_score
0        0.001003  1.141584e-05         0.000000        0.000000       1            3  ...          0.958333          1.000000          1.000000         0.991667        0.016667                1
1        0.000199  3.987312e-04         0.000399        0.000489       1            4  ...          0.958333          1.000000          1.000000         0.991667        0.016667                1
2        0.000630  4.617624e-04         0.000000        0.000000       1            5  ...          0.958333          1.000000          1.000000         0.991667        0.016667                1
3        0.000793  3.966142e-04         0.000200        0.000399      10            3  ...          0.958333          1.000000          0.958333         0.950000        0.031180               14
4        0.000798  3.993984e-04         0.000200        0.000399      10            4  ...          0.958333          1.000000          0.958333         0.950000        0.031180               14
5        0.000000  0.000000e+00         0.000803        0.000402      10            5  ...          0.958333          1.000000          0.958333         0.950000        0.031180               14
6        0.000600  4.899081e-04         0.000403        0.000494     100            3  ...          0.958333          1.000000          0.916667         0.950000        0.031180               14
7        0.000598  4.884220e-04         0.000394        0.000483     100            4  ...          0.958333          1.000000          0.916667         0.950000        0.031180               14
8        0.000598  4.882632e-04         0.000398        0.000488     100            5  ...          0.958333          1.000000          0.916667         0.950000        0.031180               14
9        0.000798  3.988986e-04         0.000000        0.000000    1000            3  ...          0.958333          1.000000          0.958333         0.958333        0.026352                7
10       0.001197  4.123125e-04         0.000000        0.000000    1000            4  ...          0.958333          1.000000          0.958333         0.958333        0.026352                7
11       0.000595  4.859547e-04         0.000404        0.000495    1000            5  ...          0.958333          1.000000          0.958333         0.958333        0.026352                7
12       0.000604  4.932794e-04         0.000604        0.000493       1          NaN  ...          0.583333          0.250000          0.625000         0.475000        0.185592               34
13       0.000803  4.015836e-04         0.000399        0.000489       1          NaN  ...          0.208333          0.250000          0.625000         0.316667        0.155009               36
14       0.000598  4.885001e-04         0.000395        0.000484      10          NaN  ...          0.875000          0.958333          0.791667         0.908333        0.066667               22
15       0.001005  1.162310e-05         0.000398        0.000488      10          NaN  ...          0.583333          0.250000          0.625000         0.475000        0.185592               34
16       0.000798  3.989354e-04         0.000197        0.000393     100          NaN  ...          0.958333          1.000000          0.916667         0.966667        0.031180                6
17       0.000599  4.890866e-04         0.000393        0.000482     100          NaN  ...          0.875000          0.958333          0.791667         0.908333        0.066667               22
18       0.000993  9.287192e-06         0.000000        0.000000       1            3  ...          0.583333          0.625000          0.625000         0.616667        0.031180               32
19       0.000803  4.018651e-04         0.000393        0.000482       1            3  ...          0.208333          0.250000          0.625000         0.316667        0.155009               36
20       0.000592  4.838546e-04         0.000396        0.000485       1            3  ...          0.208333          0.250000          0.625000         0.316667        0.155009               36
21       0.000807  4.034698e-04         0.000393        0.000482       1            4  ...          0.583333          0.625000          0.625000         0.616667        0.031180               32
22       0.000799  3.996733e-04         0.000199        0.000398       1            4  ...          0.208333          0.250000          0.625000         0.316667        0.155009               36
23       0.000805  4.026438e-04         0.000194        0.000388       1            4  ...          0.208333          0.250000          0.625000         0.316667        0.155009               36
24       0.000000  0.000000e+00         0.000798        0.000399      10            3  ...          0.791667          1.000000          0.958333         0.916667        0.069722               20
25       0.000997  3.349420e-06         0.000000        0.000000      10            3  ...          0.875000          0.958333          0.791667         0.883333        0.066667               24
26       0.000604  4.929179e-04         0.000393        0.000481      10            3  ...          0.208333          0.250000          0.625000         0.316667        0.155009               36
27       0.000798  3.989232e-04         0.000000        0.000000      10            4  ...          0.791667          1.000000          0.958333         0.916667        0.069722               20
28       0.000398  4.880004e-04         0.000400        0.000490      10            4  ...          0.875000          0.958333          0.791667         0.883333        0.066667               24
29       0.000792  3.961172e-04         0.000199        0.000399      10            4  ...          0.208333          0.250000          0.625000         0.316667        0.155009               36
30       0.000200  3.999710e-04         0.000000        0.000000     100            3  ...          0.833333          0.916667          0.916667         0.883333        0.061237               24
31       0.000997  9.368364e-07         0.000000        0.000000     100            3  ...          0.958333          1.000000          0.875000         0.958333        0.045644                7
32       0.000599  4.892520e-04         0.000199        0.000398     100            3  ...          0.875000          0.958333          0.791667         0.883333        0.066667               24
33       0.000000  0.000000e+00         0.000000        0.000000     100            4  ...          0.833333          0.916667          0.916667         0.883333        0.061237               24
34       0.000997  1.066240e-06         0.000000        0.000000     100            4  ...          0.958333          1.000000          0.875000         0.958333        0.045644                7
35       0.000399  4.886403e-04         0.000399        0.000489     100            4  ...          0.875000          0.958333          0.791667         0.883333        0.066667               24
36       0.000000  0.000000e+00         0.000199        0.000399    1000            3  ...          0.791667          0.875000          0.916667         0.841667        0.061237               30
37       0.000798  3.990461e-04         0.000199        0.000399    1000            3  ...          0.958333          1.000000          1.000000         0.991667        0.016667                1
38       0.000000  0.000000e+00         0.000599        0.000489    1000            3  ...          0.958333          1.000000          0.875000         0.958333        0.045644                7
39       0.000393  4.817685e-04         0.000000        0.000000    1000            4  ...          0.791667          0.875000          0.916667         0.841667        0.061237               30
40       0.000407  4.979388e-04         0.000398        0.000487    1000            4  ...          0.958333          1.000000          1.000000         0.991667        0.016667                1
41       0.000000  0.000000e+00         0.000000        0.000000    1000            4  ...          0.958333          1.000000          0.875000         0.958333        0.045644                7

[42 rows x 17 columns]
'''

bbb = aaa[['params', 'mean_test_score', 'rank_test_score','split0_test_score']] 
        #    'split0_test_score', 'split1_test_score', 'split2_test_score', 
        #    'split3_test_score', 'split4_test_score']]

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