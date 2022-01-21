# 피처임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여 데이터셋 재구성후 각모델별로 돌려서 결과 도출
# 기존 모델결과와 결과비교

# 결과비교
# 1.DecisionTree
# 기존 acc
# 컬럼삭제 후 acc

import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pylab as plt
import numpy as np

#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data 
y = datasets.target
# print(x.shape, y.shape)  # (150, 4) (150,)
#print(y)
#print(np.unique(y))  # [0 1 2]
x = np.delete(x,[0,1],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape, y_train.shape)  # (120, 4) (120, 3)
# print(x_test.shape, y_test.shape)  # (30, 4) (30, 3)

#2. 모델구성   -> feature importance는 트리계열에만 있음
model1 = DecisionTreeClassifier(max_depth=5)
model2 = RandomForestClassifier(max_depth=5)
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()

#3. 훈련
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)

#4. 평가, 예측
result1 = model1.score(x_test, y_test)    # score는 자동으로 맞춰서 반환해줌; 여기서 반환해주는건 'accuracy' (분류모델이기 때문에)
result2 = model2.score(x_test, y_test)
result3 = model3.score(x_test, y_test)
result4 = model4.score(x_test, y_test)

print("DecisionTreeClassifier 했을때 accuracy_score : ", result1)
print("RandomForestClassifier 했을때 accuracy_score : ", result2)
print("GradientBoostingClassifier 했을때 accuracy_score : ", result3)
print("XGBClassifier 했을때 accuracy_score : ", result4)

print(model1.feature_importances_)
print(model2.feature_importances_)
print(model3.feature_importances_)
print(model4.feature_importances_)


def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

# plt.subplot(2, 2, 1)
# plot_feature_importances_dataset(model1)
# plt.subplot(2, 2, 2)
# plot_feature_importances_dataset(model2)
# plt.subplot(2, 2, 3)
# plot_feature_importances_dataset(model3)
# plt.subplot(2, 2, 4)
# plot_feature_importances_dataset(model4)
    
plt.show()

"""
DecisionTreeClassifier 했을때 accuracy_score :  0.9666666666666667
RandomForestClassifier 했을때 accuracy_score :  0.9333333333333333
GradientBoostingClassifier 했을때 accuracy_score :  0.9666666666666667
XGBClassifier 했을때 accuracy_score :  0.9
[0.         0.0125026  0.03213177 0.95536562]
[0.11696523 0.03003001 0.42780392 0.42520084]
[0.0046703  0.01139495 0.38557268 0.59836207]
[0.01835513 0.0256969  0.6204526  0.33549538]

컬럼제거후====================================================
DecisionTreeClassifier 했을때 accuracy_score :  0.9333333333333333
[0.54517411 0.45482589]

RandomForestClassifier 했을때 accuracy_score :  0.9666666666666667
[0.53647363 0.46352637]

GradientBoostingClassifier 했을때 accuracy_score :  0.9666666666666667
[0.18140014 0.81859986]

XGBClassifier 했을때 accuracy_score :  0.9666666666666667
[0.51089597 0.489104  ]
"""




'''
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_wine, load_diabetes, load_boston, load_breast_cancer, fetch_covtype

datasets = {'Iris':load_iris(),
            'Wine':load_wine(),
            'Diabets':load_diabetes(),
            'Cancer':load_breast_cancer(),
            'Boston':load_boston(),
            'FetchCov':fetch_covtype(),
            'Kaggle_Bike':'Kaggle_Bike'
            }

model_1 = DecisionTreeClassifier(random_state=66, max_depth=5)
model_1r = DecisionTreeRegressor(random_state=66, max_depth=5)

model_2 = RandomForestClassifier(random_state=66, max_depth=5)
model_2r = RandomForestRegressor(random_state=66, max_depth=5)

model_3 = XGBClassifier(random_state=66)
model_3r = XGBRegressor(random_state=66)

model_4 = GradientBoostingClassifier(random_state=66)
model_4r = GradientBoostingRegressor(random_state=66)

def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

path = "D:\\Study\\_data\\bike\\"
train = pd.read_csv(path + "train.csv")

model_list = [model_1,model_2,model_3,model_4]
model_list_r = [model_1r,model_2r,model_3r,model_4r]

model_name = ['DecisionTree','RandomForest','XGB','GradientBoosting']

for (dataset_name, dataset) in datasets.items():
    print(f'------------{dataset_name}-----------')
    print('====================================')    
    
    if dataset_name == 'Kaggle_Bike':
        y = train['count']
        x = train.drop(['casual', 'registered', 'count'], axis=1)        
        x['datetime'] = pd.to_datetime(x['datetime'])
        x['year'] = x['datetime'].dt.year
        x['month'] = x['datetime'].dt.month
        x['day'] = x['datetime'].dt.day
        x['hour'] = x['datetime'].dt.hour
        x = x.drop('datetime', axis=1)
        y = np.log1p(y)        
    else:
        x = dataset.data
        y = dataset.target    

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=0.8, shuffle=True, random_state=66)
    plt.figure(figsize=(15,10))
    for i in range(4):        
        plt.subplot(2, 2, i+1)               # nrows=2, ncols=1, index=1
        if dataset_name == 'Cancer':
            model_list_r[i].fit(x_train, y_train)
            score = model_list_r[i].score(x_test, y_test)
            feature_importances_ = model_list_r[i].feature_importances_

            # y_predict = model_list[i].predict(x_test)
            # acc = accuracy_score(y_test, y_predict)
            print("score", score)
            # print("accuracy_score", acc)
            print("feature_importances_", feature_importances_)
            plot_feature_importances_dataset(model_list_r[i])    
            
        else: 
            model_list[i].fit(x_train, y_train)
            score = model_list[i].score(x_test, y_test)
            feature_importances_ = model_list[i].feature_importances_

            # y_predict = model_list[i].predict(x_test)
            # acc = accuracy_score(y_test, y_predict)
            print("score", score)
            # print("accuracy_score", acc)
            print("feature_importances_", feature_importances_)
            plot_feature_importances_dataset(model_list[i])    
            plt.ylabel(model_name[i])
            plt.title(dataset_name)

    plt.tight_layout()
    plt.show()
'''