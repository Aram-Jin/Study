import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pylab as plt
import numpy as np


#1. 데이터
datasets = load_breast_cancer()
# print(datasets.DESCR)
# print(datasets.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data 
y = datasets.target

x = np.delete(x,0,axis=1)
# x = np.delete(x,[0,1],axis=1)

# print(x.shape, y.shape)  # (150, 4) (150,)
#print(y)
#print(np.unique(y))  # [0 1 2]

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape, y_train.shape)  # (120, 4) (120, 3)
# print(x_test.shape, y_test.shape)  # (30, 4) (30, 3)

#2. 모델구성   -> feature importance는 트리계열에만 있음

model_list = [model1,model2,model3,model4]
model_name = ['DecisionTreeClassifier','RandomForestClassifier','XGBClassifier','GradientBoostingClassifier']
for i in range(4):
    plt.subplot(2, 2, i+1)               # nrows=2, ncols=1, index=1
    model_list[i].fit(x_train, y_train)

    result = model_list[i].score(x_test, y_test)
    feature_importances_ = model_list[i].feature_importances_

    y_predict = model_list[i].predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print("result", result)
    print("accuracy_score", acc)
    print("feature_importances_", feature_importances_)
    plot_feature_importances_dataset(model_list[i])
    plt.ylabel(model_name[i])

plt.show()


#2.모델
model1 = DecisionTreeClassifier(max_depth=5)
model2 = RandomForestClassifier(max_depth=5)
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()

# #3. 훈련
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)

# # #4. 평가, 예측
# result = model.score(x_test, y_test)    # score는 자동으로 맞춰서 반환해줌; 여기서 반환해주는건 'accuracy' (분류모델이기 때문에)

# from sklearn.metrics import accuracy_score
# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)

# print("accuracy_score : ", acc)

# print(model.feature_importances_)