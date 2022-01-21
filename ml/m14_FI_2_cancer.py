
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
# print(datasets.feature_names)   

x = datasets.data 
y = datasets.target
# print(x.shape, y.shape)  # (569, 30) (569,)
# print(y)
# print(np.unique(y))  # [0 1]

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape, y_train.shape)  # (455, 30) (455,)
# print(x_test.shape, y_test.shape)  # (114, 30) (114,)

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

# from sklearn.metrics import accuracy_score
# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)

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

plt.subplot(2, 2, 1)
plot_feature_importances_dataset(model1)
plt.subplot(2, 2, 2)
plot_feature_importances_dataset(model2)
plt.subplot(2, 2, 3)
plot_feature_importances_dataset(model3)
plt.subplot(2, 2, 4)
plot_feature_importances_dataset(model4)
    
plt.show()
