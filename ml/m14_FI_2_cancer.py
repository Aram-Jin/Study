
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

'''
DecisionTreeClassifier 했을때 accuracy_score :  0.9122807017543859
RandomForestClassifier 했을때 accuracy_score :  0.956140350877193
GradientBoostingClassifier 했을때 accuracy_score :  0.9473684210526315
XGBClassifier 했을때 accuracy_score :  0.9736842105263158
[0.         0.06054151 0.         0.         0.         0.03007617
 0.         0.02005078 0.         0.         0.01257413 0.
 0.         0.         0.         0.         0.         0.00442037
 0.         0.004774   0.         0.01642816 0.00636533 0.72839202
 0.         0.         0.         0.11637753 0.         0.        ]
[0.02831105 0.01984584 0.02518251 0.04297622 0.00673274 0.00499969
 0.05147486 0.07225873 0.00339102 0.00317985 0.02142336 0.00229644
 0.00977527 0.04203447 0.00325563 0.00286842 0.00500457 0.00231865
 0.00292819 0.00293823 0.15250909 0.02025314 0.17073351 0.11700619
 0.01105752 0.01162326 0.02097093 0.1287061  0.00690486 0.00703966]
[7.70958753e-05 4.04918331e-02 7.27831284e-04 2.05223213e-03
 9.53060449e-04 9.14623889e-06 9.95689359e-04 1.30692500e-01
 3.04838590e-03 2.50339304e-04 4.00967312e-03 7.58255023e-06
 2.26319564e-03 1.78955222e-02 1.65665524e-03 3.83606501e-03
 1.63888735e-03 7.28679807e-04 7.89590693e-06 7.29289641e-03
 3.12761634e-01 3.72752774e-02 3.64236339e-02 2.72809267e-01
 3.35167624e-03 1.42083964e-04 1.36475908e-02 1.04648758e-01
 4.37787288e-05 2.61132433e-04]
[0.01420499 0.03333857 0.         0.02365488 0.00513449 0.06629944
 0.0054994  0.09745206 0.00340272 0.00369179 0.00769183 0.00281184
 0.01171023 0.0136856  0.00430626 0.0058475  0.00037145 0.00326043
 0.00639412 0.0050556  0.01813928 0.02285904 0.22248559 0.2849308
 0.00233393 0.         0.00903706 0.11586287 0.00278498 0.00775311]
'''
