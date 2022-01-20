import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pylab as plt

#1. 데이터
path = "../_data/kaggle/bike/"    

train = pd.read_csv(path + 'train.csv')
#print(train.shape)  # (10886, 12)
test_file = pd.read_csv(path + 'test.csv')
#print(test_file.shape)  # (6493, 9)
submit_file = pd.read_csv(path + 'sampleSubmission.csv')
#print(submit_file.shape)  # (6493, 2)

x = train.drop(['datetime', 'casual','registered','count'], axis=1)  
#print(x.shape)  # (10886, 8)

y = train['count']
#print(y.shape)  # (10886,)

test_file = test_file.drop(['datetime'], axis=1)  

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성  -> feature importance는 트리계열에만 있음
model1 = DecisionTreeRegressor(max_depth=5)
model2 = RandomForestRegressor(max_depth=5)
model3 = GradientBoostingRegressor()
model4 = XGBRegressor()

# #3. 훈련
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)

#4. 평가, 예측
result1 = model1.score(x_test, y_test)    # score는 자동으로 맞춰서 반환해줌; 여기서 반환해주는건 'accuracy' (분류모델이기 때문에)
result2 = model2.score(x_test, y_test)
result3 = model3.score(x_test, y_test)
result4 = model4.score(x_test, y_test)

print(model1.feature_importances_)
print(model2.feature_importances_)
print(model3.feature_importances_)
print(model4.feature_importances_)

# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features),model.feature_importances_,align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1,n_features)

# plt.subplot(2, 2, 1)
# plot_feature_importances_dataset(model1)
# plt.subplot(2, 2, 2)
# plot_feature_importances_dataset(model2)
# plt.subplot(2, 2, 3)
# plot_feature_importances_dataset(model3)
# plt.subplot(2, 2, 4)
# plot_feature_importances_dataset(model4)
    
# plt.show()
