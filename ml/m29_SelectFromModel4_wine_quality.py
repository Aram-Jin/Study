import time
import numpy as np
import pandas as pd
from sklearn import datasets
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectFromModel

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [10, 12], 'min_samples_leaf' :[3, 7, 10], 'min_samples_split' : [3, 5]},
    {'n_estimators' : [100], 'max_depth' : [6, 12], 'min_samples_leaf' :[7, 10], 'min_samples_split' : [2, 3]},
    {'n_estimators' : [200], 'max_depth' : [10, 12], 'min_samples_leaf' :[3, 5], 'min_samples_split' : [5, 10]},
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8], 'min_samples_leaf' :[3, 10], 'min_samples_split' : [2, 10]},
    {'n_jobs' : [-1, 2, 4]}
]

#1. 데이터
path = "../_data/"    

data = pd.read_csv(path + 'winequality-white.csv', index_col=None, header=0, sep=';', dtype=float)
# print(data.shape)  # (4898, 12)
print(data.head())
print(data.describe())   # pandas에서 볼수있는기능 (수치데이터에서 용이함)
print(data.info())
print(data.columns)
# Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
#        'pH', 'sulphates', 'alcohol', 'quality'],
#       dtype='object')
# print(type(data))   # <class 'pandas.core.frame.DataFrame'>

x = data.drop(['quality'], axis=1)  
# print(x.shape)  # (4898, 11)
y = data['quality']
# print(y.shape)  # (4898,)
print(np.unique(y, return_counts=True))   # (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))

# y = np.where(y == 9, 8, y)
# y = np.where(y == 3, 4, y)
# print(np.unique(y, return_counts=True))

newlist = []
for i in y:
    # print(i)
    if i<=4 :
        newlist +=[0]
    elif i<=7:
        newlist +=[1]
    else:
        newlist +=[2]
            
y = np.array(newlist)
print(y)        
print(np.unique(y, return_counts=True))   


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성
model = GridSearchCV(XGBClassifier(
                     n_estimators=10000,   # n_estimators는 epoch와 같은 의미
                     n_jobs=-1,           
                     learning_rate=0.005, 
                     max_depth=13,         
                     min_child_weight=1,
                     subsample=1,
                     colsample_bytree=1,
                     reg_alpha=0,         # 규제  L1  
                     reg_lambda=1         # 규제  L2
                     )     , parameters, cv=kfold, verbose=1, refit=True)

#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set = [(x_test, y_test)],
          eval_metric= 'merror',        #rmse, mae, logloss, error
          early_stopping_rounds=900 
          )  # mlogloss
end = time.time()

results = model.score(x_test, y_test)
print("results: ", round(results, 4))

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", round(acc, 4))
print("f1_score : ", f1_score(y_test, y_predict, average='macro'))

print("걸린시간: ", end - start)

print(model.feature_importances_)
print(np.sort(model.feature_importances_))
aaa = np.sort(model.feature_importances_)

print("====================================================")

for thresh in aaa:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBClassifier(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    
    print("Thresh=%.3f, n=%d, acc: %.2f%%"
          %(thresh, select_x_train.shape[1], score*100))

'''
results:  0.9398
accuracy_score :  0.9398
f1_score :  0.5926975110302047
걸린시간:  481.7619113922119
'''