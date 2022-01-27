import numpy as np
import pandas as pd
import time
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer,PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score

#1. 데이터
datasets = fetch_covtype()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)


x_train = np.load('../save/_save/smote_x.npy')
y_train = np.load('../save/_save/smote_y.npy')

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier(n_estimators=1000,  
                     n_jobs=4,          
                     learning_rate=0.15, 
                     max_depth=5,        
                     min_child_weight=1,
                     subsample=1,
                     colsample_bytree=1,
                     reg_alpha=1,         
                     reg_lambda=0,
                     tree_method='gpu_hist',
                     predictor='gpu_predictor',
                     gpu_id=0
                     )   

model.fit(x_train, y_train)

score = model.score(x_test, y_test)


print("model.score : ", round(score, 4))

y_predict = model.predict(x_test)
print("accuracy score: ", round(accuracy_score(y_test, y_predict),4))
print("f1_score : ", round(f1_score(y_test, y_predict, average='macro'),4))


'''
model.score :  0.7898
accuracy score:  0.7898
f1_score :  1 4
=========================================== SMOTE 적용 ===============================================
smote걸린시간:  404.43740797042847
model.score :  0.836
accuracy score:  0.836
f1_score :  1 4
'''
'''
model.score :  0.8408
accuracy score:  0.8408
f1_score :  0.8281


model.score :  0.9111
accuracy score:  0.9111
f1_score :  0.9126
'''