import numpy as np
import pandas as pd
from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
# x, y = load_boston(return_X_y=True)
# print(x.shape, y.shape)   # (506, 13) (506,)

datasets = load_boston()
x = datasets.data 
y = datasets.target
print(datasets.feature_names)  # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

x = np.delete(x, [1,3,6,11], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBRegressor(n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
print('model.score : ', score)

print(model.feature_importances_)
print(np.sort(model.feature_importances_))
aaa = np.sort(model.feature_importances_)
# [0.01447933 0.00363372 0.01479118 0.00134153 0.06949984 0.30128664
#  0.01220458 0.05182539 0.0175432  0.03041654 0.04246344 0.01203114
#  0.4284835 ]
# [0.00134153 0.00363372 0.01203114 0.01220458 0.01447933 0.01479118
#  0.0175432  0.03041654 0.04246344 0.05182539 0.06949984 0.30128664
#  0.4284835 ]
print("====================================================")

for thresh in aaa:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    
    print("Thresh=%.3f, n=%d, R2: %.2f%%"
          %(thresh, select_x_train.shape[1], score*100))

'''
(404, 13) (102, 13)
Thresh=0.001, n=13, R2: 92.21%
(404, 12) (102, 12)
Thresh=0.004, n=12, R2: 92.16%
(404, 11) (102, 11)
Thresh=0.012, n=11, R2: 92.03%
(404, 10) (102, 10)
Thresh=0.012, n=10, R2: 92.20%
(404, 9) (102, 9)
Thresh=0.014, n=9, R2: 93.06%
(404, 8) (102, 8)
Thresh=0.015, n=8, R2: 92.36%
(404, 7) (102, 7)
Thresh=0.018, n=7, R2: 91.51%
(404, 6) (102, 6)
Thresh=0.030, n=6, R2: 92.70%
(404, 5) (102, 5)
Thresh=0.042, n=5, R2: 91.78%
(404, 4) (102, 4)
Thresh=0.052, n=4, R2: 92.08%
(404, 3) (102, 3)
Thresh=0.069, n=3, R2: 92.54%
(404, 2) (102, 2)
Thresh=0.301, n=2, R2: 69.85%
(404, 1) (102, 1)
Thresh=0.428, n=1, R2: 45.81%
--------------------------------------
(404, 9) (102, 9)
Thresh=0.011, n=9, R2: 93.06%
(404, 8) (102, 8)
Thresh=0.019, n=8, R2: 93.51%
(404, 7) (102, 7)
Thresh=0.025, n=7, R2: 93.47%
(404, 6) (102, 6)
Thresh=0.039, n=6, R2: 92.70%
(404, 5) (102, 5)
Thresh=0.048, n=5, R2: 91.78%
(404, 4) (102, 4)
Thresh=0.062, n=4, R2: 92.08%
(404, 3) (102, 3)
Thresh=0.064, n=3, R2: 92.54%
(404, 2) (102, 2)
Thresh=0.303, n=2, R2: 69.85%
(404, 1) (102, 1)
Thresh=0.430, n=1, R2: 45.81%


'''

