import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel

#1. 데이터
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape)   # (442, 10) (442,)

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
# [0.02593721 0.03284872 0.03821949 0.04788679 0.05547739 0.06321319
#  0.06597802 0.07382318 0.19681741 0.3997985
print("====================================================")

for thresh in aaa:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBRegressor(n_job=-1)

# y_predict = model.predict(x_test)
# print('r2_score : ', r2_score(y_test, y_predict))