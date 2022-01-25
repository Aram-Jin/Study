from unittest import result
from sklearn import datasets
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
import time
# import warnings
# warnings.filterwarnings('ignore')

#1. 데이터
# datasets = fetch_california_housing()   #(20640, 8) (20640,)
datasets = load_boston()   #(506, 13) (506,)

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)    #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 불러오기 // 2.모델, 3.훈련
# import pickle
path = '../save/_save/'
# model = pickle.load(open(path + 'm23_pickle1_save.dat', 'rb'))
import joblib
model = joblib.load(path + 'm24_joblib1_save.dat')

#4. 평가
results = model.score(x_test, y_test)
print("results: ", results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 : ", round(r2, 4))

print("=====================================")

hist = model.evals_result()
print(hist)


