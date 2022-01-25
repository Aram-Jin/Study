from sklearn import datasets
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston, fetch_covtype
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
# datasets = load_boston()   #(506, 13) (506,)
datasets = fetch_covtype()

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)    #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

import pickle
path = '../save/_save/'
pickle.dump(datasets, open(path + 'm26_pickle1_save_datasets.dat', 'wb'))


#2. 모델
# model = XGBRegressor()
model = XGBClassifier(n_estimators=1000,   # 0.8444120980222539   # n_estimators는 epoch와 같은 의미
                     n_jobs=-1,           # 0.8532492079366552
                     learning_rate=0.025, # 0.9412668531295966
                     max_depth=4,         # 0.9372377212851426  # 트리의 깊이
                     min_child_weight=1,
                     subsample=1,
                     colsample_bytree=1,
                     reg_alpha=1,         # 규제  L1   0.9418181554448182
                     reg_lambda=0         # 규제  L2
                     )     

#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          eval_metric='mlogloss',       # rmse, mae, logloss, error
          early_stopping_rounds=2000 
          )
end = time.time()

results = model.score(x_test, y_test)
print("results: ", round(results, 4))

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc : ", round(acc, 4))
print("걸린시간: ", end - start)

# print("=====================================")

# hist = model.evals_result()
# print(hist)

# 저장
# import pickle
# pickle.dump(model, open('../save/_save/m23_pickle1_save.dat', 'wb'))

# path = '../save/_save'
# pickle.dump(model, open(path + 'm23_pickle1_save.dat', 'wb'))

# loss1 = hist.get('validation_0').get('rmse')
# loss2 = hist.get('validation_1').get('rmse')
# plt.plot(loss1, 'y', label="training loss")
# plt.plot(loss2, 'r', label="test loss")
# plt.grid()
# plt.legend()
# plt.show()
