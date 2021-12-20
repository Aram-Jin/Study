import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, concatenate
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
path = "../samsung/"    

samsung = pd.read_csv(path +"삼성전자.csv", index_col=0, header = 0, thousands =',', encoding='cp949')
kiwoom = pd.read_csv(path + '키움증권.csv', index_col=0, header = 0, thousands =',', encoding='cp949')

samsung = samsung.iloc[:200,:].sort_values(['일자'],ascending=[True])
kiwoom = kiwoom.iloc[:200,:].sort_values(['일자'],ascending=[True])

samsung_x = samsung[['금액(백만)']].values
samsung_y = samsung[['거래량']].values
kiwoom_x = kiwoom[['금액(백만)']].values
kiwoom_y = kiwoom[['거래량']].values

# print(samsung_x.shape, samsung_y.shape)   # (200, 1) (200, 1)
# print(kiwoom_x.shape, kiwoom_y.shape)   # (200, 1) (200, 1)

samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test = train_test_split(samsung_x, samsung_y, train_size=0.8, shuffle=True, random_state=66)
kiwoom_x_train, kiwoom_x_test, kiwoom_y_train, kiwoom_y_test = train_test_split(kiwoom_x, kiwoom_y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델 불러오기
model = load_model("./save/keras.exam2.h5")

#3. 평가, 예측

results = model.evaluate([samsung_x_test, kiwoom_x_test], [samsung_y_test, kiwoom_y_test])
print('loss : ',results)

samsung_y_pred, kiwoom_y_pred = model.predict([samsung_x, kiwoom_x])

ss = samsung_y_pred[-1]   
kw = kiwoom_y_pred[-1]

print('삼성전자 거래량 : ', ss)
print('키움증권 거래량 : ', kw)


'''
삼성전자 거래량 :  [11439431.]
키움증권 거래량 :  [78190.93]
'''


# samsung_result, kiwoom_result = model.predict([])
