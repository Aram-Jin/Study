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

results = model.evaluate([samsung_x_test, kiwoom_x_test], [samsung_y_test, kiwoom_y_test], batch_size=1)
print('loss : ',results)

samsung_y_pred, kiwoom_y_pred = model.predict([samsung_x, kiwoom_x])

ss_mon = samsung_y_pred[-1]   
kw_mon = kiwoom_y_pred[-1]

print('월요일 삼성전자 거래량 : ', ss_mon)  # [11448460.]
print('월요일 키움증권 거래량 : ', kw_mon)  # [66479.56]

samsung_y2 = np.append(samsung_y,[ss_mon],axis= 0)
kiwoom_y2 = np.append(kiwoom_y,[kw_mon],axis= 0)
#-> 월요일까지 거래량(y)을 예측하여 새로운 y2로 지정해줌(화요일의 거래량 예측을 위해서)


## 월요일 금액 예상하기 =============================================

samsung_xx = samsung[['거래량']].values
samsung_yy = samsung_y2[1:]
kiwoom_xx = kiwoom[['거래량']].values
kiwoom_yy = kiwoom_y2[1:]
# print(samsung_xx.shape, samsung_yy.shape)   # (200, 1) (200, 1)

samsung_xx_train, samsung_xx_test, samsung_yy_train, samsung_yy_test = train_test_split(samsung_xx, samsung_yy, train_size=0.8, shuffle=True, random_state=66)
kiwoom_xx_train, kiwoom_xx_test, kiwoom_yy_train, kiwoom_yy_test = train_test_split(kiwoom_xx, kiwoom_yy, train_size=0.8, shuffle=True, random_state=66)

#2. 모델 불러오기
model = load_model("./save/keras.exam2_1.h5") 

#3. 평가, 예측
results = model.evaluate([samsung_xx_test, kiwoom_xx_test], [samsung_yy_test, kiwoom_yy_test], batch_size=1)
print('loss : ',results)

samsung_yy_pred, kiwoom_yy_pred = model.predict([samsung_xx, kiwoom_xx])

ss_mon2 = samsung_yy_pred[-1]   
kw_mon2 = kiwoom_yy_pred[-1]

print('월요일 삼성전자 금액(백만): ', ss_mon2)  
print('월요일 키움증권 금액(백만) : ', kw_mon2)  

samsung_x2 = np.append(samsung_x,[ss_mon2],axis= 0)
kiwoom_x2 = np.append(kiwoom_x,[kw_mon2],axis= 0)
#-> 월요일까지 금액(백만)(x)을 예측하여 새로운 x2로 지정해줌(화요일의 거래량 예측을 위해서)

samsung_result, kiwoom_result = model.predict([samsung_x2, kiwoom_x2])
ss_tue = samsung_result[-1]   
kw_tue = kiwoom_result[-1]
print('화요일 삼성전자 거래량 : ', ss_tue)
print('화요일 키움증권 거래량 : ', kw_tue)
#=============================================
'''
월요일 삼성전자 거래량 :  [11495695.]
월요일 키움증권 거래량 :  [86135.77]
월요일 삼성전자 금액(백만):  [11142885.]
월요일 키움증권 금액(백만) :  [45128.63]
화요일 삼성전자 거래량 :  [10520140.]
화요일 키움증권 거래량 :  [33671.496]
'''