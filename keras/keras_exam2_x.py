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

#2-1 모델1
input1 = Input(shape=(1,))
dense1 = Dense(100, activation='relu')(input1)
dense2 = Dense(300, activation='relu')(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(200, activation='relu')(drop1)
output1 = Dense(100)(dense3)

#2-1 모델2
input2 = Input(shape=(1,))
dense11 = Dense(160, activation='relu')(input1)
dense12 = Dense(340, activation='relu')(dense11)
drop11 = Dropout(0.2)(dense12)
dense13 = Dense(160, activation='relu')(drop11)
output2 = Dense(50)(dense13)

merge1 = Concatenate(axis=1)([output1, output2])

#2-3 output모델1
output21 = Dense(150, activation='relu')(merge1)
output22 = Dense(120)(output21)
output23 = Dense(30, activation='relu')(output22)
last_output1 = Dense(1)(output23)

#2-4 output모델2
output31 = Dense(200, activation='relu')(merge1)
output32 = Dense(120)(output31)
output33 = Dense(30, activation='relu')(output32)
last_output2 = Dense(1)(output33)

model = Model(inputs=[input1,input2], outputs=[last_output1,last_output2])
# model.summary()


#3. 컴파일, 훈련

model.compile(loss='mae', optimizer='adam') 

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

model.fit([samsung_x_train, kiwoom_x_train], [samsung_y_train, kiwoom_y_train], epochs=100, batch_size=1 ,verbose=1, validation_split=0.8, callbacks=[es]) 

model.save("./save/keras.exam2.h5")


#4. 평가, 예측

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

