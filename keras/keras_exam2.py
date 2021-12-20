import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, concatenate
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.arima_model import ARIMA

def split_xy5(dataset, time_steps, y_column):
    x,y = list(), list()  
    
    for i in range(len(dataset)):
        x_end_number= i + time_steps    
        y_end_number = x_end_number + y_column   
        
        if y_end_number > len(dataset):  
            break
        
        tmp_x = dataset[i:x_end_number, :] 
        tmp_y = dataset[x_end_number: y_end_number, 3] 
        x.append(tmp_x)
        y.append(tmp_y)   
    return np.array(x),np.array(y)

#1. 데이터
path = "../samsung/"    

samsung = pd.read_csv(path +"삼성전자.csv", index_col=0, header = 0, thousands =',', encoding='cp949')
kiwoom = pd.read_csv(path + '키움증권.csv', index_col=0, header = 0, thousands =',', encoding='cp949')

samsung = samsung.iloc[:893,:].sort_values(['일자'],ascending=[True])
kiwoom = kiwoom.iloc[:893,:].sort_values(['일자'],ascending=[True])

s = samsung[['거래량','금액(백만)','등락률']].values
k = kiwoom[['거래량','금액(백만)','등락률']].values

print(s)

'''
x1, y1 = split_xy5(samsung,5,1)
x2, y2 = split_xy5(kiwoom,5,1)

x1 = x1.reshape(888,-1)
x2 = x2.reshape(888,-1)

print(x1.shape, y1.shape)  # (888, 20) (888, 1)
print(x2.shape, y1.shape)  # (888, 20) (888, 1)

# print(samsung.index, kiwoom.index)

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, shuffle=True, random_state=66)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, shuffle=True, random_state=66)

print(x1_train.shape, x1_test.shape)   # (710, 20) (178, 20)
print(x2_train.shape, x2_test.shape)   # (710, 20) (178, 20)

scaler = MinMaxScaler()
#scaler = StandardScaler()
# scaler = RobustScaler()
#scaler = MaxAbsScaler()
x1_train = scaler.fit_transform(x1_train).reshape(len(x1_train),5,4)
x1_test = scaler.transform(x1_test).reshape(len(x1_test),5,4)

x2_train = scaler.fit_transform(x2_train).reshape(len(x2_train),5,4)
x2_test = scaler.transform(x2_test).reshape(len(x2_test),5,4)

print(x1_train.shape, x1_test.shape)   # (710, 5, 4)
print(x2_train.shape, x2_test.shape)   # (710, 5, 4)
print(y1_train.shape, y2_train.shape, y1_test.shape, y2_test.shape)   # (710, 1) (710, 1) (178, 1) (178, 1)

#2. 모델구성

#2-1 모델1
input1 = Input(shape=(5,4))
dense1 = LSTM(16, activation='relu', name='dense1')(input1)
dense2 = Dense(8, activation='relu', name='dense2')(dense1)
dense3 = Dense(4, activation='relu', name='dense3')(dense2)
output1 = Dense(1, activation='relu', name='output1')(dense3)

#2-1 모델2
input2 = Input(shape=(5,4))
dense11 = LSTM(16, activation='relu')(input1)
dense12 = Dense(8, activation='relu')(dense11)
dense13 = Dense(4, activation='relu')(dense12)
output2 = Dense(1, activation='relu')(dense13)

merge1 = Concatenate(axis=1)([output1, output2])

#2-3 output모델1
output21 = Dense(8, activation='relu')(merge1)
output22 = Dense(12, activation='relu')(output21)
output23 = Dense(4, activation='relu')(output22)
last_output1 = Dense(1, activation='relu')(output23)

#2-4 output모델2
output31 = Dense(8, activation='relu')(merge1)
output32 = Dense(12, activation='relu')(output31)
output33 = Dense(4, activation='relu')(output32)
last_output2 = Dense(1, activation='relu')(output33)

model = Model(inputs=[input1,input2], outputs=[last_output1,last_output2])
# model.summary()

#3. 컴파일, 훈련

# # (AR = 2, 차분 =1, MA=2) 파라미터로 ARIMA 모델을 학습한다.
# model = ARIMA(samsung_train_df.price.values, order = (2,1,2))
# model_fit = model.fit(trend = 'c', full_output = True, disp = True)
# print(model_fit.summary())

model.compile(loss='mae', optimizer='adam') 

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=500, verbose=1, validation_split=0.2, callbacks=[es]) 

model.save("./save/keras.exam2.h5")

#4. 평가, 예측
results = model.evaluate([x1_test, x2_test], [y1_test, y2_test])
print('loss : ',results)

y1_predict, y2_predict = model.predict([x1_test, x2_test])
print('삼성전자 거래량 : ', y1_predict[-1])
print('키움증권 거래량 : ', y2_predict[-1])

r21 = r2_score(y1_test, y1_predict)
r22 = r2_score(y2_test, y2_predict)

print('r2_1스코어 : ', r21)
print('r2_2스코어 : ', r22)

'''