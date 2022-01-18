import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import  Dense, Dropout, Input, Concatenate, concatenate, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.arima_model import ARIMA
from tensorflow.python.keras.layers.recurrent import LSTM
'''
def split_xy5(dataset, time_steps, y_column):
    x,y = list(), list()  
    
    for i in range(len(dataset)):    # for: 반복문
        x_end_number= i + time_steps    
        y_end_number = x_end_number + y_column   
        
        if y_end_number > len(dataset):  
            break
        
        tmp_x = dataset[i:x_end_number, :] 
        tmp_y = dataset[x_end_number: y_end_number, 3] 
        x.append(tmp_x)
        y.append(tmp_y)   
    return np.array(x),np.array(y)
'''

#1. 데이터
path = "../samsung/"    

samsung = pd.read_csv(path +"삼성전자.csv", index_col=0, header = 0, thousands =',', encoding='cp949').iloc[:200,:].sort_values(['일자'],ascending=[True])
kiwoom = pd.read_csv(path + '키움증권.csv', index_col=0, header = 0, thousands =',', encoding='cp949').iloc[:200,:].sort_values(['일자'],ascending=[True])

samsung_x = samsung['금액(백만)'].values
samsung_y = samsung['거래량'].values
kiwoom_x = kiwoom['금액(백만)'].values
kiwoom_y = kiwoom['거래량'].values

# samsung_x = samsung_x.reshape(1,-1)
# samsung_y = samsung_y.reshape(1,-1)
# kiwoom_x = kiwoom_x.reshape(1,-1)
# kiwoom_y = kiwoom_y.reshape(1,-1)
# print(samsung_x.shape, samsung_y.shape)   # (1, 200) (1, 200)
# print(kiwoom_x.shape, kiwoom_y.shape)   # (1, 200) (1, 200)

samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test = train_test_split(samsung_x, samsung_y, train_size=0.8, shuffle=True, random_state=66)
kiwoom_x_train, kiwoom_x_test, kiwoom_y_train, kiwoom_y_test = train_test_split(kiwoom_x, kiwoom_y, train_size=0.8, shuffle=True, random_state=66)
print(samsung_x_train.shape, samsung_x_test.shape, samsung_y.shape)   # (160,) (40,) (200,)
print(kiwoom_x_train.shape, kiwoom_x_test.shape, kiwoom_y.shape)   # (160,) (40,) (200,)

samsung_x_train = samsung_x_train.reshape(1,-1)
samsung_x_test = samsung_x_test.reshape(1,-1)
samsung_y = samsung_y.reshape(1,-1)

kiwoom_x_train = kiwoom_x_train.reshape(1,-1)
kiwoom_x_test = kiwoom_x_test.reshape(1,-1)
kiwoom_y = kiwoom_y.reshape(1,-1)

print(samsung_x_train.shape, samsung_x_test.shape, samsung_y.shape)   # (1, 160) (1, 40) (1, 200)
print(kiwoom_x_train.shape, kiwoom_x_test.shape, kiwoom_y.shape)     # (1, 160) (1, 40) (1, 200)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

samsung_x_train = scaler.fit_transform(samsung_x_train)
samsung_x_test = scaler.fit_transform(samsung_x_test)
samsung_x_train = samsung_x_train.reshape(samsung_x_train.shape[0],16,10)
samsung_x_test = samsung_x_test.reshape(samsung_x_test.shape[0],10,4)
# print(samsung_x_train.shape, samsung_x_test.shape)    # (1, 16, 10) (1, 10, 4)

kiwoom_x_train = scaler.fit_transform(kiwoom_x_train)
kiwoom_x_test = scaler.fit_transform(kiwoom_x_test)
kiwoom_x_train = kiwoom_x_train.reshape(kiwoom_x_train.shape[0],16,10)
kiwoom_x_test = kiwoom_x_test.reshape(kiwoom_x_test.shape[0],10,4)
# print(kiwoom_x_train.shape, kiwoom_x_test.shape)    # (1, 16, 10) (1, 10, 4)

#2. 모델구성
#2-1 모델1
input1 = Input(shape=(16,10))
dense1 = Conv1D(100,2, activation='relu')(input1)
# flatten = Flatten()(dense1)
dense2 = LSTM(300, activation='relu')(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(200, activation='relu')(drop1)
output1 = Dense(100)(dense3)

#2-1 모델2
input2 = Input(shape=(16,10))
dense11 = Conv1D(100,2, activation='relu')(input1)
# flatten = Flatten()(dense11)
dense12 = LSTM(340, activation='relu')(dense11)
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
model.summary()


#3. 컴파일, 훈련

model.compile(loss='mae', optimizer='adam') 

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

model.fit([samsung_x_train, kiwoom_x_train], [samsung_y_train, kiwoom_y_train], epochs=100, batch_size=1 ,verbose=1, validation_split=0.7, callbacks=[es]) 

model.save("./save/keras.exam2_2.h5")


#4. 평가, 예측

results = model.evaluate([samsung_x_test, kiwoom_x_test], [samsung_y_test, kiwoom_y_test])
print('loss : ',results)

samsung_y_pred, kiwoom_y_pred = model.predict([samsung_x, kiwoom_x])

ss = samsung_y_pred[-1]   
kw = kiwoom_y_pred[-1]

print('삼성전자 거래량 : ', ss)
print('키움증권 거래량 : ', kw)




# samsung_result, kiwoom_result = model.predict([])
