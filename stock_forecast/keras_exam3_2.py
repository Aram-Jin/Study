import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, concatenate
from tensorflow.keras.callbacks import EarlyStopping

def split_xy5(dataset, time_steps, y_column):
    x,y = list(), list()  
    
    for i in range(len(dataset)):
        x_end_number= i + time_steps    
        y_end_number = x_end_number + y_column 
        
        if y_end_number > len(dataset):  
            break
        
        tmp_x = dataset[i:x_end_number, :] 
        tmp_y = dataset[x_end_number: y_end_number, :] 
        x.append(tmp_x)
        y.append(tmp_y)   
    return np.array(x), np.array(y)

#1. 데이터
path = "../samsung/"    

samsung = pd.read_csv(path +"삼성전자.csv", index_col=0, header = 0, thousands =',', encoding='cp949')
kiwoom = pd.read_csv(path + '키움증권.csv', index_col=0, header = 0, thousands =',', encoding='cp949')

samsung = samsung.iloc[:200,:].sort_values(['일자'],ascending=[True])
kiwoom = kiwoom.iloc[:200,:].sort_values(['일자'],ascending=[True])

s = samsung[['시가','종가']].values
k = kiwoom[['시가','종가']].values
# print(s.shape,k.shape)    # (200, 2) (200, 2)

#함수적용
time_steps = 10
y_column = 3

x1, y1 = split_xy5(s, time_steps, y_column)
# print(x1.shape, y1.shape)   #(188, 10, 2) (188, 3, 2)

x2, y2 = split_xy5(k, time_steps, y_column)
# print(x2.shape, y2.shape)   #(188, 10, 2) (188, 3, 2)


x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1,x2,y1,y2, train_size=0.8, shuffle=True, random_state=66)

# print(x1_train.shape)   # (150, 10, 2)
# print(x1_test.shape)   # (38, 10, 2)
# print(x2_train.shape)   # (150, 10, 2)
# print(x2_test.shape)   # (38, 10, 2)
# print(y1_train.shape)   # (150, 3, 2)
# print(y1_test.shape)   # (38, 3, 2)
# print(y2_train.shape)   # (150, 3, 2)
# print(y2_test.shape)   # (38, 3, 2)


#2. 모델구성

#2-1 모델1
input1 = Input(shape=(10,2))
dense1 = LSTM(160, activation='tanh')(input1)
dense2 = Dense(120, activation='relu')(dense1)
drop1 = Dropout(0.1)(dense2)
dense3 = Dense(32, activation='relu')(drop1)
dense4 = Dense(16, activation='relu')(dense3)
output1 = Dense(10)(dense4)

#2-1 모델2
input2 = Input(shape=(10,2))
dense11 = LSTM(160, activation='tanh')(input1)
dense12 = Dense(120, activation='relu')(dense11)
drop11 = Dropout(0.2)(dense12)
dense13 = Dense(32, activation='relu')(drop11)
dense14 = Dense(16, activation='relu')(dense13)
output2 = Dense(10)(dense14)

merge1 = Concatenate(axis=1)([output1, output2])    #(axis=1)

#2-3 output모델1
output21 = Dense(32, activation='relu')(merge1)
output22 = Dense(86)(output21)
output23 = Dense(16, activation='relu')(output22)
last_output1 = Dense(2)(output23)

#2-4 output모델2
output31 = Dense(32, activation='relu')(merge1)
output32 = Dense(86)(output31)
output33 = Dense(16, activation='relu')(output32)
last_output2 = Dense(2)(output33)

model = Model(inputs=[input1,input2], outputs=[last_output1,last_output2])
# model.summary()

#3. 컴파일, 훈련

model.compile(loss='mae', optimizer='adam') 
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=1, batch_size=1 ,verbose=1, validation_split=0.3, callbacks=[es]) 

# model.save("./save/keras.exam3_2.h5")

#4. 평가, 예측

results = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
print('loss : ',results)

# 마지막 time_steps만큼의 데이터로 미래값을 predict하기위해 (fit했을때의 형태와) shape맞춰주기
s_pred = s[-10:]
k_pred = k[-10:]
# print(s_pred.shape)   (10,2)
# print(k_pred.shape)   (10,2)
s_pred = s_pred.reshape(1, s_pred.shape[0], s_pred.shape[1])
k_pred = k_pred.reshape(1, k_pred.shape[0], k_pred.shape[1])

print(s_pred.shape)

# 에측하기
s_wed_pred, k_wed_pred = model.predict([s_pred, k_pred])

print(s_wed_pred.shape)

print("===================== 2021/12/22 =========================")
print('삼성전자 시가와 종가: ', s_wed_pred.round(0).astype(float))
print('키움증권 시가와 종가: ', k_wed_pred.round(0).astype(float))



'''
loss :  [7379.056640625, 3022.320556640625, 4356.73486328125]
===================== 2021/12/22 =========================
삼성전자 시가와 종가:  [[75070. 74779.]]
키움증권 시가와 종가:  [[109640. 110671.]]
'''