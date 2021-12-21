
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, concatenate
from tensorflow.keras.callbacks import EarlyStopping

def split_xy2(dataset, time_steps, y_column):
    x,y = list(), list()  
    
    for i in range(len(dataset)):
        x_end_number= i + time_steps    
        y_end_number = x_end_number + y_column 
        
        if y_end_number > len(dataset):  
            break
        
        tmp_x = dataset[i:x_end_number] 
        tmp_y = dataset[x_end_number: y_end_number] 
        x.append(tmp_x)
        y.append(tmp_y)   
    return np.array(x),np.array(y)
    

#1. 데이터
path = "../samsung/"    

samsung = pd.read_csv(path +"삼성전자.csv", index_col=0, header = 0, thousands =',', encoding='cp949')
kiwoom = pd.read_csv(path + '키움증권.csv', index_col=0, header = 0, thousands =',', encoding='cp949')

samsung = samsung.iloc[:100,:].sort_values(['일자'],ascending=[True])
kiwoom = kiwoom.iloc[:100,:].sort_values(['일자'],ascending=[True])
      
s = samsung[['시가','종가']].values
k = kiwoom[['시가','종가']].values

s = s.reshape(-1,1)
# print(s.shape)   # (200, 1)

k = k.reshape(-1,1)
# print(k.shape)   # (200, 1)

time_steps = 10
y_column = 2

x1, y1 = split_xy2(s, time_steps, y_column)
# print(x1.shape, y1.shape)   # (189, 10, 1) (189, 2, 1)

x2, y2 = split_xy2(k, time_steps, y_column)
# print(x2.shape, y2.shape)   # (189, 10, 1) (189, 2, 1)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1,x2,y1,y2, train_size=0.8, shuffle=True, random_state=66)

# print(x1_train.shape)   # (151, 10, 1)
# print(x1_test.shape)   # (38, 10, 1)
# print(x2_train.shape)   # (151, 10, 1)
# print(x2_test.shape)   # (38, 10, 1)

# print(y1_train.shape)   # (151, 2, 1)
# print(y1_test.shape)   # (38, 2, 1)
# print(y2_train.shape)   # (151, 2, 1)
# print(y2_test.shape)   # (38, 2, 1)

#2. 모델구성
#2-1 모델1
input1 = Input(shape=(x1_train.shape[1],x1_train.shape[2]))
dense1 = LSTM(160, activation='tanh')(input1)
dense2 = Dense(340, activation='relu')(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(200, activation='relu')(drop1)
dense4 = Dense(32, activation='relu')(dense3)
output1 = Dense(10)(dense4)

#2-1 모델2
input2 = Input(shape=(x2_train.shape[1],x2_train.shape[2]))
dense11 = LSTM(160, activation='tanh')(input1)
dense12 = Dense(340, activation='relu')(dense11)
drop11 = Dropout(0.2)(dense12)
dense13 = Dense(160, activation='relu')(drop11)
dense14 = Dense(32, activation='relu')(dense13)
output2 = Dense(10)(dense14)

merge1 = Concatenate(axis=1)([output1, output2])#

#2-3 output모델1
output21 = Dense(200, activation='relu')(merge1)
output22 = Dense(120)(output21)
output23 = Dense(30, activation='relu')(output22)
output24 = Dense(16, activation='relu')(output23)
last_output1 = Dense(2)(output24)

#2-4 output모델2
output31 = Dense(200, activation='relu')(merge1)
output32 = Dense(120)(output31)
output33 = Dense(30, activation='relu')(output32)
output34 = Dense(16, activation='relu')(output33)
last_output2 = Dense(2)(output34)

model = Model(inputs=[input1,input2], outputs=[last_output1,last_output2])
#model.summary()

#3. 컴파일, 훈련

model.compile(loss='mae', optimizer='adam') 
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=1, batch_size=1 ,verbose=1, validation_split=0.3, callbacks=[es]) 

model.save("./save/keras.exam3.h5")

#4. 평가, 예측

results = model.evaluate([x1_test, x2_test], [y1_test, y2_test])
print('loss : ',results)

# 금요일에서 끝나고 그 다음주월요일의 값을 predict함
# 여기서 나온값을 각각 삼성 시가 종가 파일, 키움 시가종가 파일에 합쳐줌.
submit_x1= np.append(x1[-1],y1[-1],axis= 0)[-10:]
#print(submit_x1)
# print(submit_x1[-10:])

submit_x2 = np.append(x2[-1],y2[-1],axis= 0)[-10:]
# print(submit_x2)
# print(submit_x2[-10:])  

s_mon_pred, k_mon_pred = model.predict([submit_x1, submit_x2])

s_mon_pred = s_mon_pred[0].round(0).astype(float)
k_mon_pred = k_mon_pred[0].round(0).astype(float)

print('삼성전자 월요일 시가, 종가 : ', s_mon_pred)
print('키움증권 월요일 시가, 종가: .', k_mon_pred)


# print('삼성전자 수요일시가,종가 : ',s_mon_pred[-1+3], '키움증권 수요일 시가, 종가: .',k_mon_pred[-1+3])

# 삼성전자 월요일시가,종가 : , [65729.11 65420.7 ] 키움증권 월요일 시가, 종가: . [96803.77  96827.234]

# 삼성전자 월요일시 가,종가 :  [66704. 66109.] 키움증권 월요일 시가, 종가: . [96203. 96186.]    
# 삼성전자 월요일시 가,종가 :  [64356. 65419.] 키움증권 월요일 시가, 종가: . [95263. 95324.]
# 삼성전자 월요일시 가,종가 :  [66989. 67038.] 키움증권 월요일 시가, 종가: . [96904. 96912.]

# r21 = r2_score(y1_test, ss)
# r22 = r2_score(y2_test, kw)

# print('r2_1스코어 : ', r21)
# print('r2_2스코어 : ', r22)

