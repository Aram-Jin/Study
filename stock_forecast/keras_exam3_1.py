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

#print(samsung)

# samsung = samsung.values
samsung = samsung.iloc[:10,:].sort_values(['일자'],ascending=[True])
kiwoom = kiwoom.iloc[:10,:].sort_values(['일자'],ascending=[True])

# print(samsung,kiwoom)   # 2020/12/15부터 데이터를 쓰겠다.

s = samsung[['시가','종가']].values
k = kiwoom[['시가','종가']].values

s = s.reshape(-1,1)
# print(s)
print(s.shape)   # (20, 1)

k = k.reshape(-1,1)
# print(k)
print(k.shape)   # (20, 1)

time_steps = 10
y_column = 2
x1, y1 = split_xy2(s, time_steps, y_column)

# print(x1[-5:], '/n', y1[-5:])
print(x1.shape, y1.shape)   # (189, 10, 1) (189, 2, 1)

x2, y2 = split_xy2(k, time_steps, y_column)

# print(x2[-5:], '/n', y2[-5:])
print(x2.shape, y2.shape)   # (189, 10, 1) (189, 2, 1)
'''

#RNN사용시  이미알기때문에 그냥 사용함.
#print(x1.shape)     
#print(y1.shape)    

#DNN쓰기위해 2차원으로 바꿔줌.
#x1 = x1.reshape(len(x1),-1)
#x2 = x2.reshape(len(x2),-1)

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
input1 = Input(shape=(10,1))
dense1 = LSTM(100, activation='relu')(input1)
dense2 = Dense(300, activation='relu')(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(200, activation='relu')(drop1)
output1 = Dense(10)(dense3)

#2-1 모델2
input2 = Input(shape=(10,1))
dense11 = LSTM(160, activation='relu')(input1)
dense12 = Dense(340, activation='relu')(dense11)
drop11 = Dropout(0.2)(dense12)
dense13 = Dense(160, activation='relu')(drop11)
output2 = Dense(10)(dense13)

merge1 = Concatenate(axis=1)([output1, output2])

#2-3 output모델1
output21 = Dense(150, activation='relu')(merge1)
output22 = Dense(120)(output21)
output23 = Dense(30, activation='relu')(output22)
last_output1 = Dense(2)(output23)

#2-4 output모델2
output31 = Dense(200, activation='relu')(merge1)
output32 = Dense(120)(output31)
output33 = Dense(30, activation='relu')(output32)
last_output2 = Dense(2)(output33)

model = Model(inputs=[input1,input2], outputs=[last_output1,last_output2])
model.summary()

#3. 컴파일, 훈련

model.compile(loss='mae', optimizer='adam') 
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)

model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=1 ,verbose=1, validation_split=0.3, callbacks=[es]) 

model.save("./save/keras.exam3.h5")

#4. 평가, 예측

results = model.evaluate([x1_test, x2_test], [y1_test, y2_test])
print('loss : ',results)

submit_x1 = np.append(x1[-1],y1[-1],axis= 0)
print(submit_x1)
print(submit_x1[-10:])  

submit_x2 = np.append(x2[-1],y2[-1],axis= 0)
print(submit_x2)
print(submit_x2[-10:])  

# print(x1.shape)  # (189, 10, 1)

s_mon_pred, k_mon_pred = model.predict([submit_x1[-10:], submit_x2[-10:]])
print(s_mon_pred, k_mon_pred)
'''