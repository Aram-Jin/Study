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

samsung = samsung.iloc[:100,:].sort_values(['일자'],ascending=[True])
kiwoom = kiwoom.iloc[:100,:].sort_values(['일자'],ascending=[True])

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


#2.모델 불러오기
model = load_model("./save/keras.exam3_2.h5") 


#3. 평가, 예측

results = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
print('loss : ',results)

# 마지막 time_steps만큼의 데이터로 미래값을 predict하기위해 (fit했을때의 형태와) shape맞춰주기
s_pred = s[-10:]
k_pred = k[-10:]
# print(s_pred.shape)   (10,2)
# print(k_pred.shape)   (10,2)
s_pred = s_pred.reshape(1, s_pred.shape[0], s_pred.shape[1])
k_pred = k_pred.reshape(1, k_pred.shape[0], k_pred.shape[1])

# 에측하기
s_wed_pred, k_wed_pred = model.predict([s_pred, k_pred])

print("===================== 2021/12/22 =========================")
print('삼성전자 시가와 종가: ', s_wed_pred.round(0).astype(float))
print('키움증권 시가와 종가: ', k_wed_pred.round(0).astype(float))


'''
loss :  [7379.056640625, 3022.320556640625, 4356.73486328125]
===================== 2021/12/22 =========================
삼성전자 시가와 종가:  [[75070. 74779.]]
키움증권 시가와 종가:  [[109640. 110671.]]
'''