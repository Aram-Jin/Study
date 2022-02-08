import numpy as np, pandas as pd, datetime, time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

#1. 데이터
path = "../_data/kaggle/bike/"    

train = pd.read_csv(path + 'train.csv')
#print(train.shape)  # (10886, 12)
test_file = pd.read_csv(path + 'test.csv')
#print(test_file.shape)  # (6493, 9)
submit_file = pd.read_csv(path + 'sampleSubmission.csv')
#print(submit_file.shape)  # (6493, 2)

x = train.drop(['datetime', 'casual','registered','count'], axis=1)  
#print(x.shape)  # (10886, 8)

y = train['count']
#print(y.shape)  # (10886,)

test_file = test_file.drop(['datetime'], axis=1)  

# 로그변환
# y = np.log1p(y)  # log1p : y값을 log변환하기 전에 1을 더해주는 함수 
# 데이터가 많거나 한쪽으로 치우쳐진(쏠린) 경우 log를 씌워준다. y=0 또는 x=0가 나타나는 경우, 로그 변환은 가능하지 않으므로 log변환 하기 전에 1을 더해주고 계산.

# plt.plot(y)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = RobustScaler()
#scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)

# x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0],x_train.shape[1],1)
# x_test = scaler.fit_transform(x_test).reshape(x_test.shape[0],x_test.shape[1],1)
# print(x_train.shape, x_test.shape)   # (142, 13, 1) (36, 13, 1)

#2. 모델구성
model = Sequential()
model.add(Dense(80, input_dim=8, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu')) 
model.add(Dense(1)) 
#model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.001
optimizer = Adam(lr=learning_rate)

model.compile(loss='mse', optimizer=optimizer)
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto', verbose=1, factor=0.5) 

start = time.time()
model.fit(x_train, y_train, epochs=5000, batch_size=100, validation_split=0.2, callbacks=[es, reduce_lr])
end = time.time() - start

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('걸린시간 :', round(end,4))

y_predict = model.predict(x_test)
# y_predict2 = np.expm1(y_predict)
#print(y_predict[:10])
#print(y_predict2[:10])

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


############ 제출용 제작 ##############
results = model.predict(test_file)

submit_file['count'] = results

#print(submit_file[:10])
# submit_file.to_csv(path + "final.csv", index=False)


# learning_rate :  0.001
# loss :  21651.0723
# 걸린시간 : 86.1643
# r2스코어 :  0.31500723251765916
# RMSE :  147.1430174365238

'''
< 기본 결과값 >
loss : 22330.763671875
r2스코어 :  0.29350292789173293
RMSE :  149.43483486246706
걸린시간:  17.884 초
'''
