import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
#print(x.shape, y.shape)    # (442, 10) (442,)
#print(np.unique(y)) 

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0],x_train.shape[1],1)
x_test = scaler.fit_transform(x_test).reshape(x_test.shape[0],x_test.shape[1],1)
# print(x_train.shape, x_test.shape)    # (353, 10, 1) (89, 10, 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(40, return_sequences=False, input_shape=(10,1)))
model.add(Dropout(0.2))  
model.add(Dense(30))
model.add(Dropout(0.2))                  
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
#model.summary()   

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=False)

model.fit(x_train, y_train, epochs=500, validation_split=0.2, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

'''
loss : 5209.08642578125
r2스코어 :  0.19737225172757034
'''