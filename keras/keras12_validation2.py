<<<<<<< HEAD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(17))
y = np.array(range(17))
x_train = x[:14]
x_val = x[10:14]
x_test =  x[14:16]
y_train = y[:14]
y_val = y[10:14]
y_test = y[14:16]

'''
x_train = np.array(range(11))
y_train = np.array(range(11))
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])
x_val = np.array([14,15,16])
y_val = np.array([14,15,16])
'''

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict([17])
print("17의 예측값 : ", y_predict)
=======
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(17))
y = np.array(range(17))
x_train = x[:14]
x_val = x[10:14]
x_test =  x[14:16]
y_train = y[:14]
y_val = y[10:14]
y_test = y[14:16]

'''
x_train = np.array(range(11))
y_train = np.array(range(11))
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])
x_val = np.array([14,15,16])
y_val = np.array([14,15,16])
'''

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict([17])
print("17의 예측값 : ", y_predict)
>>>>>>> d14e67c0f3df3b66f0afc433754c01e7680f6f46
