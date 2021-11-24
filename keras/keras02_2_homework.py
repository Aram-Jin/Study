# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])
# 위 데이터를 훈련해서 epochs=50 일때 hidden layer구성을 짜서 예측값 '4' 도출시키기

#2. 모델구성 -> 'hidden layer'
model = Sequential()
model.add(Dense(9, input_dim=1))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=50, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([4])
print('4의 예측값 : ', result)

'''
loss :  4.3297473894199356e-05
4의 예측값 :  [[3.9863973]]
'''

'''
#1. 데이터 -> input layer
#2. 모델구성 -> hidden layer

* hyperparameter tuning
'''
