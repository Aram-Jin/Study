# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])
# 위 데이터를 훈련해서 최소의 loss 값을 찾아내자

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=3000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([4])
print('4의 예측값 : ', result)


'''
loss :  3.706664695232037e-13
4의 예측값 :  [[4.000002]]
'''

# 1000 일때 4.001 나왔는데 다시 돌리니 안나옴