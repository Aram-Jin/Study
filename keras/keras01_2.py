<<<<<<< HEAD
# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])
# 위 데이터를 훈련해서 최소의 loss 값을 찾아내자

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=10000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([6])
print('6의 예측값 : ', result)

'''
loss :  0.3800000250339508
6의 예측값 :  [[5.7003927]]
'''

=======
# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])
# 위 데이터를 훈련해서 최소의 loss 값을 찾아내자

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=10000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([6])
print('6의 예측값 : ', result)

'''
loss :  0.3800000250339508
6의 예측값 :  [[5.7003927]]
'''

>>>>>>> d14e67c0f3df3b66f0afc433754c01e7680f6f46
# loss 값이 0.38 아래로 안떨어짐