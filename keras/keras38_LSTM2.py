import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

#1. 데이터
x = np.array([[1,2,3],                               #array는 행렬을 의미! 모든 연산은 numpy로 한다. 
             [2,3,4],                                #x에서 [1,2,3]으로 짜른 범위는 timesteps, 그 안의 1은 feature
             [3,4,5],
             [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape)   # (4, 3) (4,)

# input_shape = (batch_size, timesteps, feature)
# input_shape = (행, 열, 몇개씩 짜르는지!!!)
x = x.reshape(4, 3, 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(10, activation='linear', input_shape=(3,1)))   # '3':timesteps(나눈구간의 feature의 갯수), '1':input(feature)
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()
'''
#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')      # optimizer는 loss를 최적화한다
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
model.evaluate(x, y)

x2 = np.array([5,6,7]).reshape(1,3,1)

result = model.predict(x2)
print(result)
'''