import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, Bidirectional       # 보통 첫글자가 소문자면 함수(기능), 대문자면 클래스(집단안에 모든 개체들과 액션들?) 

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
# model.add(SimpleRNN(10, input_shape=(3,1), return_sequences=True))   # '3':timesteps(나눈구간의 feature의 갯수), '1':input(feature)
model.add(Bidirectional(SimpleRNN(10),input_shape=(3,1)))    #  Bidirectional으로 RNN을 랩핑한다
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
bidirectional (Bidirectional (None, 20)                240          --------> 역으로 연산을 한번 더함(파라미터 갯수 2배)
_________________________________________________________________
dense (Dense)                (None, 10)                210
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
=================================================================
Total params: 461
Trainable params: 461
Non-trainable params: 0
_________________________________________________________________
'''

# #3. 컴파일, 훈련
# model.compile(loss='mae', optimizer='adam')      # optimizer는 loss를 최적화한다
# model.fit(x, y, epochs=100, batch_size=1)

# #4. 평가, 예측
# model.evaluate(x, y)

# x2 = np.array([5,6,7]).reshape(1,3,1)

# result = model.predict(x2)
# print(result)
