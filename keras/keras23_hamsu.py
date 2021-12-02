import numpy as np

#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array([range(701, 801)])
print(x.shape, y.shape)  # (3, 100) (1, 100)
x = np.transpose(x)
y = np.transpose(y)
print(x.shape, y.shape)  # (100, 3) (100, 1)
#x = x.reshape(1,10,10,3)
#print(x.shape, y.shape)  # (1, 10, 10, 3) (100, 1)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(3,))
dense1 = Dense(10)(input1)
dense2 = Dense(9, activation='relu')(dense1)
dense3 = Dense(8)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)
'''
________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 3)]               0
_________________________________________________________________
dense (Dense)                (None, 10)                40
_________________________________________________________________
dense_1 (Dense)              (None, 9)                 99
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 80
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 9
=================================================================
Total params: 228
Trainable params: 228
Non-trainable params: 0
_________________________________________________________________

=> 함수형으로 짠 형태의 summary ; input_1 (InputLayer)가 추가되어 나옴

'''
#model = Sequential()
#model.add(Dense(10, input_dim=3))   # (100, 3) -> (N, 3)
#model.add(Dense(10, input_shape=(3,)))   # (100, 3) -> (N, 3)
#model.add(Dense(9))
#model.add(Dense(8))
#model.add(Dense(1))
model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 10)                40
_________________________________________________________________
dense_1 (Dense)              (None, 9)                 99
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 80
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 9
=================================================================
Total params: 228
Trainable params: 228
Non-trainable params: 0
_________________________________________________________________

model.add(Dense(10, input_dim=3))   # (100, 3) -> (N, 3)
model.add(Dense(10, input_shape=(3,)))   # (100, 3) -> (N, 3)
=> 동일한 형태

 - 'Output Shape'에 None은 들어가는 숫자 상관없다는 의미
'''