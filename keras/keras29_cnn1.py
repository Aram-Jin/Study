from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(10, 10, 1)))    # 9,9,10
model.add(Conv2D(5, (3,3), activation='relu'))                       # 7,7,5                                             
model.add(Conv2D(7, (2,2), activation='relu'))                       # 6,6,7                                       
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(5, activation='softmax'))

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50    --------> 
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 7, 7, 5)           455
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 6, 6, 7)           147
_________________________________________________________________
flatten (Flatten)            (None, 252)               0
_________________________________________________________________
dense (Dense)                (None, 64)                16192
_________________________________________________________________
dropout (Dropout)            (None, 64)                0
_________________________________________________________________
dense_1 (Dense)              (None, 16)                1040
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 85
=================================================================
Total params: 17,969
Trainable params: 17,969
Non-trainable params: 0
_________________________________________________________________

'''


# 모나리자, 아람, 철수, 영희, 민수 

# Conv2D: 가로,세로 5사이즈 흑백컬러의 데이터 (데이터의 차원은 4차원, input 데이터는 3차원)
# kernel_size : 조각내려는 사이즈