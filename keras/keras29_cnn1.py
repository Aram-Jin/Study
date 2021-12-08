from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), strides=1,
                 padding='same', input_shape=(10, 10, 1)))    # 9,9,10     필터:10  채널:10,10  // padding='same' : shape 맞추는데 사용, padding='valid'는 디폴트값
model.add(MaxPooling2D())
#model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(5, (2,2), activation='relu'))                 # 7,7,5                                             
#model.add(Conv2D(7, (2,2), activation='relu'))                # 6,6,7                                       
#model.add(Flatten())
#model.add(Dense(64))
#model.add(Dropout(0.2))
#model.add(Dense(16))
#model.add(Dense(5, activation='softmax'))

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50    -------->   커널사이즈=패널   2x2x1(입력데이터의채널값)x10(출력채널)+10 = 50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 7, 7, 5)           455               { 3x3x1(3,3패널의 파라미터값)x1(바이어스값) } x 10(입력채널-전레이어의출력채널) x5
_________________________________________________________________        (출력채널)+5(출력채널만큼의바이어스값)  = 455
conv2d_2 (Conv2D)            (None, 6, 6, 7)           147               4 x 5 x 7 + 7 = 147
_________________________________________________________________
flatten (Flatten)            (None, 252)               0
_________________________________________________________________
dense (Dense)                (None, 64)                16192              252 * 64 + 4 = 16192
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

# Conv2D: 가로,세로 5사이즈 흑백컬러의 데이터 (데이터의 차원은 4차원, input 데이터는 3차원)
# kernel_size : 조각내려는 사이즈

# padding='same' shape 맞추는데 사용됨. padding='valid'는 원래 연산대로되는 디폴트값
# strides : 걷는 크기(건너서 시작하는 숫자), kernel_size보다 크게잡으면 데이터 손실. 디폴트값'1' but 성능이 좋아지는지는 모르겠음
# MaxPooling : pool로 지정한 픽셀 중 특정값만 뽑아서 나타내주는 것. pool_size=2 -> 디폴트값 
#              DNN의 dropout과 유사함 -> 성능향상에 도움


# https://dacon.io/competitions/official/235594/codeshare/2869
# https://rubber-tree.tistory.com/entry/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%AA%A8%EB%8D%B8-CNN-Convolutional-Neural-Network-%EC%84%A4%EB%AA%85