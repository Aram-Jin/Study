# 훈련데이터 10만개로 증폭
# 완료 후 기존 모델과 비교
# save_dir도 _temp에 넣고
# 증폭데이터는 temp에 저장 후 훈련 끝난 후 결과 보고 삭제

from numpy.random import rand
from tensorflow.keras import models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
# import warnings
# warnings.filterwarnings('ignore')

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(rescale=1./255,                       # ImageDataGenerator 정의하기
                                   horizontal_flip=True,                 # train용 이미지데이터는 변형하여 정의시킨다.
                                #    vertical_flip=True, 
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                #    rotation_range=5, 
                                   zoom_range=0.1, 
                                #    shear_range=0.7, 
                                   fill_mode='nearest')   # 1./255 minmax scale 지정 0~1사이로(사진에선 최대값이 255이므로), horizontal_flip은 좌우반전, vertical_flip은 상하반전, width_shift_range는 이미지 이동해도 인식(이미지증폭)

test_datagen = ImageDataGenerator(rescale=1./255)   

augment_size = 40000  
randidx = np.random.randint(x_train.shape[0], size=augment_size)   # randint : 랜덤한 정수값을 생성하겠다. (여기선 0~60000개중 40000개 추출)
print(x_train.shape[0])   # 60000
print(randidx)   # [ 6446 39645 27996 ...  8500 42272 46455]  => 형태는 list형태로 되어있음
print(np.min(randidx), np.max(randidx))   # 0 59999
print(randidx.shape)   # (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)   # (40000, 28, 28)
print(y_augmented.shape)   # (40000,)

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                    x_augmented.shape[1],
                                    x_augmented.shape[2], 1)
                                    
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

xy_train = train_datagen.flow(x_augmented, y_augmented, 
                                  batch_size=32,   #augment_size, 
                                  shuffle=False)#.next()

# print(xy_train)
# print(xy_train[0].shape, xy_train[1].shape)   # (40000, 28, 28, 1) (40000,)  -> generator를 통과한 x,y를 flow를 통해 tuple형태로 바꾸어준걸 확인가능

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28,28,1)))
model.add(Conv2D(64, (2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(len(xy_train))   #1250
model.fit_generator(xy_train, epochs=10, steps_per_epoch=len(xy_train))

#4. 평가, 예측
xy_test = test_datagen.flow(x_test, y_test, 
                                  batch_size=32,   #augment_size, 
                                  shuffle=False)#.next()

loss = model.evaluate_generator(xy_test, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)

print('loss:', loss[0])
print('accuracy:', loss[1])

'''
loss: 0.762713611125946
accuracy: 0.7279999852180481
'''
