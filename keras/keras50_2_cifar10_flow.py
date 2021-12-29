# 훈련데이터 10만개로 증폭
# 완료 후 기존 모델과 비교
# save_dir도 _temp에 넣고
# 증폭데이터는 temp에 저장 후 훈련 끝난 후 결과 보고 삭제

from numpy.random import rand
from tensorflow.keras import models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np, time

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(rescale=1./255,                       # ImageDataGenerator 정의하기
                                   horizontal_flip=True,                 # train용 이미지데이터는 변형하여 정의시킨다.
                                #    vertical_flip=True, 
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                #    rotation_range=5, 
                                   zoom_range=0.1, 
                                #    shear_range=0.7, 
                                   fill_mode='nearest')  

test_datagen = ImageDataGenerator(rescale=1./255)   

# print(x_train.shape, x_test.shape)   # (50000, 32, 32, 3) (10000, 32, 32, 3)

augment_size = 50000  
randidx = np.random.randint(x_train.shape[0], size=augment_size)   # randint : 랜덤한 정수값을 생성하겠다. (여기선 0~50000개중 50000개 추출)
# print(x_train.shape[0])   # 50000
# print(randidx)   # [35219 10043  8195 ...  8487 48820 38435]  => 형태는 list형태로 되어있음
# print(np.min(randidx), np.max(randidx))   # 1 49999
# print(randidx.shape)   # (50000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
# print(x_augmented.shape)   # (50000, 32, 32, 3)
# print(y_augmented.shape)   # (50000, 1)

x_train = np.concatenate((x_train, x_augmented))   # concatenate를 사용할때는 (())괄호 두개로 사용해주어야함
y_train = np.concatenate((y_train, y_augmented))
# print(x_train.shape, y_train.shape)   # (100000, 32, 32, 3) (100000, 1)

xy_train = train_datagen.flow(x_train, y_train, 
                                  batch_size=100,   #augment_size, 
                                  shuffle=False)#.next()

xy_test = test_datagen.flow(x_test, y_test, 
                                  batch_size=100,   #augment_size, 
                                  shuffle=False)#.next()
# print(xy_train)
# print(xy_train[0].shape, xy_train[1].shape)  # AttributeError: 'tuple' object has no attribute 'shape'


#2. 모델구성
model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), padding='same', strides=1, input_shape=(32, 32, 3)))  
model.add(MaxPooling2D())  
model.add(Conv2D(5, (2,2), activation='relu'))
model.add(MaxPooling2D())   
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(len(xy_train))   #1250
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

start = time.time()
model.fit_generator(xy_train, epochs=10, steps_per_epoch=len(xy_train), validation_data=xy_test, callbacks=[es])
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')


#4. 평가, 예측
loss = model.evaluate_generator(xy_test, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)

print('loss:', loss[0])
print('accuracy:', loss[1])


# '''
# 걸린시간 :  3631.886 초
# loss: 1.1852543354034424
# accuracy: 0.10010000318288803
# '''