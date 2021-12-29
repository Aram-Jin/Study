# 훈련데이터 10만개로 증폭
# 완료 후 기존 모델과 비교
# save_dir도 _temp에 넣고
# 증폭데이터는 temp에 저장 후 훈련 끝난 후 결과 보고 삭제

import numpy as np, time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.engine import sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout


#1. 데이터
train_datagen = ImageDataGenerator(rescale=1./255,                       # ImageDataGenerator 정의하기
                                   horizontal_flip=True,                 # train용 이미지데이터는 증폭하여 정의시킨다.
                                #    vertical_flip=True, 
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                #    rotation_range=5, 
                                   zoom_range=0.1, 
                                #    shear_range=0.7, 
                                   fill_mode='nearest')   

test_datagen = ImageDataGenerator(rescale=1./255)   #test데이터는 평가만 할 것이므로 데이터를 증폭시키지 않는다

#D:\_data\Image\brain\train

# train_path = '../_data/Image/brain/train'
# test_path = '../_data/Image/brain/test'

xy_train = train_datagen.flow_from_directory('../_data/Image/brain/train', 
                                             target_size=(150, 150), batch_size=200,
                                             class_mode='binary',
                                             shuffle='True')   #keras에서는 fit에 batchsize를 명시해주었지만 다른곳에서는 train데이터에 지정하고 돌려줌
# Found 160 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory('../_data/Image/brain/test',
                                           target_size=(150, 150),
                                           batch_size=200,
                                           class_mode='binary')
# Found 120 images belonging to 2 classes.

# print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000002B85C344F70>
# print(xy_train[31])       # 마지막배치
# print(xy_train[0][0])     # 첫번쨰 괄호 '0'은 첫번째배치 두번째괄호의 '0'은 첫번쨰 배치의 x값
# print(xy_train[0][1])
# print(xy_train[0][2])   #IndexError: tuple index out of range -> tuple은 수정이 안됨 list는 수정가능
# print(xy_train[0][0].shape, xy_train[0][1].shape)     # (5, 150, 150, 3) (5,)

# print(type(xy_train))  # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))   # <class 'tuple'>
# print(type(xy_train[0][0]))   # <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))   # <class 'numpy.ndarray'>

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

# print(x_train.shape, y_train.shape)  # (160, 150, 150, 3) (160,)
# print(x_test.shape, y_test.shape)  # (120, 150, 150, 3) (120,)

augment_size = 40   
randidx = np.random.randint(x_train.shape[0], size=augment_size)   # randint : 랜덤한 정수값을 생성하겠다. 
# print(x_train.shape[0])   # 160
# print(randidx)  
# print(np.min(randidx), np.max(randidx))   # 4 149
# print(randidx.shape)   # (40,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
# print(x_augmented.shape)   # (40, 150, 150, 3)
# print(y_augmented.shape)   # (40,)
                    

x_train = np.concatenate((x_train, x_augmented))   # concatenate를 사용할때는 (())괄호 두개로 사용해주어야함
y_train = np.concatenate((y_train, y_augmented))
# print(x_train.shape, y_train.shape)   # (200, 150, 150, 3) (200,)
# print(x_test.shape, y_test.shape)   # (120, 150, 150, 3) (120,)


#2. 모델구성
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape = (150,150,3))) 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es])  # steps_per_epochs -> 전체데이터/batchsize = 160/5=32
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss[0])
print('accuracy:', loss[1])

y_predict = model.predict(x_test)

'''
걸린시간 :  14.813 초
loss: 0.9678577184677124
accuracy: 0.7250000238418579
'''

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.plot(hist.history['acc'], marker='.', c='black', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='green', label='val_acc')
plt.xlabel('epoch')
plt.grid()
plt.legend(loc='upper right')
plt.show()
