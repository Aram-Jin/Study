# 모델링 구성!!

from numpy.random import rand
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

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

x_augmented = train_datagen.flow(x_augmented, y_augmented, #np.zeros(augument_size),
                                  batch_size=augment_size, shuffle=False).next()[0]

print(x_augmented)
print(x_augmented.shape)   # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))   # concatenate를 사용할때는 (())괄호 두개로 사용해주어야함
y_train = np.concatenate((y_train, y_augmented))

print(x_train)
print(x_train.shape, y_train.shape)   # (100000, 28, 28, 1) (100000,) -> 기존 60000개에서 랜덤하게 추출한 40000개를 붙여줌

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

model = Sequential() 
model.add(Conv2D(7, kernel_size = (3,3), input_shape = (28,28,1))) 
model.add(Conv2D(5, (3,3), activation='relu'))                     
model.add(Dropout(0.2))
model.add(Conv2D(4, (2,2), activation='relu'))         
model.add(Flatten()) 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto',
                   verbose=1, restore_best_weights=False) # restore_best_weights=True : 최종값 이전에 가장 좋았던 값 도출함
# mcp = ModelCheckpoint (monitor = 'val_acc', mode = 'min', verbose = 1, save_best_only=True,
#                        filepath = './_ModelCheckPoint/keras27_5_MCP.hdf5')
model.fit(x_train, y_train, epochs=16, batch_size=64,
          validation_split=0.1111, callbacks=[es])#,mcp

# model.save('./_save/keras30_2_save_model.h5')

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss[0])
print('accuracy:', loss[1])

y_predict = model.predict(x_test)


from sklearn.metrics import accuracy_score
y_test = np.argmax(y_test, axis=-1)
y_predict = np.argmax(y_predict, axis=-1)
accuracy = accuracy_score(y_test, y_predict)
print('accuracy 스코어:', accuracy)

'''
loss: 0.3467070460319519
accuracy: 0.8819000124931335
accuracy 스코어: 0.8819
'''