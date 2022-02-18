# [실습] cifar10으로 완성할 것

# vgg trainable : True, False
# Flatten / GAP

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)   # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)     # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape)   # (50000, 10)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)  # (40000, 32, 32, 3) (40000, 10)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 10) 

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()

n = x_train.shape[0]  # 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1)   #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
scaler.fit(x_train_reshape)              
x_train_transform = scaler.fit_transform(x_train_reshape)  #0~255 -> 0~1
x_train = x_train_transform.reshape(x_train.shape)    #--->(50000,32,32,3) 0~1

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)

#x_test_transform = scaler.transform(x_test.reshape(m,-1))
#x_test = x_test_transform.reshape(x_test.shape)


#2. 모델구성
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

# vgg16.summary()
# vgg16.trainable = False    # 가중치를 동결시킨다
# print(vgg16.weights)

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(10))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.0001
optimizer = Adam(lr=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto', verbose=1, factor=0.5) 
# mcp = ModelCheckpoint(monitor='val_accuracy', mode='max', verbose=1,
#                       save_best_only=True, filepath=model_path)   # EarlyStopping의 patience를 넓게 주어야 효과가 좋음. verbose=1은 중간중간 저장될때마다 보여줌

start = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr]) 
end = time.time() - start

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))
print('걸린시간 :', round(end,4))


######################################## 1. vgg trainable : True, Flatten ###############################################
# learning_rate :  0.001
# loss :  9.6112
# accuracy :  0.1011
# 걸린시간 : 1155.9542
######################################## 2. vgg trainable : False, Flatten ###############################################
# learning_rate :  0.001
# loss :  9.6354
# accuracy :  0.1017
# 걸린시간 : 419.3603
######################################## 3. vgg trainable : True, GAP ###############################################
#learning_rate :  0.001
# loss :  4.8193
# accuracy :  0.102
# 걸린시간 : 952.8361
######################################## 4. vgg trainable : False, GAP ###############################################
# learning_rate :  0.001
# loss :  7.9979
# accuracy :  0.1081
# 걸린시간 : 427.1827