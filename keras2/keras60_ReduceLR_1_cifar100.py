from functools import reduce
from tensorflow.keras.datasets import cifar100
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape, y_train.shape)   # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)     # (10000, 32, 32, 3) (10000, 1)
# print(np.unique(y_train, return_counts=True))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape)   # (50000, 10)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape, y_train.shape)  # (40000, 32, 32, 3) (40000, 10)
# print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 10) 

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()

n = x_train.shape[0]
x_train_reshape = x_train.reshape(n,-1) 
x_train_transform = scaler.fit_transform(x_train_reshape)
x_train = x_train_transform.reshape(x_train.shape) 

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)

#x_test_transform = scaler.transform(x_test.reshape(m,-1))
#x_test = x_test_transform.reshape(x_test.shape)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

model = Sequential()
model.add(Conv2D(128, kernel_size=(2,2), padding='same', strides=1, input_shape=(32, 32, 3)))  
model.add(MaxPooling2D())  
model.add(Conv2D(128, (2,2), activation='relu'))
model.add(MaxPooling2D())   
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D()) 
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.00125
optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto', verbose=1, factor=0.5)  #-> 5번 만에 갱신이 안된다면 (factor=0.5) LR을 50%로 줄이겠다

start = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr]) 
end = time.time() - start

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))
print('걸린시간 :', round(end,4))

# ReduceLROnPlateau -> 다 죽어가는데 포도당 찔러주는 역활 learning_rate를 줄여줌

# learning_rate :  0.1
# loss :  4.6079
# accuracy :  0.008
# 걸린시간 : 146.7234

# learning_rate :  0.1
# loss :  4.6079
# accuracy :  0.0102
# 걸린시간 : 368.6833

# learning_rate :  0.0015
# loss :  3.5073
# accuracy :  0.2807
# 걸린시간 : 163.3952

# learning_rate :  0.0015
# loss :  2.7438
# accuracy :  0.3409
# 걸린시간 : 266.9761