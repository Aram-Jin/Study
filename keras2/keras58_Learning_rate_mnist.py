import numpy as np
from tensorflow.keras.datasets import fashion_mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)/255.   # 4차원 데이터로 변환시켜야함, 흑백 데이터이므로 1을 넣었지만 (60000,28,14,2) 도 가능
x_test = x_test.reshape(10000, 28, 28, 1)/255.  
# print(x_train.shape)   # (60000, 28, 28, 1)

# print(np.unique(y_train, return_counts=True))
 
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
#print(y.shape)   # (60000, 10)

# print(x_train.shape, y_train.shape)  # (48000, 28, 28, 1) (48000, 10)
# print(x_test.shape, y_test.shape)   # (12000, 28, 28, 1) (12000, 10) 

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

model = Sequential()
model.add(Conv2D(128, kernel_size=(3,3), padding='same', input_shape=(28, 28, 1)))    
model.add(Dropout(0.1)) 
model.add(Conv2D(128, kernel_size=(2,2), activation='relu'))
model.add(Dropout(0.2)) 
model.add(Conv2D(64, kernel_size=(2,2), padding='same', activation='relu'))
model.add(Dropout(0.2)) 
model.add(Conv2D(48, kernel_size=(2,2), activation='relu'))
model.add(MaxPooling2D()) 
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
learning_rate = 0.0001
optimizer = Adam(lr=learning_rate)  # 0.001과 0.0001과 성능 비교
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

import time
start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.25)
end = time.time() - start

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))
print('걸린시간 :', round(end,4))

# learning_rate : 0.001
# loss :  0.3264
# accuracy :  0.9053
# 걸린시간 : 98.1276

# learning_rate :  0.0001
# loss :  0.296
# accuracy :  0.8923
# 걸린시간 : 98.3891

# learning_rate : 0.00001
# loss :  0.4858
# accuracy :  0.8286
# 걸린시간 : 99.7087

# learning_rate :  0.0001
# loss :  0.2685
# accuracy :  0.9036
# 걸린시간 : 142.7215