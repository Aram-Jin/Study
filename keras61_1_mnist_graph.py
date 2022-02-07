from cProfile import label
import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)   
x_test = x_test.reshape(10000, 28, 28, 1)   
print(x_train.shape)   # (60000, 28, 28, 1)

print(np.unique(y_train, return_counts=True))

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
#print(y.shape)   # (60000, 10)

print(x_train.shape, y_train.shape)  # (48000, 28, 28, 1) (48000, 10)
print(x_test.shape, y_test.shape)   # (12000, 28, 28, 1) (12000, 10) 

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(10, kernel_size=(3,3), input_shape=(28, 28, 1)))    
model.add(Conv2D(10, (2,2), activation='relu'))                                                                  
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.001
optimizer = Adam(lr=learning_rate)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc']) 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto', verbose=1, factor=0.5) 

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=80, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr])
end = time.time() - start

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))
print('걸린시간 :', round(end,4))

# learning_rate :  0.001
# loss :  0.101
# accuracy :  0.9763
# 걸린시간 : 261.2944
  
######################################### 시각화 ##################################
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])

plt.show()
