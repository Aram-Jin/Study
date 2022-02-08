### TensorBoard : TensorFlow 시각화 도구

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

import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto', verbose=1, factor=0.5) 
tb = TensorBoard(log_dir='./save/_graph', histogram_freq=0, write_graph=True, write_images=True)   # 중요한 것은 경로(log_dir)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=80, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr, tb])
end = time.time() - start

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))
print('걸린시간 :', round(end,4))

# learning_rate :  0.001
# loss :  0.096
# accuracy :  0.9751
# 걸린시간 : 319.5539
  
######################################### 시각화 ########################################
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

"""
> cmd 창에서 TensorBoard의 경로로 들어가서 웹주소 찾기

C:\Users\bitcamp>d:

D:\>cd study

D:\Study>cd save

D:\Study\save>cd _graph

D:\Study\save\_graph>dir/w
 D 드라이브의 볼륨: 새 볼륨
 볼륨 일련 번호: D288-0105

 D:\Study\save\_graph 디렉터리

[.]          [..]         [train]      [validation]
               0개 파일                   0 바이트
               4개 디렉터리  940,964,143,104 바이트 남음

D:\Study\save\_graph>tensorboard --logdir=.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.7.0 at http://localhost:6006/ (Press CTRL+C to quit)

"""