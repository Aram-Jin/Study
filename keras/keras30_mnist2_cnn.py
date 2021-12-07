import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)   # 4차원 데이터로 변환시켜야함, 흑백 데이터이므로 1을 넣었지만 (60000,28,14,2) 도 가능
x_test = x_test.reshape(10000, 28, 28, 1)   
print(x_train.shape)   # (60000, 28, 28, 1)

print(np.unique(y_train, return_counts=True))
 
x = x_train

from tensorflow.keras.utils import to_categorical
y = to_categorical(y_train)
#print(y.shape)   # (60000, 10)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)  # (48000, 28, 28, 1) (48000, 10)
print(x_test.shape, y_test.shape)   # (12000, 28, 28, 1) (12000, 10) 

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(20, kernel_size=(3,3), input_shape=(28, 28, 1)))    
model.add(Conv2D(40, (2,2), activation='relu'))                                                                  
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics는 평가지표에 의한 값이 어떤모양으로 돌아가는지 출력하여 보여줌(출력된 loss의 두번째값)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=10, verbose=1, validation_split=0.2, callbacks=[es]) # callbacks의 []는 다른게 들어갈수 있음

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])

'''
768/768 [==============================] - 34s 45ms/step - loss: 0.0289 - accuracy: 0.9909 - val_loss: 0.1502 - val_accuracy: 0.9789
Restoring model weights from the end of the best epoch.
Epoch 00024: early stopping
375/375 [==============================] - 3s 7ms/step - loss: 0.0897 - accuracy: 0.9743
loss :  0.08968890458345413
accuracy :  0.9742500185966492

Epoch 00084: early stopping
375/375 [==============================] - 2s 4ms/step - loss: 0.1367 - accuracy: 0.9647
loss :  0.1367303729057312
accuracy :  0.9646666646003723


Restoring model weights from the end of the best epoch.
Epoch 00104: early stopping
375/375 [==============================] - 2s 4ms/step - loss: 0.1206 - accuracy: 0.9704
loss :  0.12055526673793793
accuracy :  0.9704166650772095

Restoring model weights from the end of the best epoch.
Epoch 00108: early stopping
375/375 [==============================] - 1s 3ms/step - loss: 0.1168 - accuracy: 0.9711
loss :  0.11677785217761993
accuracy :  0.9710833430290222

# 트레인 발리데이션 테스트 평가
# 평가지표 accuracy 0.98 이상

#print(x_train[0])
#print('y_train[0]번째의 값 : ', y_train[0])

#import matplotlib.pyplot as plt
#plt.imshow(x_train[0], 'gray')
#plt.show()
'''