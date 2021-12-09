import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])  
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])   
#print(x_train.shape)   # (60000, 784)

#print(np.unique(y_train, return_counts=True))   # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#print(x_train.shape, y_train.shape)   # (60000, 784) (60000, 10)
#print(x_test.shape, y_test.shape)     # (10000, 784) (10000, 10)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=784, activation='relu'))
model.add(Dropout(0.2))  
model.add(Dense(32, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(16, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics는 평가지표에 의한 값이 어떤모양으로 돌아가는지 출력하여 보여줌(출력된 loss의 두번째값)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
###################################################################################################################################
import datetime
date = datetime.datetime.now()
datetime_spot = date.strftime("%m%d_%H%M")  # 1206_0456
#print(datetime_spot)   
filepath = './_ModelCheckPoint/'                 
filename = '{epoch:04d}-{val_accuracy:.4f}.hdf5'     
model_path = "".join([filepath, 'k34_', datetime_spot, '_', filename])
####################################################################################################################################
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_accuracy', mode='max', verbose=1,
                      save_best_only=True, filepath=model_path)   # EarlyStopping의 patience를 넓게 주어야 효과가 좋음. verbose=1은 중간중간 저장될때마다 보여줌

model.fit(x_train, y_train, epochs=5000, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es, mcp]) # callbacks의 []는 다른게 들어갈수 있음


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])


'''
Epoch 00137: val_accuracy did not improve from 0.79700
Epoch 00137: early stopping
313/313 [==============================] - 0s 851us/step - loss: 0.6849 - accuracy: 0.7716
loss :  0.6848528385162354
accuracy :  0.7716000080108643
'''