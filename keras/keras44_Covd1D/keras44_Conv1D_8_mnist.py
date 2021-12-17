import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)  
#print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)  
  
#print(np.unique(y_train, return_counts=True))   # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(28,28)))  
model.add(Flatten()) 
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=80, verbose=1, validation_split=0.2, callbacks=[es]) 

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])


'''
loss :  0.61527419090271
accuracy :  0.8230000138282776


loss :  0.4677174687385559, accuracy :  0.9248999953269958\n
'''
