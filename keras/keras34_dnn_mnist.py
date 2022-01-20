import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

scaler = MinMaxScaler()
#scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])  
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]) 
#print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)  --> (60000,28*28)
#print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)  -->  (10000,28*28)  
  
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# x_train = x_train.reshape(60000, 28, 28)
# x_test = x_test.reshape(10000, 28, 28)


#2. 모델구성
model = Sequential()
model.add(Dense(128, input_shape=(28*28, )))
#model.add(Dense(64, input_shape=(784, )))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
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

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=80, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])


'''
loss :  0.23126453161239624
accuracy :  0.9621999859809875
'''

