# 실습
# 아까 4가지 모델로 만들어보기
# 784개 DNN으로 만든것(최사의 성능인 것//0.978)과 비교!!
# time check! -> fit 에서만

import numpy as np
import time
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

# print(x_train.shape, x_test.shape)   # (60000, 784) (10000, 784)

x = np.append(x_train, x_test, axis=0)
# print(x.shape)   # (70000, 784)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

pca = PCA(n_components=154)
x = pca.fit_transform(x)
# print(x)
# print(x.shape)  

pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)
# print(cumsum)

# print(np.argmax(cumsum >= 0.95) + 1)  # 154
# print(np.argmax(cumsum >= 0.99) + 1)  # 331
# print(np.argmax(cumsum >= 0.999) + 1)  # 486
# print(np.argmax(cumsum == 1.0) + 1)  # 1

# print(np.argmax(cumsum) +1)  # 713


#2. 모델구성
model = Sequential()
model.add(Dense(128, input_shape=(28*28, )))
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
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics는 평가지표에 의한 값이 어떤모양으로 돌아가는지 출력하여 보여줌(출력된 loss의 두번째값)

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=80, verbose=1, validation_split=0.2, callbacks=[es])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])
print('걸린시간 : ', end - start)

# scaler = MinMaxScaler()
# #scaler = StandardScaler()
# # scaler = RobustScaler()
# # scaler = MaxAbsScaler()

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])  
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]) 
# #print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)  --> (60000,28*28)
# #print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)  -->  (10000,28*28)  

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)



'''
1. 나의 최고 DNN
time = ???
acc = 0.9621999859809875

2. 나의 최고 CNN
time = ???
acc = ???

3. PCA 0.95  --> ing
time = ???
acc = ???

4. PCA 0.99
time =  822.6579537391663
acc = 0.9609000086784363

5. PCA 0.999
time = ???
acc = ???

6. PCA 1.0
time = ???
acc = ???
'''


