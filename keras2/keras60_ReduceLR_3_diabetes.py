from sklearn import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

'''
print(x)
print(y)
print(x.shape, y.shape)  # (442, 10) (442,)

print(datasets.feature_names)
print(datasets.DESCR)
'''
### R2
# 0.62 이상이상

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=49)

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=10))
model.add(Dense(15))
model.add(Dense(14))
model.add(Dense(13))
model.add(Dense(12))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련 
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.001
optimizer = Adam(lr=learning_rate)

model.compile(loss='mse', optimizer=optimizer)

import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto', verbose=1, factor=0.5) 

start = time.time() 
model.fit(x_train, y_train, epochs=30, batch_size=10, verbose=1, callbacks=[es, reduce_lr])
end = time.time() - start

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('걸린시간 :', round(end,4))

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# learning_rate :  0.001
# loss :  2123.7446
# 걸린시간 : 7.7723
# r2스코어 :  0.6014225603384131

'''
loss : 2084.333740234375
r2스코어 :  0.6088191556103872
'''