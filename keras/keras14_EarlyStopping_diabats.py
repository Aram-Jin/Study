from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn import datasets
from sklearn.datasets import load_boston, load_diabetes
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
# datasets = load_boston()
datasets = load_diabetes()
x = datasets.data
y = datasets.target
'''
print(x)  
print(y)
print(x.shape)  # (442, 10) (442,)
print(y.shape)
'''
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, 
                 validation_split=0.2, callbacks=[es])
end = time.time() - start

print("걸린시간: ", round(end, 3),'초')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

