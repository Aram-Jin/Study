from sklearn import datasets
from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x)  
print(y)
print(x.shape)  # (506, 13)
print(y.shape)  # (506,)

print(datasets.feature_names)
print(datasets.DESCR)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

### R2 스코어 0.8 이상 만들기, train_size = 0.6~0.8

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7, shuffle=True, random_state=66)


model = Sequential()
model.add(Dense(6, input_dim=13)) 
model.add(Dense(11)) 
model.add(Dense(24)) 
model.add(Dense(50)) 
model.add(Dense(80)) 
model.add(Dense(100)) 
model.add(Dense(80)) 
model.add(Dense(50)) 
model.add(Dense(25)) 
model.add(Dense(12))  
model.add(Dense(5)) 
model.add(Dense(2)) 
model.add(Dense(1)) 

#2. 모델구성
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=500, batch_size=13, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

##################################################  [ N O T E ]  #################################################################
'''
과적합지표 loss 와 val_loss 값 비교 
'''
