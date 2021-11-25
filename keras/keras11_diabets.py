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
model.add(Dense(500, input_dim=10)) 
model.add(Dense(450)) 
model.add(Dense(400)) 
model.add(Dense(350)) 
model.add(Dense(300)) 
model.add(Dense(250)) 
model.add(Dense(200)) 
model.add(Dense(150)) 
model.add(Dense(100)) 
model.add(Dense(50)) 
model.add(Dense(25)) 
model.add(Dense(20)) 
model.add(Dense(10)) 
model.add(Dense(5)) 
model.add(Dense(1)) 

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=300, batch_size=10, verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


'''
loss : 2084.333740234375
r2스코어 :  0.6088191556103872
'''