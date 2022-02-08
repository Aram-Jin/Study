from pickletools import optimize
import numpy as np
from sklearn.model_selection import learning_curve

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,7,11,9,6])

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.0000825

# optimizer = Adam(learning_rate=learning_rate)   # learning_rate : Defaults to 0.001.
# loss :  3.1801 lr : 2e-05 결과물 :  [[11.23149]]
# optimizer = Adadelta(learning_rate=learning_rate)  
# loss :  3.3469 lr : 0.001 결과물 :  [[11.108701]]
# optimizer = Adagrad(learning_rate=learning_rate)
# loss :  2.9419 lr : 0.0012 결과물 :  [[10.556614]]  
# optimizer = Adamax(learning_rate=learning_rate)  
# loss :  2.9932 lr : 0.00145 결과물 :  [[11.04709]]
# optimizer = RMSprop(learning_rate=learning_rate) 
# loss :  3.6143 lr : 0.0008225 결과물 :  [[11.580759]] 
# optimizer = SGD(learning_rate=learning_rate)  
# loss :  3.0362 lr : 0.000125 결과물 :  [[10.438867]]
optimizer = Nadam(learning_rate=learning_rate)
# loss :  2.8828 lr : 8.25e-05 결과물 :  [[10.711576]]  

# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer=optimizer)

model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
y_predict = model.predict([11])

print('loss : ', round(loss, 4), 'lr :', learning_rate, '결과물 : ', y_predict)