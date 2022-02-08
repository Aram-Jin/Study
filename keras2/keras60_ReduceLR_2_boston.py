from sklearn import datasets
from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target

# print(x)  # 단위 구분 ex) 6.3200e-03 -> 'e-03' 은 소숫점 3자리(0.000x) 'e+03' 은 반대로 
# print(y)
# print(x.shape)  # (506, 13)
# print(y.shape)  # (506,)

# print(datasets.feature_names)
# print(datasets.DESCR)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

### R2 스코어 0.8 이상 만들기, train_size = 0.6~0.8

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7, shuffle=True, random_state=66)

#2. 모델구성
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

#3. 컴파일, 훈현
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.001
optimizer = Adam(lr=learning_rate)
model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])   # loss에 들어가는 지표는 성능에 영향을 미치지만, metrics에 들어가는 지표는 성능에 영향을 미치지 않음(단순평가지표)

import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto', verbose=1, factor=0.5) 
 
start = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=13, callbacks=[es, reduce_lr])
end = time.time() - start

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('mae : ', round(mae,4))
print('걸린시간 :', round(end,4))

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# learning_rate :  0.001
# loss :  18.8911
# mae :  3.3795
# 걸린시간 : 82.0666
# r2스코어 :  0.7713411930351318

'''
loss : 19.184654235839844
r2스코어 :  0.7677883338121207
'''

'''
loss : 17.076885223388672
r2스코어 :  0.7933008109604748
'''

'''
loss : 19.80939292907715
r2스코어 :  0.7602264967945693
'''