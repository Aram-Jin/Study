import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터

#x = np.array(1,2,3,4,5,6,7)
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])     

y = np.array([5,6,7])                             

#print(x.shape, y.shape)    # (4, 3) (3,)
        
x = x.reshape(4,3,1)

#2. 모델구성
model = Sequential()
model.add(GRU(5,input_shape=(3,1)))         
model.add(Dense(5))               
model.add(Dense(2))                 
model.add(Dense(1))                         

model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #      왜 390개가 나올까? 왜 360이 아니지?
=================================================================
gru (GRU)                    (None, 10)                390          3( N * ( N + f + 1 + 1)) 왜 1이 1개가 더 추가되지 왜 why 뭐때문에?
_________________________________________________________________   
dense (Dense)                (None, 2)                 22           
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 3
=================================================================
Total params: 415
Trainable params: 415
Non-trainable params: 0
_________________________________________________________________
'''


'''
#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam') #mae도있다.
es = EarlyStopping(monitor="loss", patience=500, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x,y, epochs=10000, batch_size=1, callbacks=[es])  

#4. 평가, 예측

model.evaluate(x,y)
y_pred = np.array([5,6,7,8]).reshape(1,2,2)
result = model.predict(y_pred)  
print(result)

'''