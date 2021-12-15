import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU
from tensorflow.keras.callbacks import EarlyStopping


#1. 데이터

x = np.array([   [1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]   ])     

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])                             
        
x = x.reshape(13,3,1)

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(10, input_shape=(3,1), return_sequences=True, activation='relu'))       # (N,3,1) -> (N,10)
model.add(LSTM(10, return_sequences=True, activation='relu'))
model.add(GRU(10, return_sequences=False, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
    
model.summary()                    


#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam') #mae도있다.
es = EarlyStopping(monitor="loss", patience=500, mode='min',verbose=2,baseline=None, restore_best_weights=True)

model.fit(x,y, epochs=10000, batch_size=1, callbacks=[es])  


#4. 평가, 예측

model.evaluate(x,y)

y_pred = np.array([50,60,70]).reshape(1,3,1)

result = model.predict(y_pred)  

print(result)

# Dropout을 하면 더 낮아짐. relu는 좋아짐 왜??