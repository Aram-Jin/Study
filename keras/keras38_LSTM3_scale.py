import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = np.array([50,60,70])

#print(x.shape, y.shape)     # (13, 3) (13,)

x = x.reshape(13, 3, 1)

# input_shape = (batch_size, timesteps, feature)
# input_shape = (행, 열, 몇개씩 짜르는지!!!)

#2. 모델구성
model = Sequential()
model.add(LSTM(12, input_shape=(3,1), activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
#model.summary()

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')      # optimizer는 loss를 최적화한다
model.fit(x, y, epochs=500, batch_size=1)

#4. 평가, 예측
model.evaluate(x, y)

result = model.predict(x_predict.reshape(1,3,1))
print(result)
