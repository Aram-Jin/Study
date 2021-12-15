import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10], [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3], [10,9,8,7,6,5,4,3,2,1]])
y = np.array([11,12,13,14,15,16,17,18,19,20])

# 시작!! 'x데이터를 잘못 전달해준 경우 칼럼이 10개가 아니라 3개일때 결과값 구하기')
# [[10, 1.3, 1]] 결과값 예측 

print(x.shape)
print(x)

x = np.transpose(x)
print(x.shape)
print(x)

#2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim=3)) 
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ',loss)
y_predict = model.predict([[10, 1.3, 1]])
print('[[10, 1.3, 1]]의 예측값 :', y_predict)

'''
loss :  0.0012446042383089662
[[10, 1.3, 1]]의 예측값 : [[20.027414]]
'''


