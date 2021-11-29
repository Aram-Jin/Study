import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

## "데이터 배치가 잘못 되었을 경우, 올바르게 하이퍼 파라미터 튜닝하기" ## 

#1. 데이터                

x = np.array([[1,2,3,4,5,6,7,8,9,10], [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]])
y = np.array([11,12,13,14,15,16,17,18,19,20])

# x = x.reshape(10,2) -> 순서가 바뀌지 않음
# x = np.transpose(x)   
# x = x.T 

print(x.shape) #(2,10)
x = np.transpose(x)
print(x.shape)
print(x)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=2))   # dim 은 열의 갯수 넣기(열/특성/피쳐/컬럼)
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=50, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ',loss)
y_predict = model.predict([[10,1.3]])
print('[10,1.3]의 예측값 :', y_predict)

'''
# 다른 방법
print(x.T)
print(x.T.shape)

model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x.T, y, epochs=500, batch_size=1)

loss = model.evaluate(x.T,y)
print('loss : ',loss)
y_predict = model.predict([[10,1.3]])
print('[10,1.3]의 예측값 :', y_predict)
'''


### [ NOTE ]
'''
1) MLP (multi-layer perceptron)
다층 퍼셉트론(MPL)은 퍼셉트론으로 이루어진 층(layer) 여러 개를 순차적으로 붙여놓은 형태
퍼셉트론은 뇌의 신경계를 구성하는 뉴런과 같은 형태(입력한 값에 따른 결과값을 도출하는 형태)

* 열 우선, 행 무시!!! 

2) Shape 변경 CASE A : 전치변환하기 

# x = np.transpose(x)   
# x = x.T 

3) Shape 변경 CASE B : 순서가 바뀌지 않음

# x = x.reshape(10,2) 
'''