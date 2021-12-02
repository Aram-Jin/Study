'''
활성화 함수(Activation) : layer 에서 다음 layer로 전달할때 값을 한정하는 역할
https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=handuelly&logNo=221824080339


'''
# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])
# 위 데이터를 훈련하여 epochs=50 일때 hidden layer구성을 짜서 예측값 '4' 도출시키기

#2. 모델구성 -> 'hidden layer'
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.summary()
'''
< summary >
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================   ex)
dense (Dense)                (None, 5)                 10        -> parameter가 10개인 이유는 연산할때 bias도 같이 연산해주어야함
_________________________________________________________________   input(1) => output(5) :y=w1x1 + w1x2 + ... + w1x5 + b(x1 + x2 + ... + x5) => 총10개
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________ 
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 10
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 3
=================================================================
Total params: 57                         => 결론적으로 Total params의 갯수는 params당 input하는 x값만 연산해주는 것이 아니라 bias도 연산을 해주어야함
Trainable params: 57
Non-trainable params: 0
_________________________________________________________________


* hyperparameter tuning 시 summary 구조 이해가 중요!!!
'''



'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=50, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([4])
print('4의 예측값 : ', result)
'''
'''
loss :  4.3297473894199356e-05
4의 예측값 :  [[3.9863973]]
'''


### [ NOTE ]
'''
#1. 데이터 -> input layer
#2. 모델구성 -> hidden layer

* hyperparameter tuning
'''