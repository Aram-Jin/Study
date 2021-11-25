from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(100))      #(0~99)
y = np.array(range(1, 101))   #(1~100)

# train과 test비율을 7:3으로 분리하시오       # 싸이킷런

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7, shuffle=True, random_state=66)

print(x_train)  
print(x_test)  # [ 8 93  4  5 52 41  0 73 88 68]
print(y_train)
print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1)) 
model.add(Dense(80)) 
model.add(Dense(60)) 
model.add(Dense(40)) 
model.add(Dense(20)) 
model.add(Dense(10)) 
model.add(Dense(5)) 
model.add(Dense(4)) 
model.add(Dense(3)) 
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10000, batch_size=20)   

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)
result_= model.predict([101])
print('101의 예측값 : ',result_)




'''
from numpy import random
x_train = random.randint(low=100, size=70) 
x_test = random.randint(low=100, size=30)

y_train = random.randint(low=1, high=101, size=70) 
y_test = random.randint(low=1, high=101, size=30)

print(x_train, x_test)
print(y_train, y_test)
'''
