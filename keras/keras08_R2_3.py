from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1.데이터                                                        # 선형회귀 -> 상대지표
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

# x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                   # train_size=0.7, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print('r2스코어 : ', r2)

# plt.scatter(x,y)   # 점 찍기
# plt.plot(x, y_predict, color='red')   # 선 긋기
# plt.show()

'''
loss : 0.3800012469291687
r2스코어 :  0.8099993487333605
'''