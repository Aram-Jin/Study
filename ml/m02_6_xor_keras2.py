from unittest import result
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

#2. 모델
# model = LinearSVC()
# model = Perceptron()
# model = SVC()
model = Sequential()
model.add(Dense(128, input_dim=2))
model.add(Dropout(0.2))
model.add(Dense(84))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(48))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dropout(0.2))
model.add(Dense(8))
model.add(Dropout(0.2))
model.add(Dense(4))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가, 예측
y_predict = model.predict(x_data)

results = model.evaluate(x_data, y_data)

print(x_data, "의 예측결과 : ", y_predict)
print("metrics_acc : ", results[1])
 
# acc = accuracy_score(y_data, np.round(y_predict,0))
# print("accuracy_score : ", acc)

'''
다층 신경망을 통하여 인공지능의 겨울을 해결함!!(accuracy 1.0 나오게됨)
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [[0.49229378]
 [0.46680132]
 [0.5140233 ]
 [0.48848817]]
metrics_acc :  0.75
'''