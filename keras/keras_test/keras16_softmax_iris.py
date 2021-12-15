import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (150, 4) (150,)
#print(y)
#print(np.unique(y))  # [0 1 2]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)  # (150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)  # (120, 4) (120, 3)
print(x_test.shape, y_test.shape)  # (30, 4) (30, 3)


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(50, activation='linear', input_dim=4))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics는 평가지표에 의한 값이 어떤모양으로 돌아가는지 출력하여 보여줌(출력된 loss의 두번째값)

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es]) # callbacks의 []는 다른게 들어갈수 있음


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])

resulte = model.predict(x_test[:7])
print(y_test[:7])
print(resulte)

'''
loss :  0.08398345112800598
accuracy :  0.9666666388511658
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[3.80378362e-04 9.97372985e-01 2.24664481e-03]
 [2.41369562e-05 9.83705103e-01 1.62708256e-02]
 [4.62491116e-05 9.73074853e-01 2.68788747e-02]
 [9.99884486e-01 1.15442803e-04 2.61055182e-18]
 [1.04987004e-04 9.96569395e-01 3.32560320e-03]
 [1.02414552e-03 9.96181369e-01 2.79448438e-03]
 [9.99911308e-01 8.87414571e-05 5.69686071e-18]]
'''

