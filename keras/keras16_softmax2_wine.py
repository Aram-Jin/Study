import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine

datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (178, 13) (178,)
print(y)
print(np.unique(y))  # [0 1 2]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)  # (178, 3)
  
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)  # (142, 13) (142, 3)
print(x_test.shape, y_test.shape)  # (36, 13) (36, 3)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30, activation='linear', input_dim=13))
model.add(Dense(20))
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

result_recover = np.argmax(result, asix=1).reshape(-1,1)
print


'''
loss :  0.2917698323726654
accuracy :  0.9444444179534912
[[0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[[1.7767713e-09 4.0107119e-05 9.9995995e-01]
 [4.6818294e-02 9.5261502e-01 5.6668493e-04]
 [7.1907732e-12 9.9999976e-01 2.6221713e-07]
 [1.0000000e+00 8.9038066e-09 6.6523307e-09]
 [5.8457501e-09 9.9990189e-01 9.8070690e-05]
 [1.5060011e-11 9.9999774e-01 2.3108171e-06]
 [4.8801120e-02 4.0519444e-05 9.5115834e-01]]
'''
