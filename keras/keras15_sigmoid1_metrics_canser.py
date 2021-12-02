<<<<<<< HEAD
import numpy as np
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (569, 30) (569,)

print(y)
# print(y[:10])
print(np.unique(y)) # [0 1] -> 분류값에서 고정이되는 값들

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

# print(y_test[:11])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(50, activation='linear', input_dim=30))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics는 평가지표에 의한 값이 어떤모양으로 돌아가는지 출력하여 보여줌(출력된 loss의 두번째값)

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es]) # callbacks의 []는 다른게 들어갈수 있음


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

resulte = model.predict(x_test[:31])
print(y_test[:31])
print(resulte)

'''
364/364 [==============================] - 0s 635us/step - loss: 0.2439 - accuracy: 0.9093 - val_loss: 0.3178 - val_accuracy: 0.9121
Restoring model weights from the end of the best epoch.
Epoch 00039: early stopping
4/4 [==============================] - 0s 0s/step - loss: 0.2379 - accuracy: 0.9211
loss :  [0.23793119192123413, 0.9210526347160339]
[1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 1 0 1 0 0 1 0 1 0 1 1 0 1 1 1 1]
'''

##################################################  [ N O T E ]  #################################################################
''' sigmoid, accracy
[평가지표 'metrics']
evaluate 했을때 출력되는 list의 첫번째 값은 'loss'값, 두번째 값은 'metrics'의 평가값
모델 가중치의 업데이트에는 영향을 미치지 않음

loss: 손실함수. 훈련셋과 연관. 훈련에 사용 -> 중요한 값 
metric: 평가지표. 검증셋과 연관. 훈련 과정을 모니터링하는데 사용
=======
import numpy as np
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (569, 30) (569,)

print(y)
# print(y[:10])
print(np.unique(y)) # [0 1] -> 분류값에서 고정이되는 값들

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

# print(y_test[:11])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(50, activation='linear', input_dim=30))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics는 평가지표에 의한 값이 어떤모양으로 돌아가는지 출력하여 보여줌(출력된 loss의 두번째값)

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es]) # callbacks의 []는 다른게 들어갈수 있음


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

resulte = model.predict(x_test[:31])
print(y_test[:31])
print(resulte)


'''
364/364 [==============================] - 0s 635us/step - loss: 0.2439 - accuracy: 0.9093 - val_loss: 0.3178 - val_accuracy: 0.9121
Restoring model weights from the end of the best epoch.
Epoch 00039: early stopping
4/4 [==============================] - 0s 0s/step - loss: 0.2379 - accuracy: 0.9211
loss :  [0.23793119192123413, 0.9210526347160339]
[1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 1 0 1 0 0 1 0 1 0 1 1 0 1 1 1 1]
'''

### NOTE sigmoid, accracy
'''
[평가지표 'metrics']
evaluate 했을때 출력되는 list의 첫번째 값은 'loss'값, 두번째 값은 'metrics'의 평가값
모델 가중치의 업데이트에는 영향을 미치지 않음

loss: 손실함수. 훈련셋과 연관. 훈련에 사용 -> 중요한 값 
metric: 평가지표. 검증셋과 연관. 훈련 과정을 모니터링하는데 사용

>>>>>>> 73d234a7d949bc694f909a16b5ad8b6eb925c0d3
'''