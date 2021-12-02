from sklearn import datasets
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = fetch_covtype()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (581012, 54) (581012,)
print(y)
print(np.unique(y))  # [1 2 3 4 5 6 7] -> unique : 특성이 있는 값만 나타내줌(결측치 제외)

# output 컬럼의 갯수는 8개인데 unique를 뽑아보니 7개가 나옴. 특성이 없는 dummy(결측치)가 1개 있다는 의미
# ONE-HOT-ENCODING을 통해서 output 갯수 조정 (아래 ONE-HOT-ENCODING 방법)

'''
1) tensorflow를 이용하여 ONE-HOT-ENCODING -> output 갯수에 맞추어 결측치 제거없이 하는 방법, 결과값 y.shape이 8개 나옴
from tensorflow.keras.utils import to_categorical  # ONE-HOT-ENCODING
y = to_categorical(y)
print(y)
print(y.shape)  # (581012, 8)
'''

#2) sklearn을 이용하여 ONE-HOT-ENCODING
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y.reshape(-1,1)) #해당 data 를 사용하여 tuple 의 형태 구축(사실 sparse matrix 로 나옴)>> 1:(1,0,0), 2:(0,1,0) 3:(0,0,1) 로 매핑될 수 있게 fitting 해두는 것

'''
3) pandas를 이용하여 ONE-HOT-ENCODING
import pandas as pd
y = pd.get_dummies(y)  # pd.get_dummies 처리 : 결측값을 제외하고 0과 1로 구성된 더미값이 만들어진다. 
# 결측값 처리(dummy_na = True 옵션) : Nan을 생성하여 결측값도 인코딩하여 처리해준다.
# y = pd.get_dummies(y, drop_first=True) : N-1개의 열을 생성
print(y.shape)
'''
# 이진분류도 다중분류로 처리가능.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)  # (464809, 54) (464809, 7)
print(x_test.shape, y_test.shape)  # (116203, 54) (116203, 7) 

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(50, activation='relu', input_dim=54))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(7, activation='softmax'))

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 50)                2750
_________________________________________________________________
dense_1 (Dense)              (None, 20)                1020
_________________________________________________________________
dense_2 (Dense)              (None, 10)                210
_________________________________________________________________
dense_3 (Dense)              (None, 7)                 77
=================================================================
Total params: 4,057
Trainable params: 4,057
Non-trainable params: 0
'''

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics는 평가지표에 의한 값이 어떤모양으로 돌아가는지 출력하여 보여줌(출력된 loss의 두번째값)

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=1, validation_split=0.2, callbacks=[es]) # callbacks의 []는 다른게 들어갈수 있음


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])

resulte = model.predict(x_test[:7])
print(y_test[:7])
print(resulte)


''' 
=========================================== Scaler만 적용 후 결과 비교 =============================================================
1. No Scaler
loss :  0.6479116082191467
accuracy :  0.720317006111145

2. MinMaxScaler
loss :  0.6336530447006226
accuracy :  0.7233117818832397

☆ 3. StandardScaler
loss :  0.6336315870285034
accuracy :  0.7247575521469116

☆ 4. RobustScaler -> ok
loss :  0.6319603323936462
accuracy :  0.7255750894546509

5. MaxAbsScaler 
loss :  0.6338685750961304
accuracy :  0.7215132117271423

================================================ Activation = 'relu' 추가 적용 후 TEST================================================= 
1. No Scaler (dense_1에 activation='relu' 적용) 
loss :  0.5363616943359375
accuracy :  0.7704878449440002

2. MinMaxScaler (dense_1에 activation='relu' 적용) 
loss :  0.4356505274772644
accuracy :  0.8147724270820618

☆ 3. StandardScaler (dense_1에 activation='relu' 적용) 
loss :  0.3987372815608978
accuracy :  0.8360713720321655

☆ 4. RobustScaler (dense_1에 activation='relu' 적용) 
loss :  0.3987893760204315
accuracy :  0.8330765962600708

5. MaxAbsScaler (dense에 activation='relu' 적용) 
loss :  0.44970643520355225
accuracy :  0.8105986714363098
==> 결론 : "fetch_covtype"데이터는 StandardScaler, RobustScaler 돌렸을때 가장 효과가 좋았으며, activation='relu' 적용했을때 loss값이 크게 개선됨. 효과가 있음!!! 

'''









'''
Epoch 60/100
7437/7437 [==============================] - 5s 625us/step - loss: 0.6565 - accuracy: 0.7148 - val_loss: 0.6453 - val_accuracy: 0.7211
Restoring model weights from the end of the best epoch.
Epoch 00060: early stopping
3632/3632 [==============================] - 2s 427us/step - loss: 0.6448 - accuracy: 0.7214
loss :  0.6448085904121399
accuracy :  0.7213669419288635
[[0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]]
[[6.5158003e-13 6.9950908e-01 2.7308086e-01 5.5235351e-08 3.0022514e-12
  6.4606185e-04 1.4948830e-07 2.6763827e-02]
 [5.3100907e-10 7.0410617e-02 9.2861837e-01 6.0712275e-05 1.0949736e-05
  5.9415243e-04 2.5507639e-04 5.0149836e-05]
 [7.5671343e-13 7.6936799e-01 2.1677803e-01 4.0062818e-07 2.0808801e-10
  1.0680374e-03 2.3477378e-05 1.2762069e-02]
 [1.1070732e-09 6.8759859e-02 9.2618126e-01 1.0889668e-04 9.3934097e-05
  4.1897101e-03 5.1530253e-04 1.5103200e-04]
 [9.9579433e-12 4.2843568e-01 5.5917305e-01 1.8488308e-05 1.5769602e-10
  3.0524875e-03 3.5193960e-05 9.2851333e-03]
 [6.7367622e-12 2.7079758e-01 7.0719385e-01 1.5364105e-05 4.1406794e-09
  2.1669621e-02 2.4355085e-04 8.0088947e-05]
 [1.5816311e-10 1.5733100e-01 8.4114534e-01 1.4786504e-05 8.7535724e-08
  5.4727792e-04 3.8302613e-05 9.2318724e-04]]
'''

# batch_size의 디폴트 값은 32입니다.(batch_size=1로 돌렸을때의 1epoch당 돌아가야하는 값과 디폴트로 놓았을때의 값 비교 계산)
