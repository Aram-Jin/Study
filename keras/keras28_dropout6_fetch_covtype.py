from sklearn import datasets
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = fetch_covtype()
#print(datasets.DESCR)
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (581012, 54) (581012,)
#print(y)
#print(np.unique(y))  # [1 2 3 4 5 6 7] -> unique : 특성이 있는 값만 나타내줌(결측치 제외)

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

#print(x_train.shape, y_train.shape)  # (464809, 54) (464809, 7)
#print(x_test.shape, y_test.shape)  # (116203, 54) (116203, 7) 

#scaler = MinMaxScaler()
scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout

input1 = Input(shape=(54,))
dense1 = Dense(50, activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(20)(drop1)
dense3 = Dense(10)(dense2)
output1 = Dense(7, activation='softmax')(dense3)
model = Model(inputs=input1, outputs=output1)
'''
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=54))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(7, activation='softmax'))
'''
#model.summary()
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

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
##################################################################################################################################
import datetime
date = datetime.datetime.now()
datetime_spot = date.strftime("%m%d_%H%M")  # 1206_0456
#print(datetime_spot)   
filepath = './_ModelCheckPoint/'                 # ' ' -> 문자열 형태
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'     # epoch:04d -> 네자리수;천단위,  val_loss:.4f -> 소수점뒤로 0네개  #2500-0.3724
model_path = "".join([filepath, 'k28_6_', datetime_spot, '_', filename])
####################################################################################################################################
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=model_path)

model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=1, validation_split=0.2, callbacks=[es, mcp]) # callbacks의 []는 다른게 들어갈수 있음

model.save("./_save/keras28_6_save_fetch_covtype.h5")  


#4. 평가, 예측
print("================================== 1. 기본 출력  =========================================")
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])

print("================================ 2. load_model 출력  =========================================")
model2 = load_model('./_save/keras28_6_save_fetch_covtype.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss : ',loss2[0])
print('accuracy : ', loss2[1])


'''
Epoch 00037: val_loss did not improve from 0.41525
Epoch 00037: early stopping
================================== 1. 기본 출력  =========================================
3632/3632 [==============================] - 2s 455us/step - loss: 0.4163 - accuracy: 0.8314
loss :  0.4162961542606354
accuracy :  0.831364095211029

================================ 2. load_model 출력  =========================================
3632/3632 [==============================] - 2s 445us/step - loss: 0.4163 - accuracy: 0.8314
loss :  0.4162961542606354
accuracy :  0.831364095211029


//////////////////////////////////// Dropout 적용 후 결과 //////////////////////////////////////////////

Epoch 00056: val_loss did not improve from 0.47259
Epoch 00056: early stopping
================================== 1. 기본 출력  =========================================
3632/3632 [==============================] - 2s 463us/step - loss: 0.4719 - accuracy: 0.7981
loss :  0.4718588888645172
accuracy :  0.7980602979660034
================================ 2. load_model 출력  =========================================
3632/3632 [==============================] - 2s 454us/step - loss: 0.4719 - accuracy: 0.7981
loss :  0.4718588888645172
accuracy :  0.7980602979660034

'''
