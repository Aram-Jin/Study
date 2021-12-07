from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_wine()
#print(datasets.DESCR)
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (178, 13) (178,)
#print(y)
#print(np.unique(y))  # [0 1 2]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
#print(y)
#print(y.shape)  # (178, 3)
  
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#print(x_train.shape, y_train.shape)  # (142, 13) (142, 3)
#print(x_test.shape, y_test.shape)  # (36, 13) (36, 3)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout

input1 = Input(shape=(13,))
dense1 = Dense(30)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(20, activation='relu')(drop1)
dense3 = Dense(10)(dense2)
drop2 = Dropout(0.1)(dense3)
output1 = Dense(3, activation='softmax')(drop2)
model = Model(inputs=input1, outputs=output1)
#model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 30)                420
_________________________________________________________________
dense_1 (Dense)              (None, 20)                620
_________________________________________________________________
dense_2 (Dense)              (None, 10)                210
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 33
=================================================================
Total params: 1,283
Trainable params: 1,283
Non-trainable params: 0
'''

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics는 평가지표에 의한 값이 어떤모양으로 돌아가는지 출력하여 보여줌(출력된 loss의 두번째값)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
###################################################################################################################################
import datetime
date = datetime.datetime.now()
datetime_spot = date.strftime("%m%d_%H%M")  # 1206_0456
#print(datetime_spot)   
filepath = './_ModelCheckPoint/'                 # ' ' -> 문자열 형태
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'     # epoch:04d -> 네자리수;천단위,  val_loss:.4f -> 소수점뒤로 0네개  #2500-0.3724
model_path = "".join([filepath, 'k28_5_', datetime_spot, '_', filename])
####################################################################################################################################
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=model_path)

model.fit(x_train, y_train, epochs=500, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es, mcp]) # callbacks의 []는 다른게 들어갈수 있음

model.save("./_save/keras28_5_save_wine.h5")  

#4. 평가, 예측
print("================================== 1. 기본 출력  =========================================")
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])



print("================================ 2. load_model 출력  =========================================")
model2 = load_model('./_save/keras28_5_save_wine.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss : ',loss2[0])
print('accuracy : ', loss2[1])



'''
Epoch 00317: val_loss did not improve from 0.00568
Epoch 00317: early stopping
================================== 1. 기본 출력  =========================================
2/2 [==============================] - 0s 0s/step - loss: 0.0198 - accuracy: 0.9722
loss :  0.019827213138341904
accuracy :  0.9722222089767456
================================ 2. load_model 출력  =========================================
2/2 [==============================] - 0s 989us/step - loss: 0.0198 - accuracy: 0.9722
loss :  0.019827213138341904
accuracy :  0.9722222089767456

//////////////////////////////////// Dropout 적용 후 결과 //////////////////////////////////////////////

Epoch 00230: val_loss did not improve from 0.00798
Epoch 00230: early stopping
================================== 1. 기본 출력  =========================================
2/2 [==============================] - 0s 0s/step - loss: 0.0130 - accuracy: 1.0000
loss :  0.012995487079024315
accuracy :  1.0
================================ 2. load_model 출력  =========================================
2/2 [==============================] - 0s 0s/step - loss: 0.0130 - accuracy: 1.0000
loss :  0.012995487079024315
accuracy :  1.0

'''

