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
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(13,))
dense1 = Dense(30)(input1)
dense2 = Dense(20, activation='relu')(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(3, activation='softmax')(dense3)
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
model_path = "".join([filepath, 'k27_5_', datetime_spot, '_', filename])
####################################################################################################################################
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=model_path)

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es, mcp]) # callbacks의 []는 다른게 들어갈수 있음

model.save("./_save/keras27_5_save_wine.h5")  

#4. 평가, 예측
print("================================== 1. 기본 출력  =========================================")
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])

resulte = model.predict(x_test[:7])
print(y_test[:7])
print(resulte)

print("================================ 2. load_model 출력  =========================================")
model2 = load_model('./_save/keras27_5_save_wine.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss : ',loss2[0])
print('accuracy : ', loss2[1])

resulte = model2.predict(x_test[:7])
print(y_test[:7])
print(resulte)


'''
Epoch 00083: val_loss did not improve from 0.03917
Epoch 00083: early stopping
================================== 1. 기본 출력  =========================================
2/2 [==============================] - 0s 2ms/step - loss: 0.0092 - accuracy: 1.0000
loss :  0.009153938852250576
accuracy :  1.0
[[0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[[1.9082179e-11 1.5419393e-05 9.9998462e-01]
 [1.3307913e-06 9.9999869e-01 5.1091170e-10]
 [3.2004660e-10 1.0000000e+00 7.5585673e-16]
 [9.9932528e-01 6.6964276e-04 5.1219658e-06]
 [5.8563985e-04 9.9941432e-01 5.9274939e-08]
 [2.7668063e-09 1.0000000e+00 5.9951528e-14]
 [2.0386191e-09 1.8313545e-05 9.9998164e-01]]
================================ 2. load_model 출력  =========================================
2/2 [==============================] - 0s 969us/step - loss: 0.0092 - accuracy: 1.0000
loss :  0.009153938852250576
accuracy :  1.0
[[0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[[1.9082179e-11 1.5419393e-05 9.9998462e-01]
 [1.3307913e-06 9.9999869e-01 5.1091170e-10]
 [3.2004660e-10 1.0000000e+00 7.5585673e-16]
 [9.9932528e-01 6.6964276e-04 5.1219658e-06]
 [5.8563985e-04 9.9941432e-01 5.9274939e-08]
 [2.7668063e-09 1.0000000e+00 5.9951528e-14]
 [2.0386191e-09 1.8313545e-05 9.9998164e-01]]
'''

