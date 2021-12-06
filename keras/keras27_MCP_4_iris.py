from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_iris()
#print(datasets.DESCR)
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)  # (150, 4) (150,)
#print(y)
#print(np.unique(y))  # [0 1 2]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
#print(y)
#print(y.shape)  # (150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#print(x_train.shape, y_train.shape)  # (120, 4) (120, 3)
#print(x_test.shape, y_test.shape)  # (30, 4) (30, 3)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(4,))
dense1 = Dense(50)(input1)
dense2 = Dense(10, activation='relu')(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(3, activation='softmax')(dense3)
model = Model(inputs=input1, outputs=output1)
#model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 50)                250
_________________________________________________________________
dense_1 (Dense)              (None, 10)                510
_________________________________________________________________
dense_2 (Dense)              (None, 10)                110
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 33
=================================================================
Total params: 903
Trainable params: 903
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
model_path = "".join([filepath, 'k27_4_', datetime_spot, '_', filename])
####################################################################################################################################

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=model_path)
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es, mcp]) # callbacks의 []는 다른게 들어갈수 있음

model.save("./_save/keras27_4_save_iris.h5")  


#4. 평가, 예측
print("================================== 1. 기본 출력  =========================================")
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])

resulte = model.predict(x_test[:7])
print(y_test[:7])
print(resulte)

print("================================ 2. load_model 출력  =========================================")
model2 = load_model('./_save/keras27_4_save_iris.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss : ',loss2[0])
print('accuracy : ', loss2[1])

resulte = model2.predict(x_test[:7])
print(y_test[:7])
print(resulte)


'''
Epoch 00034: val_loss did not improve from 0.00251
Epoch 00034: early stopping
================================== 1. 기본 출력  =========================================
1/1 [==============================] - 0s 86ms/step - loss: 0.0698 - accuracy: 0.9667
loss :  0.06982432305812836
accuracy :  0.9666666388511658
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[3.9184891e-04 9.9882656e-01 7.8152033e-04]
 [1.1715517e-04 9.6798271e-01 3.1900130e-02]
 [2.6710829e-04 9.4876152e-01 5.0971318e-02]
 [9.9995339e-01 4.6575999e-05 1.3509479e-16]
 [1.7984523e-04 9.9741900e-01 2.4010693e-03]
 [3.3594249e-03 9.9637908e-01 2.6158750e-04]
 [9.9996877e-01 3.1251227e-05 1.3782399e-16]]
 
================================ 2. load_model 출력  =========================================
1/1 [==============================] - 0s 123ms/step - loss: 0.0698 - accuracy: 0.9667
loss :  0.06982432305812836
accuracy :  0.9666666388511658
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[3.9184891e-04 9.9882656e-01 7.8152033e-04]
 [1.1715517e-04 9.6798271e-01 3.1900130e-02]
 [2.6710829e-04 9.4876152e-01 5.0971318e-02]
 [9.9995339e-01 4.6575999e-05 1.3509479e-16]
 [1.7984523e-04 9.9741900e-01 2.4010693e-03]
 [3.3594249e-03 9.9637908e-01 2.6158750e-04]
 [9.9996877e-01 3.1251227e-05 1.3782399e-16]]

'''