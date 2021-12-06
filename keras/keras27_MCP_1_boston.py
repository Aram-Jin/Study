from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

#print(x)
#print(y)
#print(x.shape)  # (506, 13)
#print(y.shape)  # (506, )
#print(np.min(x), np.max(x))   # 0.0 711.0
#x = x/711.  # 부동소수점이라 .을 사용
#x = x/np.max(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

#scaler = MinMaxScaler()
scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(13,))
dense1 = Dense(6)(input1)
dense2 = Dense(11)(dense1)
dense3 = Dense(24)(dense2)
dense4 = Dense(5, activation='relu')(dense3)
dense5 = Dense(80)(dense4)
dense6 = Dense(100)(dense5)
dense7 = Dense(80)(dense6)
dense8 = Dense(50)(dense7)
dense9 = Dense(25, activation='relu')(dense8)
dense10 = Dense(12)(dense9)
dense11 = Dense(5)(dense10)
dense12 = Dense(2)(dense11)
output1 = Dense(1)(dense12)
model = Model(inputs=input1, outputs=output1)
#model.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 13)]              0
_________________________________________________________________
dense (Dense)                (None, 6)                 84
_________________________________________________________________
dense_1 (Dense)              (None, 11)                77
_________________________________________________________________
dense_2 (Dense)              (None, 24)                288
_________________________________________________________________
dense_3 (Dense)              (None, 5)                 125
_________________________________________________________________
dense_4 (Dense)              (None, 80)                480
_________________________________________________________________
dense_5 (Dense)              (None, 100)               8100
_________________________________________________________________
dense_6 (Dense)              (None, 80)                8080
_________________________________________________________________
dense_7 (Dense)              (None, 50)                4050
_________________________________________________________________
dense_8 (Dense)              (None, 25)                1275
_________________________________________________________________
dense_9 (Dense)              (None, 12)                312
_________________________________________________________________
dense_10 (Dense)             (None, 5)                 65
_________________________________________________________________
dense_11 (Dense)             (None, 2)                 12
_________________________________________________________________
dense_12 (Dense)             (None, 1)                 3
=================================================================
Total params: 22,951
Trainable params: 22,951
Non-trainable params: 0
_________________________________________________________________
'''

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
###################################################################################################################################
import datetime
date = datetime.datetime.now()
datetime_spot = date.strftime("%m%d_%H%M")  # 1206_0456
#print(datetime_spot)   
filepath = './_ModelCheckPoint/'                 # ' ' -> 문자열 형태
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'     # epoch:04d -> 네자리수;천단위,  val_loss:.4f -> 소수점뒤로 0네개  #2500-0.3724
model_path = "".join([filepath, 'k27_1_', datetime_spot, '_', filename])
             # ./_ModelCheckPoint/k27_1_1206_0456_2500-0.3724.hdf5
####################################################################################################################################

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, filepath=model_path)   # EarlyStopping의 patience를 넓게 주어야 효과가 좋음. verbose=1은 중간중간 저장될때마다 보여줌


model.fit(x_train, y_train, epochs=500, batch_size=8, validation_split=0.2, callbacks=[es, mcp])  

model.save("./_save/keras27_1_save_model.h5")  

#4. 평가, 예측

print("================================== 1. 기본 출력  =========================================")
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


print("================================ 2. load_model 출력  =========================================")
model2 = load_model('./_save/keras27_1_save_model.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss :',loss2)

y_predict2 = model2.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2)
print('r2스코어 : ', r2)


'''
Epoch 00127: val_loss did not improve from 16.98055
Epoch 00127: early stopping
================================== 1. 기본 출력  =========================================
5/5 [==============================] - 0s 504us/step - loss: 13.4668
loss : 13.466841697692871
r2스코어 :  0.836996918245313
================================ 2. load_model 출력  =========================================
5/5 [==============================] - 0s 0s/step - loss: 13.4668
loss : 13.466841697692871
r2스코어 :  0.836996918245313

'''