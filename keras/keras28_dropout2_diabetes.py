from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

#print(x.shape)  # (442, 10) (442,)
#print(y.shape)   # (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
#scaler.fit(x_train)
#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout

input1 = Input(shape=(10,))
dense1 = Dense(100)(input1)  
dense2 = Dense(100)(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(100)(drop1)
dense4 = Dense(50)(dense3)
dense5 = Dense(10, activation='relu')(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint    # ModelCheckpoint는 EarlyStopping과 함께 쓰는 것이 효과적임
###################################################################################################################################
import datetime
date = datetime.datetime.now()
datetime_spot = date.strftime("%m%d_%H%M")  # 1206_0456
#print(datetime_spot)   
filepath = './_ModelCheckPoint/'                 # ' ' -> 문자열 형태
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'     # epoch:04d -> 네자리수;천단위,  val_loss:.4f -> 소수점뒤로 0네개  #2500-0.3724
model_path = "".join([filepath, 'k28_2_', datetime_spot, '_', filename])
####################################################################################################################################

es = EarlyStopping(monitor='val_loss', patience=300, mode='min', verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath= model_path)

model.fit(x_train, y_train, epochs=10000, batch_size=5, 
                 validation_split=0.2, callbacks=[es, mcp])

model.save("./_save/keras28_2_save_diabetes.h5")  


#4. 평가, 예측
print("================================== 1. 기본 출력  =========================================")
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


print("================================ 2. load_model 출력  =========================================")
model2 = load_model('./_save/keras28_2_save_diabetes.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss :',loss2)

y_predict2 = model2.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2)
print('r2스코어 : ', r2)


'''
Epoch 00313: val_loss did not improve from 3009.89185
Epoch 00313: early stopping
================================== 1. 기본 출력  =========================================
3/3 [==============================] - 0s 997us/step - loss: 3512.6604
loss : 3512.660400390625
r2스코어 :  0.4587613608780152
================================ 2. load_model 출력  =========================================
3/3 [==============================] - 0s 997us/step - loss: 3512.6604
loss : 3512.660400390625
r2스코어 :  0.4587613608780152

//////////////////////////////// Dropout 적용 후 결과 /////////////////////////////////////////////

Epoch 00326: val_loss did not improve from 3005.27417
Epoch 00326: early stopping
================================== 1. 기본 출력  =========================================
3/3 [==============================] - 0s 984us/step - loss: 3255.8293
loss : 3255.829345703125
r2스코어 :  0.49833447972566425
================================ 2. load_model 출력  =========================================
3/3 [==============================] - 0s 0s/step - loss: 3255.8293
loss : 3255.829345703125
r2스코어 :  0.49833447972566425

'''