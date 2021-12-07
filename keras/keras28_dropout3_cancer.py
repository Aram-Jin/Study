from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
#print(datasets.feature_names)

x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (569, 30) (569,)

#print(y)
# print(y[:10])
print(np.unique(y)) # [0 1] -> 분류값에서 고정이되는 값들

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(y_test[:11])

#2. 모델구성
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout

input1 = Input(shape=(30,))
dense1 = Dense(50, activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(10)(drop1)
dense3 = Dense(10)(dense2)
output1 = Dense(1, activation='sigmoid')(dense3)
model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics는 평가지표에 의한 값이 어떤모양으로 돌아가는지 출력하여 보여줌(출력된 loss의 두번째값)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint    # ModelCheckpoint는 EarlyStopping과 함께 쓰는 것이 효과적임
###################################################################################################################################
import datetime
date = datetime.datetime.now()
datetime_spot = date.strftime("%m%d_%H%M")  # 1206_0456
#print(datetime_spot)   
filepath = './_ModelCheckPoint/'                 # ' ' -> 문자열 형태
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'     # epoch:04d -> 네자리수;천단위,  val_loss:.4f -> 소수점뒤로 0네개  #2500-0.3724
model_path = "".join([filepath, 'k28_3_', datetime_spot, '_', filename])
####################################################################################################################################

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath= model_path)

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es, mcp]) # callbacks의 []는 다른게 들어갈수 있음

model.save("./_save/keras28_3_save_cancer.h5")  

#4. 평가, 예측
print("================================== 1. 기본 출력  =========================================")
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict(x_test[:31])

print("================================ 2. load_model 출력  =========================================")
model2 = load_model('./_save/keras28_3_save_cancer.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss :',loss2)

y_predict2 = model2.predict(x_test[:31])


'''
Epoch 00018: val_loss did not improve from 0.01950
Epoch 00018: early stopping
================================== 1. 기본 출력  =========================================
4/4 [==============================] - 0s 871us/step - loss: 0.1864 - accuracy: 0.9386
loss : [0.1864270567893982, 0.9385964870452881]
================================ 2. load_model 출력  =========================================
4/4 [==============================] - 0s 665us/step - loss: 0.1864 - accuracy: 0.9386
loss : [0.1864270567893982, 0.9385964870452881]

//////////////////////////////// Dropout 적용 후 결과 /////////////////////////////////////////////

Epoch 00027: val_loss did not improve from 0.01064
Epoch 00027: early stopping
================================== 1. 기본 출력  =========================================
4/4 [==============================] - 0s 0s/step - loss: 0.2248 - accuracy: 0.9649
loss : [0.22479061782360077, 0.9649122953414917]
================================ 2. load_model 출력  =========================================
4/4 [==============================] - 0s 3ms/step - loss: 0.2248 - accuracy: 0.9649
loss : [0.22479061782360077, 0.9649122953414917]
'''

