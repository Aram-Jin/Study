from enum import auto
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1. 데이터
path = "../_data/dacon/wine/"
train = pd.read_csv(path + 'train.csv')
#print(train.shape)  # (3231, 14)

test_file = pd.read_csv(path + 'test.csv')
#print(test_file.shape)  # (3231, 13)

submit_file = pd.read_csv(path + 'sample_submission.csv')  
#print(submit_file.shape)  # (3231, 2)
#print(submit_file.columns)  # ['id', 'quality'], dtype='object

#print(type(train))  # <class 'pandas.core.frame.DataFrame'>
#print(train.columns)
#Index(['id', 'fixed acidity', 'volatile acidity', 'citric acid',
#       'residual sugar', 'chlorides', 'free sulfur dioxide',
#       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'type',
#       'quality'],
#      dtype='object')

x = train.drop(['id', 'quality'], axis=1)   # axis값을 1을 주면 열(세로) 값을 0을주면 행(가로) 삭제함. default값은 0
test_file = test_file.drop(['id'], axis=1)
y = train['quality']

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
label = x['type']
le.fit(label)
x['type'] = le.transform(label)

label2 = test_file['type']
le.fit(label2) 
test_file['type'] = le.transform(label2)

#print(test_file['type'])       # testfile의 type열의 값이 0,1로 바뀌어있는지 확인해봄.

#print(np.unique(y))  # [4 5 6 7 8]

y = pd.get_dummies(y)  # pd.get_dummies 처리 : 결측값을 제외하고 0과 1로 구성된 더미값이 만들어진다. 
# 결측값 처리(dummy_na = True 옵션) : Nan을 생성하여 결측값도 인코딩하여 처리해준다.
# y = pd.get_dummies(y, drop_first=True) : N-1개의 열을 생성
#print(y.shape)   # (3231, 5) -> y가 5개의 열로 바뀜(quality가 원핫인코딩으로 0,1,2,3,4 다섯개의 열이 됨)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=49)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)

#2. 모델구성
input1 = Input(shape=(12,))
dense1 = Dense(100, activation='relu')(input1)
dense2 = Dense(60, activation='relu')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(10, activation='relu')(dense3)
output1 = Dense(5, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

##################################################################################################################################
import datetime
date = datetime.datetime.now()
datetime_spot = date.strftime("%m%d_%H%M")  # 1206_0456
#print(datetime_spot)   
filepath = './_ModelCheckPoint/'                 # ' ' -> 문자열 형태
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'     # epoch:04d -> 네자리수;천단위,  val_loss:.4f -> 소수점뒤로 0네개  #2500-0.3724
model_path = "".join([filepath, 'k27_8_', datetime_spot, '_', filename])
####################################################################################################################################

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath= model_path)
model.fit(x_train, y_train, epochs=10000, batch_size=5, validation_split=0.2, callbacks=[es, mcp])

model.save("./_save/keras27_8_save_dacon_wine.h5") 

#4. 평가, 예측
print("================================== 1. 기본 출력  =========================================")
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])

print("================================ 2. load_model 출력  =========================================")
model2 = load_model('./_save/keras27_8_save_dacon_wine.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss : ',loss2[0])
print('accuracy : ', loss2[1])


########################### 제출용 제작 ################################
results = model.predict(test_file)

#print(results)

results_int = np.argmax(results, axis=1).reshape(-1,1) + 4

#print(results_int)

submit_file['quality'] = results_int

#submit_file.to_csv(path+'subfile.csv', index=False)
      
acc = str(round(loss[1],4)).replace(".","_")
submit_file.to_csv(path +f"result/accuracy_{acc}.csv", index=False)

'''
Epoch 00311: val_loss did not improve from 1.06175
Epoch 00311: early stopping
================================== 1. 기본 출력  =========================================
21/21 [==============================] - 0s 498us/step - loss: 1.0467 - accuracy: 0.5626
loss :  1.0466537475585938
accuracy :  0.5625966191291809
================================ 2. load_model 출력  =========================================
21/21 [==============================] - 0s 549us/step - loss: 1.0467 - accuracy: 0.5626
loss :  1.0466537475585938
accuracy :  0.5625966191291809
'''