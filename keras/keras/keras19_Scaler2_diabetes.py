from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape)  # (442, 10) (442,)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

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
model.add(Dense(100, input_dim=10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 100)               1100
_________________________________________________________________
dense_1 (Dense)              (None, 100)               10100
_________________________________________________________________
dense_2 (Dense)              (None, 100)               10100
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11
=================================================================
Total params: 26,871
Trainable params: 26,871
Non-trainable params: 0
_________________________________________________________________

'''

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=500, batch_size=1, 
                 validation_split=0.2, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


''' 
=========================================== Scaler만 적용 후 결과 비교 =============================================================
1. No Scaler
loss : 3282.7451171875
r2스코어 :  0.4941873261286269

2. MinMaxScaler
loss : 3595.98486328125
r2스코어 :  0.44592260534774353

3. StandardScaler
loss : 3388.556640625
r2스코어 :  0.477883544655893

☆ 4. RobustScaler -> ok
loss : 3238.885986328125
r2스코어 :  0.5009452174047841

5. MaxAbsScaler 
loss : 3378.660888671875
r2스코어 :  0.47940836872282344
================================================ Activation = 'relu' 추가 적용 후 TEST================================================= 
1. No Scaler
No Scaler (dense_1에 activation='relu' 적용) -> Total params: 26,871. params 갯수가 안줄어듬
loss : 3726.605224609375
r2스코어 :  0.42579627569340106

No Scaler (dense_4에 activation='relu' 적용) 
loss : 3328.349609375
r2스코어 :  0.4871604047985997

No Scaler (dense_1,4에 activation='relu' 적용) 
loss : 4283.560546875
r2스코어 :  0.33997937637602127

2. MinMaxScaler (dense_4에 activation='relu' 적용) 
loss : 3352.708251953125
r2스코어 :  0.4834072088392447

3. StandardScaler (dense_4에 activation='relu' 적용) 
loss : 4164.03564453125
r2스코어 :  0.3583960199504742

4. RobustScaler (dense_4에 activation='relu' 적용) 
loss : 3674.73583984375
r2스코어 :  0.4337884307728206

5. MaxAbsScaler (dense_4에 activation='relu' 적용) 
loss : 3906.833984375
r2스코어 :  0.3980262449716383


==>> 결론 : activation='relu'를 적용해도 성능변화 없음

'''



