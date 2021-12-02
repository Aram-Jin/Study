from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (569, 30) (569,)

print(y)
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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(30,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(10)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1, activation='sigmoid')(dense3)
model = Model(inputs=input1, outputs=output1)
'''
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=30))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))
'''
model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 50)                1550
_________________________________________________________________
dense_1 (Dense)              (None, 10)                510
_________________________________________________________________
dense_2 (Dense)              (None, 10)                110
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 11
=================================================================
Total params: 2,181
Trainable params: 2,181
Non-trainable params: 0
'''

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # metrics는 평가지표에 의한 값이 어떤모양으로 돌아가는지 출력하여 보여줌(출력된 loss의 두번째값)

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es]) # callbacks의 []는 다른게 들어갈수 있음


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

resulte = model.predict(x_test[:31])
#print(y_test[:31])
#print(resulte)

''' 
=========================================== Scaler만 적용 후 결과 비교 =============================================================
1. No Scaler
4/4 [==============================] - 0s 0s/step - loss: 0.2536 - accuracy: 0.9123
loss :  [0.2535550594329834, 0.9122806787490845]

2. MinMaxScaler
4/4 [==============================] - 0s 986us/step - loss: 0.0991 - accuracy: 0.9649
loss :  [0.09907896816730499, 0.9649122953414917]

3. StandardScaler
4/4 [==============================] - 0s 998us/step - loss: 0.0779 - accuracy: 0.9737
loss :  [0.07791540771722794, 0.9736841917037964]

☆ 4. RobustScaler -> ok
4/4 [==============================] - 0s 988us/step - loss: 0.0695 - accuracy: 0.9737
loss :  [0.06952772289514542, 0.9736841917037964]

5. MaxAbsScaler 
4/4 [==============================] - 0s 5ms/step - loss: 0.0920 - accuracy: 0.9737
loss :  [0.0920235812664032, 0.9736841917037964]
================================================ Activation = 'relu' 추가 적용 후 TEST================================================= 

1. No Scaler (dense에 activation='relu' 적용) 
4/4 [==============================] - 0s 0s/step - loss: 0.2496 - accuracy: 0.9123
loss :  [0.2495831549167633, 0.9122806787490845]

2. MinMaxScaler (dense에 activation='relu' 적용) 
4/4 [==============================] - 0s 770us/step - loss: 0.1546 - accuracy: 0.9737
loss :  [0.15462443232536316, 0.9736841917037964]

3. StandardScaler (dense에 activation='relu' 적용) 
4/4 [==============================] - 0s 957us/step - loss: 0.5543 - accuracy: 0.9649
loss :  [0.5543385148048401, 0.9649122953414917]

☆ 4. RobustScaler (dense에 activation='relu' 적용) 
4/4 [==============================] - 0s 0s/step - loss: 0.1050 - accuracy: 0.9737
loss :  [0.10496465116739273, 0.9736841917037964]

5. MaxAbsScaler (dense에 activation='relu' 적용) 
4/4 [==============================] - 0s 0s/step - loss: 0.1387 - accuracy: 0.9737
loss :  [0.1387193202972412, 0.9736841917037964]


==> 결론 : "cancer"데이터는 RobustScaler로 돌렸을때 가장 효과가 좋았으며, activation='relu' 적용했을때 loss값이 개선되지 않음

=============================================== input_shape 모델로 튜닝 후 TEST =======================================================

< RobustScaler & dense에 activation='relu' 적용  >

4/4 [==============================] - 0s 0s/step - loss: 0.1156 - accuracy: 0.9561
loss :  [0.11562397330999374, 0.9561403393745422]

==>> 별로 차이 없는듯 

'''


##################################################  [ N O T E ]  #################################################################
''' sigmoid, accracy
[평가지표 'metrics']
evaluate 했을때 출력되는 list의 첫번째 값은 'loss'값, 두번째 값은 'metrics'의 평가값
모델 가중치의 업데이트에는 영향을 미치지 않음

loss: 손실함수. 훈련셋과 연관. 훈련에 사용 -> 중요한 값 
metric: 평가지표. 검증셋과 연관. 훈련 과정을 모니터링하는데 사용

'''