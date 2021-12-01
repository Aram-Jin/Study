from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (178, 13) (178,)
print(y)
print(np.unique(y))  # [0 1 2]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)  # (178, 3)
  
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)  # (142, 13) (142, 3)
print(x_test.shape, y_test.shape)  # (36, 13) (36, 3)

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
model.add(Dense(30, activation='linear', input_dim=13))
model.add(Dense(20, activation='relu'))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

model.summary()
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

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es]) # callbacks의 []는 다른게 들어갈수 있음


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss[0])
print('accuracy : ', loss[1])

resulte = model.predict(x_test[:7])
print(y_test[:7])
print(resulte)

''' 
=========================================== Scaler만 적용 후 결과 비교 =============================================================
1. No Scaler
loss :  1.3803682327270508
accuracy :  0.6944444179534912

2. MinMaxScaler
loss :  0.15322503447532654
accuracy :  0.9722222089767456

3. StandardScaler
loss :  0.3783946931362152
accuracy :  0.9722222089767456

4. RobustScaler 
loss :  0.3063912093639374
accuracy :  0.9722222089767456

☆ 5. MaxAbsScaler 
loss :  0.06411120295524597
accuracy :  0.9722222089767456

================================================ Activation = 'relu' 추가 적용 후 TEST================================================= 
1. No Scaler (dense_1에 activation='relu' 적용) 
loss :  0.7188189029693604
accuracy :  0.8055555820465088

loss :  1.3856316804885864
accuracy :  0.6388888955116272

2. MinMaxScaler (dense_1에 activation='relu' 적용) 
loss :  0.16183114051818848
accuracy :  0.9722222089767456

3. StandardScaler (dense_1에 activation='relu' 적용) 
loss :  0.17851372063159943
accuracy :  0.9722222089767456

4. RobustScaler (dense_1에 activation='relu' 적용) 
loss :  0.283075749874115
accuracy :  0.9722222089767456

☆ 5. MaxAbsScaler (dense에 activation='relu' 적용) 
loss :  0.03006211295723915
accuracy :  1.0

==> 결론 : "wine"데이터는 MaxAbsScaler 돌렸을때 가장 효과가 좋았으며, activation='relu' 적용했을때 loss값이 개선됨. 효과가 있는 것 같음!!! 

'''




